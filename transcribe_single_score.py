import gin
import torch
import cv2
import fire
import numpy as np
from rich import progress
from torchvision import transforms
from ModelManager import SMT
from data import load_grandstaff_singleSys
from eval_functions import compute_poliphony_metrics

@gin.configurable
def main(stage=None, data_path=None, corpus_name=None, model_name=None, metric_to_watch=None, sample_image=None, model_weights=None):

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = cv2.imread(sample_image, 3)
    
    if img.shape[1] > 3056:
        width = int(np.ceil(3056 * 0.85))
        height = int(np.ceil(max(img.shape[0], 256) * 0.85))
    else:
        width = int(np.ceil(img.shape[1] * 0.85))
        height = int(np.ceil(img.shape[0] * 0.85))

    img = cv2.resize(img, (width, height))
    
    transform = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor()]
    )
    tensor_img = transform(img)

    text_sequence = []
    global_weights = []
    self_weights = []
    model = SMT.load_from_checkpoint(model_weights)
    model.eval()

    w2i, i2w = model.w2i, model.i2w
    encoder_output = model.forward_encoder(tensor_img.unsqueeze(0).to(device))
    predicted_sequence = torch.from_numpy(np.asarray([w2i['<bos>']])).to(device).unsqueeze(0)
    cache = None

    with torch.no_grad():
        for i in progress.track(range(2048)):
            output, predictions, cache, weights = model.forward_decoder(encoder_output, predicted_sequence.long(), cache=cache)
            predicted_token = torch.argmax(predictions[:, :, -1]).cpu().detach().item()
            predicted_sequence = torch.cat([predicted_sequence, torch.argmax(predictions[:, :, -1], dim=1, keepdim=True)], dim=1)
            predicted_char = i2w[predicted_token]
            if predicted_char == '<eos>':
                break
            text_sequence.append(predicted_char)
    
    with open("prediction.krn", "w") as predfile:
        text_sequence = "".join(text_sequence).replace("<t>", "\t")
        text_sequence = text_sequence.replace("<b>", "\n")
        text_sequence = text_sequence.replace("<s>", " ").replace('**ekern_1.0', '**kern')
        predfile.write(text_sequence)

def launch(config, sample_image, model_weights):
    gin.parse_config_file(config)
    main(sample_image=sample_image, model_weights=model_weights)

if __name__ == "__main__":
    fire.Fire(launch)