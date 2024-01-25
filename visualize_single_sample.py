import gin
import torch
import cv2
import numpy as np
from rich import progress
from torchvision import transforms
from visualizer.WeightsVisualizer import DAN_WeightsVisualizer
from ModelManager import Poliphony_DAN
from data import load_grandstaff_singleSys

@gin.configurable
def main(stage=None, data_path=None, corpus_name=None, model_name=None, metric_to_watch=None):

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx = 0
    visualizer_module = DAN_WeightsVisualizer(frames_path="frames/", animation_path="animation/")

    img = cv2.imread("original_m-66-69_distorted.jpg", 3)
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
    
    seq = ""
    with open("original_m-66-69.krn") as krnfile:
        seq = krnfile.read()

    #seq = "".join(seq[1:-1])
    seq = seq.replace(" ", "\t-\t")
    seq = seq.replace("·", "")
    seq = seq.replace("\t", "\t--\t")
    seq = seq.replace("\n", "\t¶\n")
    seq = seq.expandtabs(2)

    text_sequence = []
    global_weights = []
    self_weights = []
    model = Poliphony_DAN.load_from_checkpoint("weights/Camera_GrandStaff/DAN_B.ckpt")
    model.eval()

    w2i, i2w = model.w2i, model.i2w
    #device = torch.device("cpu")
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
            weights["self"] = [torch.reshape(w.unsqueeze(1).cpu(), (-1, 1, encoder_output.shape[2], encoder_output.shape[3])) for w in weights["self"]]
            weights_append = weights["self"][-1]
            self_weights.append(weights["mix"][-1])
            global_weights.append(weights_append[i])
            text_sequence.append(predicted_char)

    attention_weights = global_weights
    attention_weights = np.stack(attention_weights, axis=0)
    attention_weights = torch.tensor(attention_weights).squeeze(1).detach().numpy()
    zero_weights = np.zeros((1, attention_weights.shape[1], attention_weights.shape[2]))
    attention_weights = np.concatenate([zero_weights, attention_weights, zero_weights], axis=0)
    #with open("prediction.krn", "w") as predfile:
    #    text_sequence = "".join(text_sequence).replace("<t>", "\t")
    #    text_sequence = text_sequence.replace("<b>", "\n")
    #    text_sequence = text_sequence.replace("<s>", " ")
    #    predfile.write(text_sequence)

    #try:
    visualizer_module.render(x=img, y=seq, predicted_seq=text_sequence, self_weights=self_weights, attn_weights=attention_weights, animation_name=str(idx))
    #except:
    #    logger.error(f"Error rendering {idx} video")
    idx += 1

if __name__ == "__main__":
    gin.parse_config_file("config/GrandStaff/DAN_B_Dist.gin")
    main()