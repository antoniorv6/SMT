import gin
import torch
from rich.progress import track
import numpy as np
from rich import progress
from torchvision import transforms
from torch.utils.data import DataLoader
from ModelManager import Poliphony_DAN
from data import GrandStaffSingleSystem, batch_preparation_img2seq
from eval_functions import compute_poliphony_metrics

@gin.configurable
def main(stage=None, data_path=None, corpus_name=None, model_name=None, metric_to_watch=None):

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = GrandStaffSingleSystem(data_path=f"{data_path}/test.txt")
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=20, collate_fn=batch_preparation_img2seq)
    model = Poliphony_DAN.load_from_checkpoint("weights/Degraded_Quartets/DAN_Next.ckpt")
    model.eval()
    w2i, i2w = model.w2i, model.i2w
    idx = 0
    out_path = "out/Degraded_Quartets/DAN_Next/"
    test_dataset.set_dictionaries(w2i, i2w)
    for (sample, _, gt) in progress.track(test_dataloader):

        #device = torch.device("cpu")
        encoder_output = model.forward_encoder(sample.to(device))
        predicted_sequence = torch.from_numpy(np.asarray([w2i['<bos>']])).to(device).unsqueeze(0)
        cache = None
        text_sequence = []

        with torch.no_grad():
            for i in range(model.maxlen):
                output, predictions, cache, weights = model.forward_decoder(encoder_output, predicted_sequence.long(), cache=cache)
                predicted_token = torch.argmax(predictions[:, :, -1]).cpu().detach().item()
                predicted_sequence = torch.cat([predicted_sequence, torch.argmax(predictions[:, :, -1], dim=1, keepdim=True)], dim=1)
                predicted_char = i2w[predicted_token]
                if predicted_char == '<eos>':
                    break
                text_sequence.append(predicted_char)
            
        gt_seq = [i2w[token.item()] for token in gt.squeeze(0)[:-2]]
        
        with open(f"{out_path}hyp/{idx}.krn", "w") as predfile:
            text_sequence = "".join(text_sequence).replace("<t>", "\t")
            text_sequence = text_sequence.replace("<b>", "\n")
            text_sequence = text_sequence.replace("<s>", " ").replace('**ekern_1.0', '**kern')
            predfile.write(text_sequence)
        
        with open(f"{out_path}gt/{idx}.krn", "w") as gtfile:
            gt_seq = "".join(gt_seq).replace("<t>", "\t")
            gt_seq = gt_seq.replace("<b>", "\n")
            gt_seq = gt_seq.replace("<s>", " ").replace('**ekern_1.0', '**kern')
            gtfile.write(gt_seq)
        
        idx += 1
    

if __name__ == "__main__":
    gin.parse_config_file("config/StringQuartets/DAN_NexT_Dist.gin")
    main()