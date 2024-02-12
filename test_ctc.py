import gin
import torch
from rich.progress import track
import numpy as np
from rich import progress
from torchvision import transforms
from torch.utils.data import DataLoader
from ModelManager import LighntingE2EModelUnfolding
from data import CTCDataset, batch_preparation_ctc
from eval_functions import compute_poliphony_metrics
from itertools import groupby

@gin.configurable
def main(stage=None, data_path=None, corpus_name=None, model_name=None, metric_to_watch=None):

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = CTCDataset(data_path=f"{data_path}/test.txt")
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=20, collate_fn=batch_preparation_ctc)
    model = LighntingE2EModelUnfolding.load_from_checkpoint("weights/Degraded_Quartets/CRNN_CTC.ckpt")
    model.eval()
    i2w = model.i2w
    idx = 0
    out_path = "out/Degraded_Quartets/CRNN_CTC/"
    w2i = np.load("vocab/QuartetsGlobalw2i.npy", allow_pickle=True).item()
    i2w = np.load("vocab/QuartetsGlobali2w.npy", allow_pickle=True).item()
    test_dataset.set_dictionaries(w2i, i2w)
    for (sample, gt, _, _) in progress.track(test_dataloader):

        #device = torch.device("cpu")
        pred = model(sample.to(device))
        pred = pred.permute(1,0,2).contiguous()
        pred = pred[0]
        out_best = torch.argmax(pred,dim=1)
        out_best = [k for k, g in groupby(list(out_best))]
        decoded = []
        for c in out_best:
            if c.item() != len(i2w):
                decoded.append(c.item())
    
        text_sequence = [i2w[tok] for tok in decoded]
            
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
    gin.parse_config_file("config/StringQuartets/Baseline_dist.gin")
    main()