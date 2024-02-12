import gin
import torch
import cv2
import numpy as np
from rich import progress
from torchvision import transforms
from visualizer.WeightsVisualizer import DAN_WeightsVisualizer
from ModelManager import LighntingE2EModelUnfolding
from data import load_grandstaff_singleSys
from eval_functions import compute_poliphony_metrics
from itertools import groupby

@gin.configurable
def main(stage=None, data_path=None, corpus_name=None, model_name=None, metric_to_watch=None):

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx = 0
    visualizer_module = DAN_WeightsVisualizer(frames_path="frames/", animation_path="animation/")

    img = cv2.imread("op71n3-03.025_distorted.png", 3)
    
    width = int(np.ceil(img.shape[1] * 0.5))
    height = int(np.ceil(img.shape[0] * 0.5))

    img = cv2.resize(img, (width, height))
    
    transform = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor()]
    )
    tensor_img = transform(img)
    
    seq = ""
    with open("op71n3-03.025.krn") as krnfile:
        seq = krnfile.read()
    
    gtseq = seq

    #seq = "".join(seq[1:-1])
    seq = seq.replace(" ", "\t-\t")
    seq = seq.replace("·", "")
    seq = seq.replace("\t", "\t--\t")
    seq = seq.replace("\n", "\t¶\n")
    seq = seq.expandtabs(2)

    text_sequence = []
    global_weights = []
    self_weights = []
    model = LighntingE2EModelUnfolding.load_from_checkpoint("weights/Degraded_Quartets/CRNN_CTC.ckpt")
    model.eval()

    i2w = model.i2w
    
    pred = model(tensor_img.unsqueeze(0).to(device))
    pred = pred.permute(1,0,2).contiguous()
    pred = pred[0]
    out_best = torch.argmax(pred,dim=1)
    out_best = [k for k, g in groupby(list(out_best))]
    decoded = []
    for c in out_best:
        if c.item() != len(i2w):
            decoded.append(c.item())
    
    text_sequence = [i2w[tok] for tok in decoded]
    
    
    with open("prediction_ctc.krn", "w") as predfile:
        text_sequence = "".join(text_sequence).replace("<t>", "\t")
        text_sequence = text_sequence.replace("<b>", "\n")
        text_sequence = text_sequence.replace("<s>", " ")
        predfile.write(text_sequence)
    
    gtseq = gtseq.replace("<t>", "\t").replace("<b>", "\n").replace("<s>", " ")
    print(compute_poliphony_metrics([text_sequence],[gtseq]))

if __name__ == "__main__":
    gin.parse_config_file("config/StringQuartets/DAN_B_Dist.gin")
    main()