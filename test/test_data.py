import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from data import batch_preparation_img2seq, BatchCollator

def test_batch_padding():
    # Simulate batch data
    # format: [(image, decoder_input, y), ...]
    # image: tensor of shape (1, H, W)
    # decoder_input: tensor of seq len (includes <bos> and <eos>)
    # y: same as decoder_input
    
    pad_token = 99
    collator = BatchCollator(pad_token=pad_token)
    
    dec_in_1 = torch.tensor([1, 2, 3, 4, 5]) # e.g. <bos>, a, b, c, <eos>
    dec_in_2 = torch.tensor([1, 2, 5])       # e.g. <bos>, a, <eos>
    
    gt_1 = dec_in_1.clone()
    gt_2 = dec_in_2.clone()
    
    img_1 = torch.ones(1, 100, 100)
    img_2 = torch.ones(1, 50, 50)
    
    data = [
        (img_1, dec_in_1, gt_1),
        (img_2, dec_in_2, gt_2)
    ]
    
    X_train, decoder_input, y = collator(data)

    print("Decoder input:")
    print(decoder_input)
    print("Labels (y):")
    print(y)
    
    assert decoder_input.shape == (2, 4)
    assert y.shape == (2, 4)
    
    # sequence 2 should be padded with 99 (pad_token)
    assert decoder_input[1, 2].item() == 99
    assert y[1, 2].item() == 99
    
    # double check the direct function handles it too
    _, decoder_input2, y2 = batch_preparation_img2seq(data, pad_token=77)
    assert decoder_input2[1, 2].item() == 77
    assert y2[1, 2].item() == 77

def test_batch_validation_loop():
    # Simulated validation_step loop for a batch size of 2
    b = 2
    
    # 2 is <bos>, 3 is 'a', 4 is 'b', 5 is 'c', 1 is <eos>, 0 is <pad>
    y = torch.tensor([
        [2, 3, 4, 1, 0],
        [2, 5, 1, 0, 0]
    ])
    
    i2w = {0: '<pad>', 1: '<eos>', 2: '<bos>', 3: 'a', 4: 'b', 5: 'c'}
    
    preds = []
    grtrs = []
    
    for i in range(b):
        y_i = y[i]
        
        gt_tokens = []
        for token in y_i:
            token_item = token.item()
            token_str = i2w.get(token_item, "")
            if token_str == '<eos>':
                break
            if token_str not in ['<pad>', '<bos>', '']:
                gt_tokens.append(token_str)
                
        gt = "".join(gt_tokens)
        grtrs.append(gt)
    
    assert grtrs[0] == "ab", f"Got {grtrs[0]}"
    assert grtrs[1] == "c", f"Got {grtrs[1]}"
    print("test_batch_validation_loop passed!")

if __name__ == "__main__":
    test_batch_padding()
    test_batch_validation_loop()
