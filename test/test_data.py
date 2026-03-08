import torch
import numpy as np

def test_batch_padding():
    # Simulate batch data
    # format: [(image, decoder_input, y), ...]
    # image: tensor of shape (1, H, W)
    # decoder_input: tensor of seq len (includes <bos> and <eos>)
    # y: same as decoder_input
    
    pad_token = 0
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
    
    images = [sample[0] for sample in data]
    dec_in = [sample[1] for sample in data]
    gt = [sample[2] for sample in data]

    max_image_width = max(128, max([img.shape[2] for img in images]))
    max_image_height = max(256, max([img.shape[1] for img in images]))

    X_train = torch.ones(size=[len(images), 1, max_image_height, max_image_width], dtype=torch.float32)

    for i, img in enumerate(images):
        _, h, w = img.size()
        X_train[i, :, :h, :w] = img

    max_length_seq = max([len(w) for w in gt])

    decoder_input = torch.full(size=[len(dec_in),max_length_seq-1], fill_value=pad_token, dtype=torch.long)
    y = torch.full(size=[len(gt),max_length_seq-1], fill_value=pad_token, dtype=torch.long)

    for i, seq in enumerate(dec_in):
        seq_tensor = torch.as_tensor(seq[:-1])
        decoder_input[i, :len(seq_tensor)] = seq_tensor # all tokens but <eos>

    for i, seq in enumerate(gt):
        seq_tensor = torch.as_tensor(seq[1:])
        y[i, :len(seq_tensor)] = seq_tensor # all tokens but <bos>

    print("Decoder input:")
    print(decoder_input)
    print("Labels (y):")
    print(y)
    
    assert decoder_input.shape == (2, 4)
    assert y.shape == (2, 4)
    
    # sequence 2 should be padded with 0 (pad_token)
    assert decoder_input[1, 2].item() == 0
    assert y[1, 2].item() == 0

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
