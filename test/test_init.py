import torch
import sys
import os

# Add to path so imports work
sys.path.append(os.path.abspath("/nlp/projekty/music_ocr/SMT-deep"))

from smt_model import SMTConfig
from smt_model import build_model
import json
from ExperimentConfig import experiment_config_from_dict

print("Testing SMT init...")
try:
    # We just need a basic dummy SMTConfig
    config = SMTConfig(maxh=128, maxw=128, maxlen=200, out_categories=50,
                       padding_token=0, in_channels=1, w2i={"<pad>":0, "<bos>":1, "<eos>":2},
                       i2w={0:"<pad>", 1:"<bos>", 2:"<eos>"}, d_model=256, dim_ff=256,
                       attn_heads=4, num_dec_layers=2)

    model_smt = build_model(config, arch_type="smt")
    print("SMT model built successfully!")
    
    model_deepseek = build_model(config, arch_type="deepseek")
    print("Deepseek model built successfully!")

    # Test basic forward encoder shape
    dummy_img = torch.randn(1, 1, 128, 128)
    # SMT Encoder
    out_smt = model_smt.forward_encoder(dummy_img)
    print("SMT encoder output shape:", out_smt.shape)
    
    # DeepSeek Encoder
    out_ds = model_deepseek.forward_encoder(dummy_img)
    print("DeepSeek encoder output shape:", out_ds.shape)

    print("Testing autoregressive predict...")
    strings_smt, _ = model_smt.predict(dummy_img)
    print("SMT predict returned", len(strings_smt), "sequences")
    
    strings_ds, _ = model_deepseek.predict(dummy_img)
    print("DeepSeek predict returned", len(strings_ds), "sequences")

except Exception as e:
    import traceback
    traceback.print_exc()
