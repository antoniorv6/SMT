from transformers import AutoModel, AutoConfig
import torch
import sys

try:
    print("Loading Authentic DeepSeek-OCR-2 from HF Hub...")
    ds_config = AutoConfig.from_pretrained('deepseek-ai/DeepSeek-OCR-2', trust_remote_code=True)
    ds_config._attn_implementation = 'eager'
    ds_model = AutoModel.from_config(ds_config, trust_remote_code=True).eval()

    vision = ds_model.vision_model
    aligner = ds_model.aligner
    
    print("\nTesting Vision Model Signature:")
    try:
        # Test 4D
        dummy_4d = torch.randn(1, 3, 1024, 1024)
        out_4d = vision(dummy_4d)
        print("Vision model accepted 4D tensor: [B, C, H, W]")
    except Exception as e:
        print(f"4D tensor failed: {e}")
        try:
            # Test 5D (B, num_images, C, H, W) -> standard VL2 format
            dummy_5d = torch.randn(1, 1, 3, 1024, 1024)
            out_5d = vision(dummy_5d)
            print("Vision model accepted 5D tensor: [B, N, C, H, W]")
        except Exception as e:
            print(f"5D tensor failed: {e}")

except Exception as e:
    print(f"Exception during model load: {e}")
