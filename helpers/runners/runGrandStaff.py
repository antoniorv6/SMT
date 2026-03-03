import sys
import torch
import cv2
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torchvision

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data_augmentation.data_augmentation import convert_img_to_tensor
from smt_model import SMTModelForCausalLM

def load_translated_model(repo_id, device):
    print(f"Initializing model structure from {repo_id}...")
    model = SMTModelForCausalLM.from_pretrained(repo_id)
    
    print("Downloading weights and translating keys...")
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    loaded_weights = load_file(checkpoint_path)
    
    translated_dict = {}
    for key, value in loaded_weights.items():
        new_key = key.replace("cross_attention", "cross_attn") \
                     .replace("input_attention", "self_attn") \
                     .replace("self_attention", "self_attn") \
                     .replace(".lq.", ".q_proj.") \
                     .replace(".lk.", ".k_proj.") \
                     .replace(".lv.", ".v_proj.") \
                     .replace("ffNet", "ffn") \
                     .replace("norm1", "norm_layers.0") \
                     .replace("norm2", "norm_layers.1") \
                     .replace("norm3", "norm_layers.2") \
                     .replace("decoder.out_layer", "decoder.vocab_projection")
        
        if "vocab_projection.weight" in new_key and value.dim() == 3:
            value = value.squeeze(-1)
            
        translated_dict[new_key] = value

    info = model.load_state_dict(translated_dict, strict=False)
    print(f"Load Status: {info}")
    
    return model.to(device).eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
img_path = "/nlp/projekty/music_ocr/SMT-deep/helpers/assets/grandStaff_official.png"

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found at {img_path}")

model = load_translated_model("antoniorv6/smt-grandstaff", device)

image = cv2.imread(img_path)
h, w = image.shape[:2]
image_rescaled = cv2.resize(image, (int(w * (128 / h)), 128))

input_tensor = convert_img_to_tensor(image_rescaled)
input_tensor = (input_tensor - 0.5) / 0.5
input_tensor = input_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    predictions, _ = model.predict(input_tensor, convert_to_str=True)

output = "".join(predictions).replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t')
print("\n--- Model Output ---")
print(output)

print(f"Tensor Shape: {input_tensor.shape}")
print(f"Tensor Range: Min={input_tensor.min():.4f}, Max={input_tensor.max():.4f}")
print(f"Tensor Mean: {input_tensor.mean():.4f}")

torchvision.utils.save_image((input_tensor * 0.5) + 0.5, "debug_input.png")