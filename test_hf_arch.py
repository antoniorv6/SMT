from transformers import AutoModel, AutoConfig
import torch

model = AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR-2', trust_remote_code=True)

print("Modules in DeepSeek-OCR-2 AutoModel:")
for name, module in model.named_children():
    print(f"- {name} ({type(module).__name__})")

print("\nDecoder architecture:")
if hasattr(model, 'language_model'):
    print(model.language_model)
    print("LM embed_tokens size:", model.language_model.model.embed_tokens.weight.shape)
elif hasattr(model, 'language'):
    print(model.language)
    print("LM embed_tokens size:", model.language.model.embed_tokens.weight.shape)
else:
    print("No language_model found or named differently.")
