import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

config = Qwen2Config(
    vocab_size=215,
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=256,
    max_position_embeddings=4360 + 1024
)

model = Qwen2ForCausalLM(config).eval()

b = 1
num_vis = 256
num_text = 4360
seq_len = num_vis + num_text

inputs_embeds = torch.randn(b, seq_len, 256)
attention_mask = torch.ones(b, seq_len, dtype=torch.long)

with torch.no_grad():
    try:
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_attentions=False)
        print("Success without SDPA fix (output_attentions=False)")
    except Exception as e:
        print(f"Error without SDPA fix (output_attentions=False): {e}")

    try:
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_attentions=True)
        print("Success with output_attentions=True")
    except Exception as e:
        print(f"Error with output_attentions=True: {e}")

config = Qwen2Config(
    vocab_size=215,
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=256,
    max_position_embeddings=4360 + 1024,
    use_sliding_window=False,
    sliding_window=None
)
model2 = Qwen2ForCausalLM(config).eval()
with torch.no_grad():
    try:
        out = model2(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_attentions=False)
        print("Success with sliding_window=None")
    except Exception as e:
        print(f"Error with sliding_window=None: {e}")

