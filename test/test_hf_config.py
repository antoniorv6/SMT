from transformers import AutoConfig, AutoModel
import sys

try:
    print("Loading config from HuggingFace...")
    config = AutoConfig.from_pretrained('deepseek-ai/DeepSeek-OCR-2', trust_remote_code=True)
    print("Config successfully loaded:")
    print(config)
    
    print("\nText Config:")
    print(config.text_config)

except Exception as e:
    print(f"Failed to load config: {e}")
