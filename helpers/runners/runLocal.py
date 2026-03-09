import torch
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from smt_trainer import SMT_Trainer
from data_augmentation.data_augmentation import convert_img_to_tensor

print("⏳ Načítám obrázek a lokální model...")
image_path = "/nlp/projekty/music_ocr/SMT-deep/helpers/assets/oficial_image_smt.png"
image = cv2.imread(image_path)

if image is None:
    print(f"❌ Chyba: Nemohu najít obrázek '{image_path}'.")
    exit()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Používám zařízení: {device}")

ckpt_path = "/nlp/projekty/music_ocr/SMT-deep/weights/Polish_Scores/FP-Polish_Scores-system-level.ckpt" 
print(f"Načítám váhy ze souboru: {ckpt_path}")

# 1. Načteme obálku z PyTorch Lightning
lightning_wrapper = SMT_Trainer.load_from_checkpoint(ckpt_path, map_location=device)
model = lightning_wrapper.model.to(device)

# ==========================================
# OPRAVA 1: Oprava poškozených slovníků v configu
# ==========================================
# Lightning při ukládání .ckpt převede 'int' klíče na 'str'.
# Opravíme je přímo uvnitř 'model.config', kde je model reálně hledá.
if hasattr(model, 'config'):
    model.config.i2w = {int(k): v for k, v in model.config.i2w.items()}
    model.config.w2i = {k: int(v) for k, v in model.config.w2i.items()}

# Pro jistotu opravíme i přímo na modelu (pokud si drží kopii)
if hasattr(model, 'i2w'):
    model.i2w = {int(k): v for k, v in model.i2w.items()}
if hasattr(model, 'w2i'):
    model.w2i = {k: int(v) for k, v in model.w2i.items()}
# ==========================================

print("✅ Model úspěšně opraven a načten!")

print("🎹 Probíhá transkripce přes SMT preprocessing...")
tensor_img = convert_img_to_tensor(image).unsqueeze(0).to(device)

try:
    # ==========================================
    # OPRAVA 2: Volání přesně podle autora
    # ==========================================
    # Podle smt_trainer.py autor volá čistě 'predict(input=...)' 
    predictions, _ = model.predict(input=tensor_img)
    
    # We unwrap the first item from the batch
    vystup = "".join(predictions[0]).replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t')

    print("\n" + "="*40)
    print("VÝSLEDEK (Kern Notace):")
    print("="*40)
    print(vystup)
    print("="*40)
except Exception as e:
    import traceback
    print(f"❌ Chyba při generování: {e}")
    traceback.print_exc()