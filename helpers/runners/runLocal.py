import torch
import cv2
# Změna 1: Musíme importovat SMT_Trainer, abychom rozbalili .ckpt soubor
from SMT.smt_trainer import SMT_Trainer
from SMT.data_augmentation.data_augmentation import convert_img_to_tensor

print("⏳ Načítám obrázek a lokální model...")

image_path = "../assets/oficial_image_smt.png"

# Načítáme přes klasické cv2 (to ta funkce očekává)
image = cv2.imread(image_path)

if image is None:
    print(f"❌ Chyba: Nemohu najít obrázek '{image_path}'.")
    exit()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Používám zařízení: {device}")

# ==========================================
# MAGIE: NAČTENÍ TVÝCH VLASTNÍCH VAH
# ==========================================
# Cesta k tvému uložení (pokud spouštíš skript ze složky src)
ckpt_path = "SMT/weights/Polish_Scores/last.ckpt" 
# ckpt_path = "SMT/weights/Polish_Scores/FP-Polish_Scores-system-level.ckpt" # Můžeš použít i tento!

print(f"Načítám váhy ze souboru: {ckpt_path}")

# 1. Načteme celou PyTorch Lightning obálku z disku
lightning_wrapper = SMT_Trainer.load_from_checkpoint(ckpt_path)

# 2. Vybalíme z ní ten čistý jazykový model a pošleme ho na grafiku
model = lightning_wrapper.model.to(device)
# ==========================================

print("✅ Model úspěšně načten!")

print("🎹 Probíhá transkripce přes SMT preprocessing...")

# ZDE SE DĚJE TA MAGIE AUTORA
tensor_img = convert_img_to_tensor(image).unsqueeze(0).to(device)

print(f"📏 Tenzor obrázku připraven, velikost: {tensor_img.shape}")

try:
    predictions, _ = model.predict(tensor_img, convert_to_str=True)
    vystup = "".join(predictions).replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t')

    print("\n" + "="*40)
    print("VÝSLEDEK (Kern Notace):")
    print("="*40)
    print(vystup)
    print("="*40)
except Exception as e:
    print(f"❌ Chyba při generování: {e}")