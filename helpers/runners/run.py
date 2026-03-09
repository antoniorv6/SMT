import torch
import cv2
from SMT.smt_model import SMTModelForCausalLM
# Odtud vytáhneme tu autorovu zázračnou funkci!
from SMT.data_augmentation.data_augmentation import convert_img_to_tensor

print("⏳ Načítám obrázek a model...")

image_path = "../assets/oficial_image_smt.png"

# Načítáme přes klasické cv2 (to ta funkce očekává)
image = cv2.imread(image_path)

if image is None:
    print(f"❌ Chyba: Nemohu najít obrázek '{image_path}'.")
    exit()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Používám zařízení: {device}")

model_id = "antoniorv6/smt-grandstaff"
model = SMTModelForCausalLM.from_pretrained(model_id).to(device)
print("✅ Model úspěšně načten!")

print("🎹 Probíhá transkripce přes SMT preprocessing...")

# ZDE SE DĚJE TA MAGIE AUTORA
# convert_img_to_tensor to invertuje, obarví do šeda a natáhne
tensor_img = convert_img_to_tensor(image).unsqueeze(0).to(device)

print(f"📏 Tenzor obrázku připraven, velikost: {tensor_img.shape}")

try:
    predictions, _ = model.predict(tensor_img, convert_to_str=True)
    # the predictions are now batched lists of tokens, so we extract index 0
    vystup = "".join(predictions[0]).replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t')

    print("\n" + "="*40)
    print("VÝSLEDEK (Kern Notace):")
    print("="*40)
    print(vystup)
    print("="*40)
except Exception as e:
    print(f"❌ Chyba při generování: {e}")