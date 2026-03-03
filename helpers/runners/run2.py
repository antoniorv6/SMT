import torch
import cv2
from SMT.data_augmentation.data_augmentation import convert_img_to_tensor
from SMT.smt_model import SMTModelForCausalLM

image = cv2.imread("../assets/oficial_image_smt.png")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SMTModelForCausalLM.from_pretrained("antoniorv6/smt-grandstaff").to(device)

predictions, _ = model.predict(convert_img_to_tensor(image).unsqueeze(0).to(device), 
                               convert_to_str=True)

print("".join(predictions).replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t'))
