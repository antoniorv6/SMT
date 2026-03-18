import os
import shutil

src_dir = "/nlp/projekty/music_ocr/opensource_models/DeepSeek-OCR-2/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/deepencoderv2"
dest_dir = "/nlp/projekty/music_ocr/SMT-deep/smt_model/architectures/deepencoderv2"

os.makedirs(dest_dir, exist_ok=True)
for f in ["sam_vary_sdpa.py", "qwen2_d2e.py", "build_linear.py", "__init__.py"]:
    src_f = os.path.join(src_dir, f)
    if os.path.exists(src_f):
        shutil.copy(src_f, os.path.join(dest_dir, f))
    else:
        open(os.path.join(dest_dir, f), 'a').close() # touch file

smt_arch_path = "/nlp/projekty/music_ocr/SMT-deep/smt_model/architectures/smt_arch.py"
modeling_smt_path = "/nlp/projekty/music_ocr/SMT-deep/smt_model/modeling_smt.py"
if os.path.exists(modeling_smt_path) and not os.path.exists(smt_arch_path):
    shutil.move(modeling_smt_path, smt_arch_path)

with open("/nlp/projekty/music_ocr/SMT-deep/smt_model/architectures/__init__.py", "a") as f:
    pass

print("Done copying")
