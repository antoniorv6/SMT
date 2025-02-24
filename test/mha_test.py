import sys
from pathlib import Path
import torch
sys.path.append(str(Path(__file__).parent.parent))
from smt_model import SMTConfig, SMTModelForCausalLM
from torchinfo import summary

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize module
config = SMTConfig(maxh=1024, maxw=1024, out_categories=600)

model = SMTModelForCausalLM(config).to(device)

# Create sample input
decoder_input = torch.randint(0, 600, size=(1,512)).to(device)
encoder_output = torch.rand((1, 1, 1024, 1024)).to(device)

# Test forward pass
output = model(encoder_input=encoder_output, decoder_input=decoder_input)

summary(model, input_size=[(1, 1, 512, 512), (1, 1024)], dtypes=[torch.float, torch.long], device=device)