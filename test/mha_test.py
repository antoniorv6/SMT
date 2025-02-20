import sys
from pathlib import Path
import torch
sys.path.append(str(Path(__file__).parent.parent))
from smt_model import Decoder
from torchinfo import summary

torch.manual_seed(42)

# Initialize module
model = Decoder(num_dec_layers=8,
                d_model=256, dim_ff=256, n_heads=4, 
                max_seq_length=1024, out_categories=600)

# Create sample input
decoder_input = torch.randint(0, 600, size=(1,512))
encoder_output = torch.rand((1, 1024, 256))

# Test forward pass
output, predictions, weights = model(decoder_input=decoder_input, encoder_output=encoder_output)

summary(model, input_size=[(1, 512), (1, 1024, 256)], dtypes=[torch.long, torch.float], device="cpu")