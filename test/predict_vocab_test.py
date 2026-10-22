import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent))
from smt_model import SMTModelForCausalLM


def build_model(i2w):
    """Build a model shell whose decoder always predicts token 3, then <eos> (token 1)."""
    model = SMTModelForCausalLM.__new__(SMTModelForCausalLM)
    torch.nn.Module.__init__(model)
    model.w2i = {"<bos>": 2}
    model.i2w = i2w
    model.maxlen = 6
    model.forward_encoder = lambda x: x

    def forward_decoder(encoder_output, last_predictions, return_weights=False):
        token = 3 if last_predictions.shape[-1] < 3 else 1
        logits = torch.zeros(1, last_predictions.shape[-1], 4)
        logits[:, -1, token] = 1.0
        return SimpleNamespace(logits=logits)

    model.forward_decoder = forward_decoder
    return model


@pytest.mark.parametrize("key_type", [int, str])
@pytest.mark.parametrize("convert_to_str", [False, True])
def test_predict_accepts_int_and_str_vocabulary_keys(key_type, convert_to_str):
    i2w = {key_type(0): "<pad>", key_type(1): "<eos>", key_type(2): "<bos>", key_type(3): "*clefF4"}
    model = build_model(i2w)

    tokens, _ = model.predict(torch.zeros(1, 1, 4, 4), convert_to_str=convert_to_str)

    assert tokens == ["*clefF4", "*clefF4"]
