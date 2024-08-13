from transformers import PretrainedConfig

class SMTConfig(PretrainedConfig):
    model_type = "SMT"

    def __init__(self, maxh=3508, maxw=2480, maxlen=1512, out_categories=2512, padding_token=0, 
                 in_channels=1, w2i={}, i2w={}, out_dir="out_smt", 
                 d_model=256, dim_ff=256, num_dec_layers=8, attn_heads=4,
                 use_flash_attn=False, **kwargs):
        self.architectures = ["SMT"]
        self.maxh = maxh
        self.maxw = maxw
        self.maxlen = maxlen
        self.out_categories = out_categories
        self.padding_token = padding_token
        self.in_channels = in_channels
        self.w2i = w2i
        self.i2w = i2w
        self.out_dir = out_dir
        self.d_model = d_model
        self.dim_ff = dim_ff
        self.num_attn_heads = attn_heads
        self.num_dec_layers = num_dec_layers
        self.use_flash_attn = use_flash_attn