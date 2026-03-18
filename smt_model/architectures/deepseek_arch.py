import torch
import torch.nn as nn
from transformers import PreTrainedModel
from smt_model.configuration_smt import SMTConfig
from .deepencoderv2.sam_vary_sdpa import build_sam_vit_b
from .deepencoderv2.qwen2_d2e import build_qwen2_decoder_as_encoder
from .deepencoderv2.build_linear import MlpProjector
class Dict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value
from smt_model.architectures.smt_arch import SMTOutput

class DeepSeekOCR2Wrapper(PreTrainedModel):
    config_class = SMTConfig

    def __init__(self, config: SMTConfig):
        super().__init__(config)
        self.config = config
        
        # DeepSeek-OCR-2 Vision Encoder components
        self.sam_model = build_sam_vit_b().to(torch.bfloat16)
        self.qwen2_model = build_qwen2_decoder_as_encoder().to(torch.bfloat16)
        
        n_embed = config.d_model  # Project to SMT's expected dim or keep it 1280
        # Usually it's 1280 for deepseek, but we can project to d_model for SMT's vocabulary projection
        self.projector = MlpProjector(Dict(projector_type="linear", input_dim=896, n_embed=n_embed)).to(torch.bfloat16)

        self.view_separator = nn.Parameter(torch.randn(n_embed) * (1 / (n_embed**0.5)))
        
        # We need an SMT-compatible decoder instead of deepseek LLM if we want to train it from scratch on SMT data
        # The goal is "extract the core vision/language model architecture". 
        # DeepSeek-OCR-2 uses a large LLM as decoder. If we just extract the architecture, we could use a custom decoder 
        # or stick to the SMT Decoder to keep vocabulary size / pipeline exactly the same, 
        # or use a standard transformer decoder. Let's use the SMT Decoder as the language model component
        # so it seamlessly integrates, but replacing the ConvNeXt encoder with DeepSeek's SAM+Qwen encoder.
        from .smt_arch import Decoder
        self.decoder = Decoder(num_dec_layers=config.num_dec_layers,
                               d_model=n_embed, dim_ff=config.dim_ff, n_heads=config.num_attn_heads,
                               max_seq_length=config.maxlen, out_categories=config.out_categories)
        self.loss = nn.CrossEntropyLoss(ignore_index=config.padding_token)

        self.w2i = config.w2i
        self.i2w = config.i2w
        self.maxlen = int(config.maxlen)
        self.padding_token = config.padding_token

        # A 1-to-3 channel projection if SMT provides 1-channel images
        self.gray_to_rgb = nn.Conv2d(1, 3, kernel_size=1) if config.in_channels == 1 else nn.Identity()

    def forward_encoder(self, x):
        # x is (B, C, H, W)
        x = self.gray_to_rgb(x)
        x = x.to(torch.bfloat16)

        # In DeepSeek-OCR-2, this is processed in patches/crops. 
        # SMT does global feature extraction typically on the whole image.
        # We'll pass the whole image as global_features
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            global_features_1 = self.sam_model(x)
            global_features_2 = self.qwen2_model(global_features_1) 
            global_features = self.projector(global_features_2)
            
            # global_features shape: [B, HW, n_embed]
            # Since SMT expects 2D pos encoding in its original decoder, we need to return
            # features that can be shaped back to (B, n_embed, H, W) or we modify how it handles it.
            B, HW, n_embed = global_features.shape
            
            # Estimate H and W from HW (assuming square or aspect ratio of x)
            # SMT requires encoder_output as [B, C, H, W] for pos2D
            orig_h, orig_w = x.shape[2], x.shape[3]
            # SAM uses patch_size 16. Then Neck + Conv2 + Conv3 downsamples by 2*2=4.
            # Total downsample = 16 * 2 * 2 = 64.
            h_out = orig_h // 64
            w_out = orig_w // 64
            
            # Ensure hw matches h_out * w_out. Pad or truncate if necessary, or just rely on dynamic reshape.
            # To be robust:
            h_feat = int(torch.sqrt(torch.tensor(HW * (orig_h / orig_w))))
            w_feat = HW // h_feat
            
            try:
                global_features = global_features.view(B, h_feat, w_feat, n_embed).permute(0, 3, 1, 2)
            except Exception:
                # If cannot reshape exactly, return as 1D sequence (B, n_embed, 1, HW)
                global_features = global_features.permute(0, 2, 1).unsqueeze(2)

        return global_features.float()

    def forward_decoder(self, encoder_output, last_predictions, return_weights=False):
        # encoder_output is [B, C, H, W]
        # In SMT's original model, there's a pos2D encoding. 
        # DeepSeek uses its own rotary/learned embeddings, but since we use SMT decoder here...
        # Let's bypass Pos2D if shape is weird, or just use 1D sequence.
        
        b = encoder_output.size(0)
        # Flatten features
        encoder_features = torch.flatten(encoder_output, start_dim=2, end_dim=3).permute(0, 2, 1) # [B, HW, C]
        encoder_features_2D = encoder_features # Bypass 2D Pos encoding if incompatible
        
        key_target_mask = self._generate_token_mask([lp.shape[0] for lp in last_predictions], last_predictions.size(), device=last_predictions.device)
        causal_mask = self._generate_causal_mask(last_predictions.size(1), last_predictions.device)

        output, predictions, weights = self.decoder(decoder_input=last_predictions,
                                                    encoder_output_2D=encoder_features_2D, encoder_output_raw=encoder_features,
                                                    tgt_mask=causal_mask, tgt_key_padding_mask=key_target_mask,
                                                    memory_key_padding_mask=None,
                                                    return_weights=return_weights)

        return SMTOutput(
            logits=predictions,
            hidden_states=output,
            attentions=None if weights is None else weights["self_attn"],
            cross_attentions=None if weights is None else weights["cross_attn"]
        )

    def forward(self, encoder_input, decoder_input, labels=None):
        x = self.forward_encoder(encoder_input)
        output = self.forward_decoder(x, decoder_input)

        if labels is not None:
            output.loss = self.loss(output.logits.permute(0,2,1).contiguous(), labels)
        return output

    @torch.no_grad
    def predict(self, input, convert_to_str=False, return_weights=False):
        b = input.size(0)
        predicted_sequence = torch.full((b, 1), self.w2i['<bos>'], dtype=torch.long, device=input.device)
        encoder_output = self.forward_encoder(input)
        
        has_eos = torch.zeros(b, dtype=torch.bool, device=input.device)
        eos_id = self.w2i['<eos>']

        for i in range(self.maxlen - predicted_sequence.size(1)):
            output = self.forward_decoder(encoder_output=encoder_output, last_predictions=predicted_sequence,
                                          return_weights=return_weights)
            predicted_tokens = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
            predicted_sequence = torch.cat([predicted_sequence, predicted_tokens], dim=1)
            
            has_eos |= (predicted_tokens.squeeze(1) == eos_id)
            if has_eos.all():
                break

        text_sequences = []
        for b_idx in range(b):
            seq = []
            for token_id in predicted_sequence[b_idx, 1:]:
                token_val = str(token_id.item()) if convert_to_str else token_id.item()
                token_str = self.i2w.get(token_val, "")
                if token_str == '<eos>':
                    break
                seq.append(token_str)
            text_sequences.append(seq)

        return text_sequences, output

    def _generate_token_mask(self, token_len, total_size, device):
        batch_size, len_mask = total_size
        mask = torch.zeros((batch_size, len_mask), dtype=torch.bool, device=device)
        for i, len_ in enumerate(token_len):
            mask[i, :len_] = True
        return mask

    def _generate_causal_mask(self, token_len, device):
        causal_mask = torch.triu(
                torch.ones(token_len, token_len, dtype=torch.bool, device=device),
                diagonal=1
            )
        return causal_mask
