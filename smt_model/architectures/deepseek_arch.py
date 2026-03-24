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
from transformers import Qwen2Config, Qwen2ForCausalLM

class DeepSeekOCR2Wrapper(PreTrainedModel):
    config_class = SMTConfig

    def __init__(self, config: SMTConfig):
        super().__init__(config)
        self.config = config
        
        # DeepSeek-OCR-2 Vision Encoder components
        self.sam_model = build_sam_vit_b()
        self.qwen2_model = build_qwen2_decoder_as_encoder()
        
        n_embed = config.d_model  # Project to SMT's expected dim or keep it 1280
        # Usually it's 1280 for deepseek, but we can project to d_model for SMT's vocabulary projection
        self.projector = MlpProjector(Dict(projector_type="linear", input_dim=896, n_embed=n_embed))

        self.view_separator = nn.Parameter(torch.randn(n_embed) * (1 / (n_embed**0.5)))
        
        # We need an SMT-compatible decoder instead of deepseek LLM if we want to train it from scratch on SMT data
        # The goal is "extract the core vision/language model architecture". 
        # DeepSeek-OCR-2 uses a large LLM as decoder. If we just extract the architecture, we could use a custom decoder 
        # or stick to the SMT Decoder to keep vocabulary size / pipeline exactly the same, 
        # or use a standard transformer decoder. Let's use the SMT Decoder as the language model component
        # so it seamlessly integrates, but replacing the ConvNeXt encoder with DeepSeek's SAM+Qwen encoder.
        # Replace SMT Decoder with DeepSeek native Qwen2 decoder architecture, initialized with SMT's vocabulary size.
        qwen_config = Qwen2Config(
            vocab_size=config.out_categories,
            hidden_size=n_embed,
            num_hidden_layers=config.num_dec_layers,
            num_attention_heads=config.num_attn_heads,
            intermediate_size=config.dim_ff,
            max_position_embeddings=config.maxlen + 1024  # capacity for visual tokens + max seq length
        )
        self.decoder = Qwen2ForCausalLM(qwen_config)
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

        # SAM expects 1024×1024 input — resize to match its positional embeddings
        x = torch.nn.functional.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)

        # Let Lightning's AMP handle precision (float16) — no manual bfloat16 casting
        global_features_1 = self.sam_model(x)
        global_features_2 = self.qwen2_model(global_features_1) 
        global_features = self.projector(global_features_2)
        
        # global_features shape: [B, HW, n_embed]
        # Since we always feed 1024x1024 to SAM, the spatial output is deterministic.
        # SMT requires encoder_output as [B, C, H, W] for pos2D
        B, HW, n_embed = global_features.shape
        
        # SAM 1024 -> patches 64x64 -> neck 64x64 -> conv2 32x32 -> conv3 16x16
        # qwen2 outputs 256 tokens (16x16) as query, so HW = 256
        h_feat = int(HW ** 0.5)
        w_feat = HW // h_feat
        
        global_features = global_features.view(B, h_feat, w_feat, n_embed).permute(0, 3, 1, 2)

        return global_features

    def forward_decoder(self, encoder_output, last_predictions, return_weights=False):
        # encoder_output is [B, C, H, W]
        b = encoder_output.size(0)
        
        # Flatten features: [B, HW, C]
        encoder_features = torch.flatten(encoder_output, start_dim=2, end_dim=3).permute(0, 2, 1) 
        
        # SMT text predictions -> tokens -> embeddings
        text_embeds = self.decoder.model.embed_tokens(last_predictions)
        
        # Concatenate: [Visual_Tokens, Text_Tokens]
        inputs_embeds = torch.cat([encoder_features, text_embeds], dim=1)
        
        # Manage padding mask (visual tokens are always valid)
        num_vis = encoder_features.size(1)
        text_attention_mask = (last_predictions != self.padding_token).long()
        vis_attention_mask = torch.ones((b, num_vis), dtype=torch.long, device=last_predictions.device)
        attention_mask = torch.cat([vis_attention_mask, text_attention_mask], dim=1)

        # Standard autoregressive forward pass
        outputs = self.decoder(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            output_attentions=return_weights
        )

        # We only care about predicting the text sequence, so slice off the visual token logits
        logits = outputs.logits[:, num_vis:, :]

        return SMTOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=None
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
        encoder_features = torch.flatten(encoder_output, start_dim=2, end_dim=3).permute(0, 2, 1)
        
        has_eos = torch.zeros(b, dtype=torch.bool, device=input.device)
        eos_id = self.w2i['<eos>']

        # Simplified step-by-step autoregression without full KV caching for equivalence to previous SMT structure
        # (Could be significantly optimized with huggingface GenerationMixin later if inference speed is a priority)
        outputs = None
        for i in range(self.maxlen - predicted_sequence.size(1)):
            text_embeds = self.decoder.model.embed_tokens(predicted_sequence)
            inputs_embeds = torch.cat([encoder_features, text_embeds], dim=1)
            
            outputs = self.decoder(inputs_embeds=inputs_embeds, output_attentions=return_weights)
            
            # The next token prediction is the last predicted logit in the sequence
            predicted_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
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

        # Create dummy wrapper for outputs for compatibility
        decoder_output = SMTOutput(
            logits=outputs.logits if outputs else None,
            hidden_states=outputs.hidden_states if outputs else None,
            attentions=outputs.attentions if outputs else None,
            cross_attentions=None
        )

        return text_sequences, decoder_output
