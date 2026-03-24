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
        # Extract authentic DeepSeek components dynamically!
        from transformers import AutoModel, AutoConfig
        
        # Determine if we should load pretrained weights from HF Hub
        use_pretrained = getattr(config, 'use_pretrained_weights', True)
        
        try:
            model_name = 'deepseek-ai/DeepSeek-OCR-2'
            if use_pretrained:
                print(f"Loading Authentic DeepSeek-OCR-2 from HF Hub with pretrained weights...")
                ds_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, _attn_implementation='eager')
            else:
                print(f"Loading Authentic DeepSeek-OCR-2 Architecture (Random Weights)...")
                ds_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                ds_config._attn_implementation = 'eager'
                ds_model = AutoModel.from_config(ds_config, trust_remote_code=True)
                
            # Unnest the heavily customized DeepSeek components
            # DeepSeek-OCR-2 actually forces the Vision modules INSIDE the Language Model core natively!
            self.decoder = ds_model
            
            self.vision_model = ds_model.model.sam_model
            self.qwen2_adapter = ds_model.model.qwen2_model
            self.aligner = ds_model.model.projector
            
            self.view_separator = ds_model.model.view_seperator
            
            # Strip the vision branches from the parent decoder to strictly decouple LM and Vision phases for SMT compatibility.
            del ds_model.model.sam_model
            del ds_model.model.qwen2_model
            del ds_model.model.projector
            
            # SLICE & RESIZE text vocabulary to exactly fit SMT (e.g. 215 tokens)
            print(f"Resizing DeepSeek Vocabulary to SMT categories: {config.out_categories}")
            old_embeds = self.decoder.model.embed_tokens
            old_lm_head = self.decoder.lm_head
            
            self.decoder.model.embed_tokens = nn.Embedding(config.out_categories, old_embeds.embedding_dim)
            self.decoder.lm_head = nn.Linear(old_lm_head.in_features, config.out_categories, bias=False)
            self.decoder.config.vocab_size = config.out_categories
            
            self.n_embed = old_embeds.embedding_dim
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Remote DeepSeek Model: {e}")

        self.loss = nn.CrossEntropyLoss(ignore_index=config.padding_token)

        self.w2i = config.w2i
        self.i2w = config.i2w
        self.maxlen = int(config.maxlen)
        self.padding_token = config.padding_token

        # SMT images might be 1-channel, ensure 3-channel for SAM
        self.gray_to_rgb = nn.Conv2d(1, 3, kernel_size=1) if config.in_channels == 1 else nn.Identity()

    def forward_encoder(self, x):
        # x is (B, C, H, W) -> Grayscale to RGB
        x = self.gray_to_rgb(x)
        
        b, c, h, w = x.shape
        # DeepSeek native padding & patching strategy to avoid 1024x1024 harsh squish
        # We use standard interpolation for the exact required dimensions safely
        x_global = torch.nn.functional.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        
        # For simplicity in native memory architecture (skipping dynamic tile calculation logic for safety), 
        # we will use the single unified Global View that maps perfectly to SAM's native token structure.
        # This keeps VRAM stable, while using native parameters. 
        # DeepSeek automatically flattens Vision outputs to token grids.
        
        # Pass to extracted native components
        features_1 = self.vision_model(x_global)
        features_2 = self.qwen2_adapter(features_1)
        global_features = self.aligner(features_2)
        
        # Extract the sequence tokens: shape [B, num_vis_tokens, n_embed]
        # Reshape to [B, C, H, W] to fit SMT intermediate requirements for deepseek_arch.py interface bridging
        HW = global_features.size(1)
        b = global_features.size(0)
        
        h_feat = int(HW ** 0.5)
        w_feat = HW // h_feat
        
        global_features = global_features.view(b, h_feat, w_feat, -1).permute(0, 3, 1, 2)
        
        return global_features

    def forward_decoder(self, encoder_output, last_predictions, return_weights=False):
        # encoder_output is [B, C, H, W]
        b = encoder_output.size(0)
        
        # Flatten features: [B, C, HW] -> [B, HW, C]
        encoder_features = torch.flatten(encoder_output, start_dim=2, end_dim=3).permute(0, 2, 1) 
        
        # SMT text predictions -> tokens -> embeddings
        text_embeds = self.decoder.model.embed_tokens(last_predictions)
        
        # Incorporate the DeepSeek vision-text separator token natively
        sep = self.view_separator[None, None, :].expand(b, 1, self.n_embed)
        
        # Concatenate: [Visual_Tokens, Separator, Text_Tokens]
        inputs_embeds = torch.cat([encoder_features, sep, text_embeds], dim=1)
        
        # Manage padding mask (visual tokens and their separator are always valid)
        num_vis = encoder_features.size(1)
        text_attention_mask = (last_predictions != self.padding_token).long()
        
        # Add + 1 to account for the view_separator directly in the valid attention matrix
        vis_attention_mask = torch.ones((b, num_vis + 1), dtype=torch.long, device=last_predictions.device)
        
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
