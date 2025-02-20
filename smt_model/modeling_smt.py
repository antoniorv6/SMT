import math
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger
from typing import Optional, Tuple

class PositionalEncoding2D(nn.Module):

    def __init__(self, dim, h_max, w_max):
        super(PositionalEncoding2D, self).__init__()
        self.h_max = h_max
        self.max_w = w_max
        self.dim = dim
        self.pe = torch.zeros((1, dim, h_max, w_max), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), requires_grad=False)

        div = torch.exp(-torch.arange(0., dim // 2, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        w_pos = torch.arange(0., w_max)
        h_pos = torch.arange(0., h_max)
        self.pe[:, :dim // 2:2, :, :] = torch.sin(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        self.pe[:, 1:dim // 2:2, :, :] = torch.cos(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        self.pe[:, dim // 2::2, :, :] = torch.sin(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)
        self.pe[:, dim // 2 + 1::2, :, :] = torch.cos(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)

    def forward(self, x):
        """
        Add 2D positional encoding to x
        x: (B, C, H, W)
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

    def get_pe_by_size(self, h, w, device):
        return self.pe[:, :, :h, :w].to(device)


class PositionalEncoding1D(nn.Module):

    def __init__(self, dim, len_max):
        super(PositionalEncoding1D, self).__init__()
        self.len_max = len_max
        self.dim = dim
        self.pe = torch.zeros((1, dim, len_max), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), requires_grad=False)

        div = torch.exp(-torch.arange(0., dim, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        l_pos = torch.arange(0., len_max)
        self.pe[:, ::2, :] = torch.sin(l_pos * div).unsqueeze(0)
        self.pe[:, 1::2, :] = torch.cos(l_pos * div).unsqueeze(0)

    def forward(self, x, start):
        """
        Add 1D positional encoding to x
        x: (B, C, L)
        start: index for x[:,:, 0]
        """
        if isinstance(start, int):
            return x + self.pe[:, :, start:start+x.size(2)].to(x.device)
        else:
            for i in range(x.size(0)):
                x[i] = x[i] + self.pe[0, :, start[i]:start[i]+x.size(2)]
            return x

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model:int, num_heads:int, dropout: float = 0.1,
                 bias:bool = True, batch_first:bool = True):
        super().__init__()
        
        assert(d_model % num_heads == 0), logger.error("The embeddings depth must be divisible by the number of heads")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = self.d_head ** -0.5
        self.batch_first = batch_first
        
        self.has_flash_attn = hasattr(F, 'scaled_dot_product_attention')
        if not self.has_flash_attn:
            logger.warning("This program cannot run Flash Attention, for optimal computing, check your GPU driver and your PyTorch version")
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self._init_parameters()
        
        self.dropout = nn.Dropout(dropout, inplace=True)
    

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    

    def _split_heads(self, tensor:torch.Tensor) -> torch.Tensor:
        """Split the heads and put them into a batch-first format."""
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.d_head)
        return tensor.transpose(1,2) # (batch_size, num_heads, seq_len, d_head)
    

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Merge heads and transpose back to batch-first format."""
        batch_size = tensor.shape[0]
        tensor = tensor.transpose(1, 2)
        return tensor.reshape(batch_size, -1, self.d_model).contiguous()
    

    def compute_flash_attn(self, 
                           q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                           attn_mask:Optional[torch.Tensor] = None, is_causal: bool = False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask if not is_causal else None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal = is_causal
            )
    
    def _compute_regular_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ):
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(q.size(2), k.size(2), dtype=torch.bool, device=q.device),
                diagonal=1
            )
            attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_weights.masked_fill_(~attn_mask, float('-inf'))
            else:
                attn_weights += attn_mask
                
        if key_padding_mask is not None:
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)
        
        return attn_output, attn_weights
    
    def forward(self,
                query: torch.Tensor, key: Optional[torch.Tensor] = None, value:Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False, is_causal: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        if key is None and value is None:
            """Since we compute everything from the same source, 
            we just compute linear projection in the query and split the features
            into the three variables to compute attention""" 
            qkv = self.qkv_proj(query)
            q, k, v = torch.split(qkv, self.d_model, dim=-1)
        else:
            if key is None or value is None:
                raise ValueError("Both key and value must be provided for cross-attention")
        
            # We manually multiply each weight section from qkv_proj to their respective vectors
            q = self.qkv_proj.weight[:self.d_model].mm(query.reshape(-1, query.size(-1)).t())
            k = self.qkv_proj.weight[self.d_model:2*self.d_model].mm(key.reshape(-1, key.size(-1)).t())
            v = self.qkv_proj.weight[2*self.d_model:].mm(value.reshape(-1, value.size(-1)).t())
            q = q.t().view(query.shape)
            k = k.t().view(key.shape)
            v = v.t().view(value.shape)

            if self.qkv_proj.bias is not None:
                    q += self.qkv_proj.bias[:self.d_model].unsqueeze(0)
                    k += self.qkv_proj.bias[self.d_model:2*self.d_model].unsqueeze(0)
                    v += self.qkv_proj.bias[2*self.d_model:].unsqueeze(0)
        
        # Split the heads in q, k and v
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        
        # Scale the query
        q = q * self.scale
        
        use_flash_attn = self.has_flash_attn and not need_weights
        
        if use_flash_attn:
            attn_output = self.compute_flash_attn(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
            attn_weights = None
        else:
            attn_output, attn_weights = self._compute_regular_attention(q, k, v, key_padding_mask, attn_mask, is_causal)
        
        output = self._merge_heads(attn_output)
        output = self.out_proj(output)
        
        if need_weights:
            return output, attn_weights
        
        return output, None

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_ff:int,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.activation = nn.ReLU() if activation.lower() == "relu" else nn.GELU()
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        
        self.norm_layers = [nn.LayerNorm(d_model) for _ in range(3)]
        self.dropout_layers = [nn.Dropout(dropout) for _ in range(3)]
    
    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                return_weights=False):
        
        attn_output, self_weights = self.self_attn(
            query=x, key=None, value=None, 
            key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask,
            is_causal=True
        )
        
        x = x + self.dropout_layers[0](attn_output)
        x = self.norm_layers[0](x)
        
        attn_output, cross_weights = self.cross_attn(
            query=x,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=memory_key_padding_mask,
            is_causal=False
        )
        
        x = x + self.dropout_layers[1](attn_output)
        x = self.norm_layers[1](x)
        
        ffn_output = self.ffn(x)
        x = x + self.dropout_layers[2](ffn_output)
        x = self.norm_layers[2](x)
        
        if return_weights:
            return x, [self_weights, cross_weights]
            
        return x, None

class DecoderStack(nn.Module):
    def __init__(self, num_dec_layers:int,
                 d_model:int, dim_ff:int, num_heads:int,
                 dropout:float):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, 
                                                  num_heads=num_heads, 
                                                  dim_ff=dim_ff) for _ in range(num_dec_layers)])
    def forward(self,
                x:torch.Tensor, encoder_output:torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                return_weights=False):
        
        output = x
        all_weights = {
            "self_attn": [],
            "cross_attn": []
        }
        
        for i, dec_layer in enumerate(self.layers):
            output, weights = dec_layer(x=output, encoder_output=encoder_output, 
                                        tgt_mask=tgt_mask, 
                                        tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask,
                                        return_weights=return_weights)
            if return_weights:
                all_weights["self_attn"].append(weights[0])
                all_weights["cross_attn"].append(weights[1])
        
        if return_weights:
            return output, all_weights
        
        return output, None

class Decoder(nn.Module):
    def __init__(self, num_dec_layers:int, 
                 d_model:int, dim_ff:int, n_heads:int,
                 max_seq_length:int, out_categories:int, dropout:float = 0.1):
        
        super().__init__()
        
        self.decoder = DecoderStack(num_dec_layers=num_dec_layers,
                                    d_model=d_model, dim_ff=dim_ff, num_heads=n_heads,
                                    dropout=dropout)
        
        self.embedding = nn.Embedding(num_embeddings=out_categories, embedding_dim=d_model)
        
        self.position_encoding = PositionalEncoding1D(dim=d_model, len_max=max_seq_length)
        
        self.vocab_projection = nn.Linear(in_features=d_model, out_features=out_categories)
        
        self.end_relu = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, decoder_input:torch.Tensor, encoder_output:torch.Tensor,
                tgt_mask:Optional[torch.Tensor] = None, 
                tgt_key_padding_mask:Optional[torch.Tensor] = None, 
                memory_key_padding_mask:Optional[torch.Tensor] = None, 
                return_weights = False):
        
        decoder_input = self.embedding(decoder_input).permute(0,2,1).contiguous()
        decoder_input = self.position_encoding(decoder_input, start=0).permute(0,2,1).contiguous()
        
        output, weights = self.decoder(x=decoder_input, encoder_output=encoder_output, 
                                       tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, 
                                       memory_key_padding_mask=memory_key_padding_mask, 
                                       return_weights=return_weights)
        
        output = self.dropout(self.end_relu(output))
        
        predictions = self.vocab_projection(output)
        
        return output, predictions, weights
        