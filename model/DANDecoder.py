import gin

import torch
import torch.nn as nn

from torch.nn.init import xavier_uniform_

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

class MHA(nn.Module):

    def __init__(self, embedding_dim, num_heads=None, dropout=0, proj_value=True) -> None:
        super().__init__()

        self.proj_value = proj_value
        self.lq = nn.Linear(embedding_dim, embedding_dim)
        self.lk = nn.Linear(embedding_dim, embedding_dim)
        if proj_value:
            self.lv = nn.Linear(embedding_dim, embedding_dim)
        
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale_factor = float(self.head_dim) ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query, key, value, key_pad_mask=None, attn_mask=None, get_weights=True):
        
        target_len, b, c = query.size()
        source_len = key.size(0)

        q = self.lq(query)
        k = self.lk(key)
        v = self.lv(value) if self.proj_value else value

        q = torch.reshape(q, (target_len, b*self.num_heads, self.head_dim)).transpose(0, 1)
        k = torch.reshape(k, (source_len, b*self.num_heads, self.head_dim)).transpose(0, 1)
        v = torch.reshape(v, (source_len, b*self.num_heads, self.head_dim)).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1,2))

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if attn_mask.dtype == torch.bool:
                attn_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_weights += attn_mask

        if key_pad_mask is not None:
            attn_output_weigths = attn_weights.view(b, self.num_heads, target_len, source_len)
            attn_output_weigths = attn_weights.masked_fill(key_pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            attn_output_weigths = attn_weights.view(b*self.num_heads, target_len, source_len)
        
        attn_output_weigths_raw = self.softmax(attn_output_weigths)
        attn_output_weigths = self.dropout(attn_output_weigths_raw)

        attn_output = torch.bmm(attn_output_weigths, v)
        attn_output = attn_output.transpose(0,1).contiguous().view(target_len, b, c)
        attn_output = self.out_proj(attn_output)
        
        if get_weights:
            attn_output_weigths_raw = attn_output_weigths_raw.view(b, self.num_heads, target_len, source_len)
            return attn_output, attn_output_weigths_raw.sum(dim=1) / self.num_heads
        
        return attn_output

    def init_weights(self):
        xavier_uniform_(self.in_proj_q.weight)
        xavier_uniform_(self.in_proj_k.weight)
        if self.proj_value:
            xavier_uniform_(self.in_proj_v.weight)

@gin.configurable
class DecoderLayer(nn.Module):

    def __init__(self, d_model, dim_ff, n_heads=4) -> None:
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.ff = dim_ff

        self.input_attention = MHA(embedding_dim=self.d_model,
                             num_heads=n_heads,
                             proj_value=True,
                             dropout=0.1)
        
        self.norm1 = nn.LayerNorm(self.d_model)

        self.cross_attention = MHA(embedding_dim=self.d_model,
                             num_heads=n_heads,
                             proj_value=True,
                             dropout=0.1)

        self.ffNet = nn.Sequential(
            nn.Linear(self.d_model, self.ff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.ff, self.d_model)
        )

        self.dropout = nn.Dropout(0.1)

        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)
    
    def set_lm_mode(self):
        for parameter in self.cross_attention.parameters():
            parameter.requires_grad = False
        
        for parameter in self.norm2.parameters():
            parameter.requires_grad = False
    
    def set_transcription_mode(self):
        for parameter in self.cross_attention.parameters():
            parameter.requires_grad = True
        
        for parameter in self.norm2.parameters():
            parameter.requires_grad = True

    def forward(self, tgt, memory_key, memory_value=None, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                predict_n_last_only=None):
        
        if memory_value is None:
            memory_value = memory_key
        
        mha_q = tgt[-predict_n_last_only:] if predict_n_last_only else tgt

        tgt2, weights_input = self.input_attention(mha_q, tgt, tgt, attn_mask=tgt_mask, key_pad_mask=tgt_key_padding_mask, get_weights=True)
        tgt = mha_q + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        att_query = tgt

        tgt2, weights_cross = self.cross_attention(att_query, memory_key, memory_value, attn_mask=memory_mask, key_pad_mask=memory_key_padding_mask, get_weights=True)

        tgt = att_query + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.ffNet(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, weights_input, weights_cross

    def forward_lm(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, predict_n_last_only=None):
        
        mha_q = tgt[-predict_n_last_only:] if predict_n_last_only else tgt
        tgt2, weights_input = self.input_attention(mha_q, tgt, tgt, attn_mask=tgt_mask, key_pad_mask=tgt_key_padding_mask, get_weights=True)
        tgt = mha_q + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.ffNet(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, weights_input


class DecoderStack(nn.Module):

    def __init__(self, num_dec_layers, d_model, dim_ff) -> None:
        super(DecoderStack, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, dim_ff=dim_ff) for _ in range(num_dec_layers)])
    
    def set_lm_mode(self):
        for layer in self.layers:
            layer.set_lm_mode()
    
    def set_transcription_mode(self):
        for layer in self.layers:
            layer.set_transcription_mode()

    def forward(self, tgt, memory_key, memory_value, tgt_mask, memory_mask, tgt_key_padding_mask, 
                memory_key_padding_mask, use_cache=False, cache=None, predict_last_n_only=False, keep_all_weights=True):

        output = tgt
        cache_t = list()
        all_weights = {
            "self": list(),
            "mix": list()
        }

        for i, dec_layer in enumerate(self.layers):
            output, weights, weights_self = dec_layer(output, memory_key=memory_key,
                                        memory_value=memory_value,
                                        tgt_mask=tgt_mask,
                                        memory_mask=memory_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask,
                                        predict_n_last_only=predict_last_n_only)

            if use_cache:
                cache_t.append(output)
                if cache is not None:
                    output = torch.cat([cache[i], output], dim=0)
        
            if keep_all_weights:
                all_weights["self"].append(weights_self)
                all_weights["mix"].append(weights)

        if use_cache:
            cache = torch.cat([cache, torch.stack(cache_t, dim=0)], dim=1) if cache is not None else torch.stack(cache_t, dim=0)

        if predict_last_n_only:
            output = output[-predict_last_n_only:]

        if keep_all_weights:
            return output, all_weights, cache

        return output, weights, cache
    
    def forward_lm(self, tgt, tgt_mask, tgt_key_padding_mask, use_cache=False, cache=None, predict_last_n_only=False, keep_all_weights=False):

        output = tgt
        cache_t = []
        all_weights = {
            "mix": []
        }

        for i, dec_layer in enumerate(self.layers):
            output, weights = dec_layer.forward_lm(output,
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask,
                                        predict_n_last_only=predict_last_n_only)

            if use_cache:
                cache_t.append(output)
                if cache is not None:
                    output = torch.cat([cache[i], output], dim=0)
        
            if keep_all_weights:
                all_weights["mix"].append(weights)

        if use_cache:
            cache = torch.cat([cache, torch.stack(cache_t, dim=0)], dim=1) if cache is not None else torch.stack(cache_t, dim=0)

        if predict_last_n_only:
            output = output[-predict_last_n_only:]

        if keep_all_weights:
            return output, all_weights, cache

        return output, weights, cache


class Decoder(nn.Module):

    def __init__(self, d_model, dim_ff, n_layers, maxlen, out_categories, attention_window=100) -> None:
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.dec_attn_win = maxlen
        self.positional_1D = PositionalEncoding1D(d_model, maxlen)

        self.decoder = DecoderStack(num_dec_layers=n_layers, d_model=d_model, dim_ff=dim_ff)

        self.embedding = nn.Embedding(num_embeddings=out_categories, embedding_dim=d_model)

        self.end_relu = nn.ReLU()

        self.out_layer = nn.Conv1d(d_model, out_categories, kernel_size=1)
    
    def set_lm_mode(self):
        self.decoder.set_lm_mode()
    
    def set_transcription_mode(self):
        self.decoder.set_transcription_mode()

    def forward(self, raw_features_1D, enhanced_features_1D, tokens, reduced_size, token_len, features_size, start=0, hidden_predict=None, num_pred=None, cache=None, keep_all_weights=True, token_line=None, token_pg=None):
        
        device = raw_features_1D.device
        
        pos_tokens = self.embedding(tokens).permute(0,2,1)

        pos_tokens = self.positional_1D(pos_tokens, start=start)
        pos_tokens = pos_tokens.permute(2,0,1).contiguous()

        if num_pred is None:
            num_pred = tokens.size(1)
        
        if self.dec_attn_win > 1 and cache is not None:
            cache = cache[:, -self.dec_attn_win-1]
        else:
            cache = None
        
        num_tokens_to_keep = num_pred if self.dec_attn_win is None else min([num_pred + self.dec_attn_win - 1, pos_tokens.size(0), token_len[0]])
        pos_tokens = pos_tokens[-num_tokens_to_keep:]

        target_mask = self.generate_target_mask(tokens.size(1), device=device)
        memory_mask = None

        key_target_mask = self.generate_token_mask(token_len, tokens.size(), device)
        key_memory_mask = self.generate_enc_mask(reduced_size, features_size, device)

        target_mask = target_mask[-num_pred:, -num_tokens_to_keep:]
        key_target_mask = key_target_mask[:, -num_tokens_to_keep:]

        output, weights, cache = self.decoder(pos_tokens, memory_key=enhanced_features_1D, memory_value=raw_features_1D, 
                                       tgt_mask=target_mask, memory_mask=memory_mask, tgt_key_padding_mask=key_target_mask, 
                                       memory_key_padding_mask=key_memory_mask, use_cache=True, cache=cache, predict_last_n_only=num_pred, keep_all_weights=keep_all_weights)

        dpoutput = self.dropout(self.end_relu(output))

        predictions = self.out_layer(dpoutput.permute(1,2,0).contiguous())

        if not keep_all_weights:
            weights = torch.sum(weights, dim=1, keepdim=True).reshape(-1, 1, features_size[2], features_size[3])

        return output, predictions, hidden_predict, cache, weights

    def forward_lm(self, tokens, token_len, start=0,
                hidden_predict=None, num_pred=None, cache=None, keep_all_weights=False):
        
        device = tokens.device
        
        pos_tokens = self.embedding(tokens).permute(0,2,1)

        try:
            pos_tokens = self.positional_1D(pos_tokens, start=start)
        except:
            self.positional_1D = PositionalEncoding1D(dim=pos_tokens.size(1), len_max=pos_tokens.size(2))
            pos_tokens = self.positional_1D(pos_tokens, start=start)

        pos_tokens = pos_tokens.permute(2,0,1).contiguous()

        if num_pred is None:
            num_pred = tokens.size(1)
        
        if self.dec_attn_win > 1 and cache is not None:
            cache = cache[:, -self.dec_attn_win-1]
        else:
            cache = None
        
        num_tokens_to_keep = num_pred if self.dec_attn_win is None else min([num_pred + self.dec_attn_win - 1, pos_tokens.size(0), token_len[0]])
        pos_tokens = pos_tokens[-num_tokens_to_keep:]

        target_mask = self.generate_target_mask(tokens.size(1), device=device)
        key_target_mask = self.generate_token_mask(token_len, tokens.size(), device)

        target_mask = target_mask[-num_pred:, -num_tokens_to_keep:]
        key_target_mask = key_target_mask[:, -num_tokens_to_keep:]

        output, _, cache = self.decoder.forward_lm(pos_tokens,
                                       tgt_mask=target_mask, tgt_key_padding_mask=key_target_mask, 
                                       use_cache=True, cache=cache, predict_last_n_only=num_pred, keep_all_weights=keep_all_weights)

        dpoutput = self.dropout(self.end_relu(output))

        predictions = self.out_layer(dpoutput.permute(1,2,0).contiguous())

        return output, predictions, hidden_predict, cache

    def generate_enc_mask(self, batch_reduced_size, total_size, device):
        batch_size, _, h_max, w_max = total_size
        mask = torch.ones((batch_size, h_max, w_max), dtype=torch.bool, device=device)
        for i, (h, w) in enumerate(batch_reduced_size):
            mask[i, :h, :w] = False
        return torch.flatten(mask, start_dim=1, end_dim=2)

    def generate_token_mask(self, token_len, total_size, device):
        batch_size, len_mask = total_size
        mask = torch.zeros((batch_size, len_mask), dtype=torch.bool, device=device)
        for i, len_ in enumerate(token_len):
            mask[i, :len_] = False
        
        return mask
    
    def generate_target_mask(self, target_len, device):
        if self.dec_attn_win == 1:
            return torch.triu(torch.ones((target_len, target_len), dtype=torch.bool, device=device), diagonal=1)
        else:
            return torch.logical_not(
                torch.logical_and(torch.tril(torch.ones((target_len, target_len), dtype=torch.bool, device=device), diagonal=0),
                                  torch.triu(torch.ones((target_len, target_len), dtype=torch.bool, device=device), diagonal=-self.dec_attn_win+1)))

if __name__ == "__main__":
    pass
