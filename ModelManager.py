import gin

import numpy as np

import torch
import torch.nn as nn
import lightning.pytorch as L

from transformers import SwinConfig, SwinModel
from itertools import groupby
from model.DANEncoder import Encoder
from model.ConvNextEncoder import ConvNextEncoder
from model.DANDecoder import Decoder
from model.E2EScoreUnfolding import get_cnntrf_model
from torchinfo import summary
from eval_functions import compute_poliphony_metrics

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

@gin.configurable
class DAN(L.LightningModule):
    def __init__(self, maxh, maxw, maxlen, out_categories, padding_token, in_channels, w2i, i2w, out_dir, d_model=None, dim_ff=None, num_dec_layers=None, encoder_type="Normal", swin_image_size=(256,800)) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        print(swin_image_size)
        if encoder_type == "NexT":
            self.encoder = ConvNextEncoder(in_chans=in_channels, depths=[3,3,9], dims=[64, 128, 256])
        if encoder_type == "Swin":
            config = SwinConfig(image_size=swin_image_size, embed_dim=64, num_channels=in_channels, depths=[2,2,6] , num_heads=[4, 8, 16])
            self.encoder = SwinModel(config, add_pooling_layer=False)
        else:
            self.encoder = Encoder(in_channels=in_channels)

        self.decoder = Decoder(d_model, dim_ff, num_dec_layers, maxlen, out_categories)
        self.positional_2D = PositionalEncoding2D(d_model, maxh, maxw)

        self.padding_token = padding_token

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token)

        self.eximgs = []
        self.expreds = []
        self.exgts = []

        self.valpredictions = []
        self.valgts = []

        self.w2i = w2i
        self.i2w = i2w
        self.maxlen = maxlen
        self.out_dir=out_dir

        self.save_hyperparameters()

    def forward(self, x, y_pred):
        if self.encoder_type == "Swin":
            encoder_output = self.encoder(x).last_hidden_state
            b, _, _ = encoder_output.size()
            reduced_size = [s.shape[1] for s in encoder_output]
            ylens = [len(sample) for sample in y_pred]
            cache = None

            features = encoder_output.permute(1, 0, 2).contiguous()
            enhanced_features = encoder_output.permute(1, 0, 2).contiguous()

            output, predictions, _, _, weights = self.decoder(features, enhanced_features, y_pred[:, :-1], reduced_size, 
                                                               [max(ylens) for _ in range(b)], encoder_output.size(), 
                                                               start=0, cache=cache, keep_all_weights=True, is_swin_output=True)
        else:
            encoder_output = self.encoder(x)
            b, c, h, w = encoder_output.size()
            reduced_size = [s.shape[:2] for s in encoder_output]
            ylens = [len(sample) for sample in y_pred]
            cache = None

            pos_features = self.positional_2D(encoder_output)
            features = torch.flatten(encoder_output, start_dim=2, end_dim=3).permute(2,0,1)
            enhanced_features = features
            enhanced_features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2,0,1)
        
            output, predictions, _, _, weights = self.decoder(features, enhanced_features, y_pred[:, :-1], reduced_size, 
                                                               [max(ylens) for _ in range(b)], encoder_output.size(), 
                                                               start=0, cache=cache, keep_all_weights=True, is_swin_output=False)
    
        return output, predictions, cache, weights


    def forward_encoder(self, x):
        if self.encoder_type == "Swin":
            return self.encoder(x).last_hidden_state
        return self.encoder(x)
    
    def forward_decoder(self, encoder_output, last_preds, cache=None):
        if self.encoder_type == "Swin":
            b, _, _ = encoder_output.size()
            reduced_size = [s.shape[1] for s in encoder_output]
            ylens = [len(sample) for sample in last_preds]
            cache = cache

            features = encoder_output.permute(1, 0, 2).contiguous()
            enhanced_features = encoder_output.permute(1, 0, 2).contiguous()
            output, predictions, _, _, weights = self.decoder(features, enhanced_features, last_preds[:, :], reduced_size, 
                                                           [max(ylens) for _ in range(b)], encoder_output.size(), 
                                                           start=0, cache=cache, keep_all_weights=True, is_swin_output=True)
        else:
            b, c, h, w = encoder_output.size()
            reduced_size = [s.shape[:2] for s in encoder_output]
            ylens = [len(sample) for sample in last_preds]
            cache = cache

            pos_features = self.positional_2D(encoder_output)
            features = torch.flatten(encoder_output, start_dim=2, end_dim=3).permute(2,0,1)
            enhanced_features = features
            enhanced_features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2,0,1)
            output, predictions, _, _, weights = self.decoder(features, enhanced_features, last_preds[:, :], reduced_size, 
                                                           [max(ylens) for _ in range(b)], encoder_output.size(), 
                                                           start=0, cache=cache, keep_all_weights=True, is_swin_output=False)
    
        return output, predictions, cache, weights
    
    def configure_optimizers(self):
        return torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-4, amsgrad=False)

    def training_step(self, train_batch):
        x, di, y = train_batch
        output, predictions, cache, weights = self.forward(x, di)
        loss = self.loss(predictions, y[:, :-1])
        self.log('loss', loss, on_epoch=True, batch_size=1, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x, _, y = val_batch
        encoder_output = self.forward_encoder(x)
        predicted_sequence = torch.from_numpy(np.asarray([self.w2i['<bos>']])).to(device).unsqueeze(0)
        cache = None
        for i in range(self.maxlen):
             output, predictions, cache, weights = self.forward_decoder(encoder_output, predicted_sequence.long(), cache=cache)
             predicted_token = torch.argmax(predictions[:, :, -1]).item()
             predicted_sequence = torch.cat([predicted_sequence, torch.argmax(predictions[:, :, -1], dim=1, keepdim=True)], dim=1)
             if predicted_token == self.w2i['<eos>']:
                 break
        
        dec = "".join([self.i2w[token.item()] for token in predicted_sequence.squeeze(0)[1:]])
        dec = dec.replace("<t>", "\t")
        dec = dec.replace("<b>", "\n")
        dec = dec.replace("<s>", " ")

        gt = "".join([self.i2w[token.item()] for token in y.squeeze(0)[:-1]])
        gt = gt.replace("<t>", "\t")
        gt = gt.replace("<b>", "\n")
        gt = gt.replace("<s>", " ")

        self.valpredictions.append(dec)
        self.valgts.append(gt)
    
    
    def test_step(self, test_batch, batch_idx):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x, _, y = test_batch
        encoder_output = self.forward_encoder(x)
        predicted_sequence = torch.from_numpy(np.asarray([self.w2i['<bos>']])).to(device).unsqueeze(0)
        cache = None
        for i in range(self.maxlen):
             output, predictions, cache, weights = self.forward_decoder(encoder_output, predicted_sequence.long(), cache=cache)
             predicted_token = torch.argmax(predictions[:, :, -1]).item()
             predicted_sequence = torch.cat([predicted_sequence, torch.argmax(predictions[:, :, -1], dim=1, keepdim=True)], dim=1)
             if predicted_token == self.w2i['<eos>']:
                 break
        
        dec = "".join([self.i2w[token.item()] for token in predicted_sequence.squeeze(0)[1:]])
        dec = dec.replace("<t>", "\t")
        dec = dec.replace("<b>", "\n")
        dec = dec.replace("<s>", " ")

        gt = "".join([self.i2w[token.item()] for token in y.squeeze(0)[:-1]])
        gt = gt.replace("<t>", "\t")
        gt = gt.replace("<b>", "\n")
        gt = gt.replace("<s>", " ")

        self.valpredictions.append(dec)
        self.valgts.append(gt)

class Poliphony_DAN(DAN):
    def __init__(self, maxh, maxw, maxlen, out_categories, padding_token, in_channels, w2i, i2w, out_dir) -> None:
        super().__init__(maxh, maxw, maxlen, out_categories, padding_token, in_channels, w2i, i2w, out_dir)
    
    def on_validation_epoch_end(self):
        cer, ser, ler = compute_poliphony_metrics(self.valpredictions, self.valgts)
        
        random_index = np.random.randint(0, len(self.valpredictions))
        predtoshow = self.valpredictions[random_index]
        gttoshow = self.valgts[random_index]
        print(f"[Prediction] - {predtoshow}")
        print(f"[GT] - {gttoshow}")

        self.log('val_CER', cer, prog_bar=True)
        self.log('val_SER', ser, prog_bar=True)
        self.log('val_LER', ler, prog_bar=True)

        self.valpredictions = []
        self.valgts = []

        return ser

    def on_test_epoch_end(self):
        cer, ser, ler = compute_poliphony_metrics(self.valpredictions, self.valgts)

        #for index, sample in enumerate(self.valpredictions):
        #    with open(f"{self.out_dir}/hyp/{index}.krn", "w+") as krnfile:
        #        krnfile.write(sample)
        #
        #for index, sample in enumerate(self.valgts):
        #    with open(f"{self.out_dir}/gt/{index}.krn", "w+") as krnfile:
        #        krnfile.write(sample)

        self.log('test_CER', cer)
        self.log('test_SER', ser)
        self.log('test_LER', ler)

        self.valpredictions = []
        self.valgts = []

        return ser

class LighntingE2EModelUnfolding(L.LightningModule):
    def __init__(self, model, blank_idx, i2w, output_path) -> None:
        super(LighntingE2EModelUnfolding, self).__init__()
        self.model = model
        self.loss = nn.CTCLoss(blank=blank_idx)
        self.blank_idx = blank_idx
        self.i2w = i2w
        self.accum_ed = 0
        self.accum_len = 0
        
        self.dec_val_ex = []
        self.gt_val_ex = []

        self.out_path = output_path

        self.save_hyperparameters()

    def forward(self, input):
        return self.model(input)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def training_step(self, train_batch, batch_idx):
         X_tr, Y_tr, L_tr, T_tr = train_batch
         predictions = self.forward(X_tr)
         loss = self.loss(predictions, Y_tr, L_tr, T_tr)
         self.log('loss', loss, on_epoch=True, batch_size=1, prog_bar=True)
         return loss

    def compute_prediction(self, batch):
        X, Y, _, _ = batch
        pred = self.forward(X)
        pred = pred.permute(1,0,2).contiguous()
        pred = pred[0]
        out_best = torch.argmax(pred,dim=1)
        out_best = [k for k, g in groupby(list(out_best))]
        decoded = []
        for c in out_best:
            if c.item() != self.blank_idx:
                decoded.append(c.item())
        
        decoded = [self.i2w[tok] for tok in decoded]
        gt = [self.i2w[int(tok.item())] for tok in Y[0]]

        return decoded, gt

    def validation_step(self, val_batch, batch_idx):
        dec, gt = self.compute_prediction(val_batch)
        
        dec = "".join(dec)
        dec = dec.replace("<t>", "\t")
        dec = dec.replace("<b>", "\n")
        dec = dec.replace("<s>", " ")

        gt = "".join(gt)
        gt = gt.replace("<t>", "\t")
        gt = gt.replace("<b>", "\n")
        gt = gt.replace("<s>", " ")

        self.dec_val_ex.append(dec)
        self.gt_val_ex.append(gt)

    def on_validation_epoch_end(self):        
        
        cer, ser, ler = compute_poliphony_metrics(self.dec_val_ex, self.gt_val_ex)

        self.log('val_CER', cer)
        self.log('val_SER', ser)
        self.log('val_LER', ler)

        return ser

    def test_step(self, test_batch, batch_idx):
        dec, gt = self.compute_prediction(test_batch)
        
        dec = "".join(dec)
        dec = dec.replace("<t>", "\t")
        dec = dec.replace("<b>", "\n")
        dec = dec.replace("<s>", " ")

        gt = "".join(gt)
        gt = gt.replace("<t>", "\t")
        gt = gt.replace("<b>", "\n")
        gt = gt.replace("<s>", " ")

        self.dec_val_ex.append(dec)
        self.gt_val_ex.append(gt)
    
    def on_test_epoch_end(self) -> None:
        cer, ser, ler = compute_poliphony_metrics(self.dec_val_ex, self.gt_val_ex)

        self.log('val_CER', cer)
        self.log('val_SER', ser)
        self.log('val_LER', ler)

        self.gt_val_ex = []
        self.dec_val_ex = []

        return ser

def get_model(maxwidth, maxheight, in_channels, out_size, blank_idx, i2w, model_name, output_path):
    model = get_cnntrf_model(maxwidth, maxheight, in_channels, out_size)
    lighningModel = LighntingE2EModelUnfolding(model=model, blank_idx=blank_idx, i2w=i2w, output_path=output_path)
    summary(lighningModel, input_size=([1, in_channels, maxheight, maxwidth]))
    return lighningModel, model

    
def get_DAN_network(in_channels, max_height, max_width, max_len, out_categories, w2i, i2w, out_dir, model_name=None):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Poliphony_DAN(in_channels=in_channels, maxh=(max_height//16)+1, maxw=(max_width//8)+1, 
                maxlen=max_len+1, out_categories=out_categories, 
                padding_token=0, w2i=w2i, i2w=i2w, out_dir=out_dir).to(device)
    
    #with torch.no_grad():
    #    print(max_height, max_width, max_len)
    #    _ = model(torch.randn((1,1,max_height,max_width), device=device), torch.randint(low=0, high=100,size=(1,max_len), device=device).long())
    #import sys
    #sys.exit()
    summary(model, input_size=[(1,1,max_height,max_width), (1,max_len)], dtypes=[torch.float, torch.long])

    return model
