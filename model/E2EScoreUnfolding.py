import torch.nn as nn
import torch.nn.functional as F
import torch
import random
from torchinfo import summary
import gin
from loguru import logger

class DepthSepConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding=True, stride=(1,1), dilation=(1,1)):
        super(DepthSepConv2D, self).__init__()

        self.padding = None
        
        if padding:
            if padding is True:
                padding = [int((k-1)/2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h//2, padding_h-padding_h//2, padding_w//2, padding_w-padding_w//2]
                    padding = (0, 0)

        else:
            padding = (0, 0)

        self.depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=(1,1))
        self.activation = activation

    def forward(self, inputs):
        x = self.depth_conv(inputs)
        if self.padding:
            x = F.pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        
        x = self.point_conv(x)

        return x

class MixDropout(nn.Module):
    def __init__(self, dropout_prob=0.4, dropout_2d_prob=0.2):
        super(MixDropout, self).__init__()

        self.dropout = nn.Dropout(dropout_prob)
        self.dropout2D = nn.Dropout2d(dropout_2d_prob)
    
    def forward(self, inputs):
        if random.random() < 0.5:
            return self.dropout(inputs)
        return self.dropout2D(inputs)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=(1,1), kernel=3, activation=nn.ReLU, dropout=0.4):
        super(ConvBlock, self).__init__()

        self.activation = activation()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel, padding=kernel//2)
        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=kernel, padding=kernel//2)
        self.conv3 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3,3), padding=(1,1), stride=stride)
        self.normLayer = nn.InstanceNorm2d(num_features=out_c, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_prob=dropout, dropout_2d_prob=dropout/2)

    def forward(self, inputs):
        pos = random.randint(1,3)

        x = self.conv1(inputs)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)
        
        x = self.normLayer(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        
        return x

class DSCBlock(nn.Module):

    def __init__(self, in_c, out_c, stride=(2, 1), activation=nn.ReLU, dropout=0.4):
        super(DSCBlock, self).__init__()

        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_c, out_c, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_c, out_c, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(out_c, out_c, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = nn.InstanceNorm2d(out_c, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_prob=dropout, dropout_2d_prob=dropout/2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)
        #x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        return x


class Encoder(nn.Module):

    def __init__(self, in_channels, dropout=0.4):
        super(Encoder, self).__init__()

        self.conv_blocks = nn.ModuleList([
            ConvBlock(in_c=in_channels, out_c=32, stride=(1,1), dropout=dropout),
            ConvBlock(in_c=32, out_c=64, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=64, out_c=128, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=128, out_c=256, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=256, out_c=512, stride=(2,1), dropout=dropout)
        ])

        self.dscblocks = nn.ModuleList([
            DSCBlock(in_c=512, out_c=512, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=512, out_c=512, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=512, out_c=512, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=512, out_c=512, stride=(1,1), dropout = dropout)
        ])
    
    def forward(self, x):
        for layer in self.conv_blocks:
            x = layer(x)
        
        for layer in self.dscblocks:
            xt = layer(x)
            x = x + xt if x.size() == xt.size() else xt

        return x

class StaveRNNDecoder(nn.Module):
    def __init__(self, img_height, height_reduction, out_channels, out_categories) -> None:
        super().__init__()
        features = (img_height // height_reduction) * out_channels
        self.dec_lstm = nn.LSTM(input_size=features, hidden_size=256, bidirectional=True, batch_first=True)
        self.out_dense = nn.Linear(in_features=512, out_features=out_categories)
        self.reshape_features = (img_height // height_reduction) * out_channels

    def forward(self, x):
        b, _, _, _ = x.size()
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(b, -1, self.reshape_features)
        x, _ = self.dec_lstm(x)
        x = self.out_dense(x)
        x = x.permute(1,0,2)
        return F.log_softmax(x, dim=-1)

class StaveTransformerDecoder(nn.Module):
    def __init__(self, img_height, height_reduction, out_channels, out_cats, max_len) -> None:
        super().__init__()
        features = (img_height // height_reduction) * out_channels
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.projection_layer = nn.Linear(features, 512)

        self.pos_encoding = PositionalEncoding1D(dim=512, len_max=max_len, device=self.dummy_param.device)
        transf_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, batch_first=True)
        self.dec_transf = nn.TransformerEncoder(transf_layer, num_layers=1)

        self.reshape_features = (img_height // height_reduction) * out_channels
        self.out_dense = nn.Linear(in_features=512, out_features=out_cats)

    def forward(self, x):
        b, _, _, _ = x.size()
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.reshape(b, -1, self.reshape_features)
        x = self.projection_layer(x)
        x = self.pos_encoding(x.permute(0, 2, 1).contiguous())
        x = x.permute(0,2,1).contiguous()
        x = self.dec_transf(x)
        x = self.out_dense(x)
        x = x.permute(1,0,2).contiguous()
        return F.log_softmax(x, dim=-1)


class PageDecoder(nn.Module):

    def __init__(self, out_cats):
        super(PageDecoder, self).__init__()
        self.dec_conv = nn.Conv2d(in_channels= 512, out_channels= out_cats, kernel_size=(5,5), padding=(2,2))
    
    def forward(self, inputs):
        x = self.dec_conv(inputs)
        x = F.log_softmax(x, dim=1)
        b, c, h, w = x.size()
        x = x.reshape(b, c, h*w)
        x = x.permute(2,0,1)
        return x
    
class E2EScore_FCN(nn.Module):

    def __init__(self, in_channels, out_cats):
        super(E2EScore_FCN, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = PageDecoder(out_cats=out_cats)
    
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

class RecurrentScoreUnfolding(nn.Module):

    def __init__(self, out_cats):
        super(RecurrentScoreUnfolding, self).__init__()
        self.dec_lstm = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)
        self.out_dense = nn.Linear(in_features=512, out_features=out_cats)
    
    def forward(self, inputs):
        x = inputs
        b, c, h, w = x.size()
        x = x.reshape(b, c, h*w)
        x = x.permute(0,2,1)
        x, _ = self.dec_lstm(x)
        x = self.out_dense(x)
        x = x.permute(1,0,2)
        return F.log_softmax(x, dim=2)

class PositionalEncoding1D(nn.Module):

    def __init__(self, dim, len_max, device):
        super(PositionalEncoding1D, self).__init__()
        self.len_max = len_max
        self.dim = dim
        self.pe = torch.zeros((1, dim, len_max), device=device, requires_grad=False)

        div = torch.exp(-torch.arange(0., dim, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        l_pos = torch.arange(0., len_max)
        self.pe[:, ::2, :] = torch.sin(l_pos * div).unsqueeze(0)
        self.pe[:, 1::2, :] = torch.cos(l_pos * div).unsqueeze(0)

    def forward(self, x, start=0):
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
            return 

class TransformerScoreUnfolding(nn.Module):

    def __init__(self, out_cats, max_len):
        super(TransformerScoreUnfolding, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.pos_encoding = PositionalEncoding1D(dim=512, len_max=max_len, device=self.dummy_param.device)
        transf_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, batch_first=True)
        self.dec_transf = nn.TransformerEncoder(transf_layer, num_layers=1)
        self.out_dense = nn.Linear(in_features=512, out_features=out_cats)

    def forward(self, inputs):
        x = inputs
        b, c, h, w = x.size()
        x = x.reshape(b, c, h*w)
        x = self.pos_encoding(x)
        x = x.permute(0,2,1)
        x = self.dec_transf(x)
        x = self.out_dense(x)
        x = x.permute(1,0,2)
        return F.log_softmax(x, dim=2)

class E2EScore_CRNN(nn.Module):

    def __init__(self, in_channels, out_cats, pretrain_path=None):
        super(E2EScore_CRNN, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)

        if pretrain_path != None:
            print(f"Loading weights from {pretrain_path}")
            self.encoder.load_state_dict(torch.load(pretrain_path), strict=True)

        self.decoder = RecurrentScoreUnfolding(out_cats=out_cats)
    
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

class E2EScore_CNNT(nn.Module):

    def __init__(self, in_channels, out_cats, max_len, pretrain_path=None):
        super(E2EScore_CNNT, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)

        if pretrain_path != None:
            print(f"Loading weights from {pretrain_path}")
            self.encoder.load_state_dict(torch.load(pretrain_path), strict=True)

        self.decoder = TransformerScoreUnfolding(out_cats=out_cats, max_len=max_len)
    
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

@gin.configurable
class E2EStave_CRNN(nn.Module):

    def __init__(self, in_channels, out_cats, pretrain_path=None, img_height=None, height_reduction=None, out_channels=None):
        super(E2EStave_CRNN, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)

        if pretrain_path != None:
            print(f"Loading weights from {pretrain_path}")
            self.encoder.load_state_dict(torch.load(pretrain_path), strict=True)

        self.decoder = StaveRNNDecoder(img_height, height_reduction, out_channels, out_categories=out_cats)
    
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x


@gin.configurable
class E2EStave_CNNT(nn.Module):

    def __init__(self, in_channels, out_cats, pretrain_path=None, img_height=None, height_reduction=None, out_channels=None):
        super(E2EStave_CRNN, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)

        if pretrain_path != None:
            print(f"Loading weights from {pretrain_path}")
            self.encoder.load_state_dict(torch.load(pretrain_path), strict=True)

        self.decoder = StaveRNNDecoder(img_height, height_reduction, out_channels, out_categories=out_cats)
    
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

def get_fcn_model(maxwidth, maxheight, in_channels, out_size, maxlen=None):
    model = E2EScore_FCN(in_channels=in_channels, out_cats=out_size)
    summary(model, input_size=[(1,in_channels,maxheight,maxwidth)], dtypes=[torch.float])
    
    return model

def get_rcnn_model(maxwidth, maxheight, in_channels, out_size, weights_path=None, maxlen=None):
    model = E2EScore_CRNN(in_channels=in_channels, out_cats=out_size)
    summary(model, input_size=[(1,in_channels,maxheight,maxwidth)], dtypes=[torch.float])
    
    return model

def get_cnntrf_model(maxwidth, maxheight, in_channels, out_size, maxlen=None):
    model = E2EScore_CNNT(in_channels=in_channels, out_cats=out_size, max_len=4092)
    summary(model, input_size=[(1,in_channels,maxheight,maxwidth)], dtypes=[torch.float])
    #with torch.no_grad():
    #    output = model(torch.randn((1,in_channels,maxheight,maxwidth)))
    #    import sys
    #    sys.exit()
    return model

def get_stave_crnn(maxwidth, maxheight, in_channels, out_size, maxlen=None):
    model = E2EStave_CRNN(in_channels=in_channels, out_cats=out_size)
    summary(model, input_size=[(1,in_channels,maxheight,maxwidth)], dtypes=[torch.float])

    return model

def get_stave_cnnt(maxwidth, maxheight, in_channels, out_size, maxlen=None):
    model = E2EScore_CNNT(in_channels=in_channels, out_cats=out_size, max_len=maxlen)
    
    #summary(model, input_size=[(1,in_channels,maxheight,maxwidth)], dtypes=[torch.float])

    return model
