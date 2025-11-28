import torch
import torch.nn as nn
from .blocks import DownBlock, UpBlock, MiddleBlock 

class VQVAE(nn.Module):
    def __init__(self, im_channels, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']

        self.attns = model_config['attn_down']

        self.z_channels = model_config['z_channels']
        self.codebook_size = model_config['codebook_size']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']

        assert self.mid_channels[0] ==self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        self.up_sample = list(reversed(self.down_sample))

        self.encoder_conv_in = nn.Conv2d(im_channels, self.down_channels[0],kernel_size = 3, padding = (1,1))

        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.encoder_layers.append(DownBlock(self.down_channels[i], 
            self.down_channels[i+1], t_emb_dim = None, down_sample = self.down_sample[i], num_down_layers = self.num_down_layers[i], attn = self.attns[i], norm_channels = self.norm_channels, num_heads = self.num_heads))

        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.encoder_mids.append(MiddleBlock(self.mid_channels[i], self.mid_channels[i+1], t_emb_dim = None, num_layers = self.num_mid_layers, norm_channels = self.norm_channels, num_heads = self.num_heads))    
        
        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], self.z_channels, kernel_size=3, padding=1)
        
        self.pre_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)

        self.embedding = nn.Embedding(self.codebook_size, self.z_channels)

        
