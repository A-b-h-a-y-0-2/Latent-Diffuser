import torch
import torch.nn as nn

def get_time_embeddings(time_steps, temb_dimensions):
    r"""converting time steps tensor into time embeddings using the sinusodial time embedding formula
    
    Args:
        time_steps (torch.Tensor): 1D tensor of length batch size.
        temb_dimensions (int): Dimensions of embeddings.
    Returns:
        B*D embedding representation of B time steps.
    """
    assert temb_dimensions %2 ==0, "time embedding dimensions must be divisible by 2"

    factor = 10000 ** (torch.arange(start = 0, end = temb_dimensions//2, step = 2, dtype = torch.float32, device = time_steps.device)/(temb_dimensions//2))
    t_emb = time_steps[:,None].repeat(1,temb_dimensions//2)/factor #Creates a matrix where each row represents a time step
    #Each column contains the time step scaled by a different frequency
    #These scaled values will be fed into sin/cos functions to create unique, smooth embeddings that help the model understand the progression of the diffusion process
    t_emb = torch.cat([torch.sin(t_emb),torch.cos(t_emb)],dim = -1)
    return t_emb

class DownBlock(nn.Module):
    r"""Down conv block with attention, with the following sequence:
        Resnet -> Self-Attention -> Downsample"""
    
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads, num_layers, attn, norm_channels, cross_attn = False,
    context_dim = None):
        super().__init__()
        self.down_sample = down_sample
        self.num_layers = num_layers
        self.attn = attn
        self.cross_attn = cross_attn
        self.context_dim = context_dim
        self.t_emb_dim = t_emb_dim
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            ) 
            for i in range(self.num_layers)
        ]
        )        
        if self.t_emb_dim is not None:
            self.t_emb_layer = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.t_emb_dim, out_channels)
                )
                for _ in range(self.num_layers)
            ])
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
            nn.GroupNorm(norm_channels, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            )
            for _ in range(self.num_layers)
        ])
        if self.attn is not None:
            self.attention_norms = nn.ModuleList([
                nn.GroupNorm(norm_channels, out_channels)
                for _ in range(self.num_layers)
            ])
            self.attentions = nn.ModuleList([
                nn.MultiHeadAttention(out_channels, num_heads, batch_first = True)
                for _ in range(self.num_layers)
            ])
        if self.cross_attn is not None:
            assert context_dim is not None, "Context dimension must be specified for cross attention"
            self.cross_attention_norms = nn.ModuleList([
                nn.GroupNorm(norm_channels, out_channels)
                for _ in range(self.num_layers)
            ])
            self.cross_attentions = nn.ModuleList([
                nn.MultiHeadAttention(out_channels, num_heads, batch_first = True)
                for _ in range(self.num_layers)
            ])
            self.context_proj = nn.ModuleList([
                nn.Linear(context_dim, out_channels)
                    for _ in range(self.num_layers)
                ])
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i ==0 else out_channels, out_channels, kernel_size = 1, stride = 1)
            for i in range(self.num_layers)
        ])
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels,4,2,1) if self.down_sample else nn.Identity()
            