import torch
import torch.nn as nn
import torch.nn.functional as F

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        # self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x # if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)
    
class TSCP2(nn.Module):
    # TODO: Causal TCN and kernel_size=4, window=100 were used
    # TODO: Normalize repr after projector
    def __init__(self, input_dims, hidden_dims=64, window_size=100, output_dims=64, depth=4):
        super().__init__()
        self.input_fc = nn.Conv1d(input_dims, hidden_dims, 1)
        self.hidden_dims = hidden_dims
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth,
            kernel_size=3
        )
        self.flat = nn.Flatten()
        self.projector = nn.Sequential(
            nn.Linear(window_size*hidden_dims, 2*window_size),
            nn.ReLU(),
            nn.Linear(2*window_size, window_size),
            nn.ReLU(),
            nn.Linear(window_size, output_dims)
        )

        
    def forward(self, x):  # x: B x T x input_dims
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.input_fc(x)
        x = self.feature_extractor(x)  # B x Co x T
        x = self.flat(x)
        x = self.projector(x)
        return x

class TNC(nn.Module):
    def __init__(self, input_dims, hidden_dims=64, window_size=100, output_dims=64, depth=4):
        super().__init__()
        self.input_fc = nn.Conv1d(input_dims, hidden_dims, 1)
        self.hidden_dims = hidden_dims
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth,
            kernel_size=3
        )
        self.flat = nn.Flatten()
        self.projector = nn.Sequential(
            nn.Linear(window_size*hidden_dims, 2*window_size),
            nn.ReLU(),
            nn.Linear(2*window_size, window_size),
            nn.ReLU(),
            nn.Linear(window_size, output_dims)
        )

        self.discriminator = torch.nn.Sequential( # discriminator for concatenated input (anchor, pos) or (anchor, neg)
            torch.nn.Linear(2*output_dims, 4*output_dims),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4*output_dims, 1)
        )

        
    def forward(self, x):  # x: B x T x input_dims
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.input_fc(x)
        x = self.feature_extractor(x)  # B x Co x T
        x = self.flat(x)
        x = self.projector(x)
        return x
    
    def disc(self, x, x_prime):
        # print(x.shape, x_prime.shape)
        x_concat = torch.concat([x,x_prime], dim=1)
        return self.discriminator(x_concat)


def instance_contrastive_loss(z1, z2):
    # each timestamp of z1 has positive timestamp in z2 at the same position
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss