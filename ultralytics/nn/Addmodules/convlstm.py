import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias)
        self.hidden_dim = hidden_dim

    def forward(self, x_seq, h_c_init=None):
        # x_seq: (B, T, C, H, W)
        B, T, C, H, W = x_seq.shape
        device = x_seq.device
        h, c = h_c_init if h_c_init is not None else (
            torch.zeros(B, self.hidden_dim, H, W, device=device),
            torch.zeros(B, self.hidden_dim, H, W, device=device)
        )
        outputs = []
        for t in range(T):
            h, c = self.cell(x_seq[:, t], h, c)
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1), (h, c) 