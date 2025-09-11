import torch
from torch import nn

try:
    from torchdiffeq import odeint
except Exception as e:
    odeint = None


class ODEFunc(nn.Module):
    """Latent ODE function operating on feature maps (B, C, H, W) via 1x1 convs."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 2):
        super().__init__()
        layers = []
        c_in = in_channels
        for _ in range(max(1, num_layers - 1)):
            layers.append(nn.Conv2d(c_in, hidden_channels, kernel_size=1))
            layers.append(nn.GELU())
            c_in = hidden_channels
        layers.append(nn.Conv2d(c_in, in_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, t, x):  # t unused for autonomous system
        return self.net(x)


class LatentNeuralODE(nn.Module):
    """Integrate latent feature maps over time using Neural ODE.

    Inputs:
        x0: Tensor (B, C, H, W)
        steps: int, number of future steps to generate
    Returns:
        xs: Tensor (B, steps, C, H, W)
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 2,
                 method: str = 'rk4', step_size: float = 1.0):
        super().__init__()
        self.func = ODEFunc(in_channels, hidden_channels, num_layers)
        self.method = method
        self.step_size = step_size

    def forward(self, x0: torch.Tensor, steps: int) -> torch.Tensor:
        assert odeint is not None, 'torchdiffeq is required for LatentNeuralODE'
        device = x0.device
        # times include initial time 0; we will drop the first state
        t_end = float(self.step_size * steps)
        tt = torch.linspace(0.0, t_end, steps + 1, device=device, dtype=x0.dtype)
        x_path = odeint(self.func, y0=x0, t=tt, method=self.method)
        # x_path: (steps+1, B, C, H, W); drop initial state
        x_future = x_path[1:]
        x_future = x_future.transpose(0, 1).contiguous()  # (B, steps, C, H, W)
        return x_future


