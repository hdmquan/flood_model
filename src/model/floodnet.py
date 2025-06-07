import torch
import torch.nn as nn
import torch.nn.functional as F


class FloodNet(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=64):
        super(FloodNet, self).__init__()
        # GELU because it's smoother than ReLU
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x, dem, rainfall, dt=1.0, dx=1.0):
        features = self.encoder(x)
        h_pred = self.decoder(features)
        return self.calc_loss(h_pred, dem, rainfall, dt, dx)

    def calc_loss(self, h_pred, dem, rainfall, dt=1.0, dx=1.0):
        """
        Physics-informed loss using simplified shallow water continuity equation:
        d(h)/dt + d(hu)/dx + d(hv)/dy = R (rainfall input)
        """
        # Central difference
        dh_dx = (h_pred[:, :, :, 2:] - h_pred[:, :, :, :-2]) / (2 * dx)
        dh_dy = (h_pred[:, :, 2:, :] - h_pred[:, :, :-2, :]) / (2 * dx)

        # Simulate flow velocity (roughly proportional to slope)
        slope_x = (dem[:, :, :, 2:] - dem[:, :, :, :-2]) / (2 * dx)
        slope_y = (dem[:, :, 2:, :] - dem[:, :, :-2, :]) / (2 * dx)

        u = -slope_x
        v = -slope_y

        # Estimated flow components
        hu = h_pred[:, :, :, 1:-1] * u
        hv = h_pred[:, :, 1:-1, :] * v

        d_hu_dx = (hu[:, :, :, 1:] - hu[:, :, :, :-1]) / dx
        d_hv_dy = (hv[:, :, 1:, :] - hv[:, :, :-1, :]) / dx

        divergence = d_hu_dx[:, :, :-1, :] + d_hv_dy[:, :, :, :-1]

        # Rainfall needs to be cropped to match the central difference shape
        rainfall_crop = rainfall[:, :, 1:-1, 1:-1]

        # Mass conservation loss
        continuity_residual = (divergence + rainfall_crop).pow(2)
        continuity_loss = continuity_residual.mean()

        return continuity_loss


# tensors = terrain.to_tensor(device="cuda")
# x_input = torch.stack(
#     [
#         tensors["dem"],
#         tensors["slope"],
#         tensors["flow_acc"],
#         tensors["roads"],
#         tensors["water"],
#         tensors["hand"],
#     ],
#     dim=0,
# ).unsqueeze(
#     0
# )  # [1, 6, H, W]

# rainfall_tensor = (
#     torch.rand_like(tensors["dem"]).unsqueeze(0).unsqueeze(0)
# )  # [1, 1, H, W]
