import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.nn as nn
import torch.nn.functional as F


class FloodNet(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=64):
        super(FloodNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels + 1, hidden_dim, kernel_size=3, padding=1
            ),  # +1 for previous h
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
            nn.ReLU(),  # ensures non-negative water height
        )

    def forward(self, x, h_prev):
        x_in = torch.cat([x, h_prev], dim=1)
        features = self.encoder(x_in)
        h_pred = self.decoder(features)
        return h_pred

    def calc_loss(self, h_pred, dem, rainfall, dt=1.0, dx=1.0):
        """
        Physics-informed loss using simplified shallow water continuity equation:
        d(h)/dt + d(hu)/dx + d(hv)/dy = R (rainfall input)
        """
        # Compute slope-based velocities
        slope_x = (dem[:, :, :, 2:] - dem[:, :, :, :-2]) / (2 * dx)
        slope_y = (dem[:, :, 2:, :] - dem[:, :, :-2, :]) / (2 * dx)
        u = -slope_x
        v = -slope_y

        # Crop h_pred for multiplication
        h_cropped_x = h_pred[:, :, :, 1:-1]  # shape matches u
        h_cropped_y = h_pred[:, :, 1:-1, :]  # shape matches v

        # Flow terms
        hu = h_cropped_x * u
        hv = h_cropped_y * v

        # First-order divergence (∂(hu)/∂x and ∂(hv)/∂y)
        d_hu_dx = (hu[:, :, :, 1:] - hu[:, :, :, :-1]) / dx
        d_hv_dy = (hv[:, :, 1:, :] - hv[:, :, :-1, :]) / dx

        # Final cropping to match shape [B, 1, H-2, W-2]
        min_h = min(d_hu_dx.shape[2], d_hv_dy.shape[2])
        min_w = min(d_hu_dx.shape[3], d_hv_dy.shape[3])

        d_hu_dx_crop = d_hu_dx[:, :, :min_h, :min_w]
        d_hv_dy_crop = d_hv_dy[:, :, :min_h, :min_w]
        rainfall_crop = rainfall[:, :, :min_h, :min_w]

        divergence = d_hu_dx_crop + d_hv_dy_crop
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
