# %%
from src.model.floodnet import FloodNet
from src.data.dataclass import FloodDataset
import torch
from torch.optim import Adam
import torch.nn.functional as F


# %%
def simulate_flood(
    model, dataset, steps=288, rainfall_type="constant", infiltration=0.001
):
    x, dem, _ = dataset[0]
    device = x.device
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-4)

    H, W = dem.shape[-2:]
    h = torch.zeros((1, 1, H, W), device=device)
    losses = []

    for epoch in range(3):  # small for now
        h = torch.zeros_like(h)
        optimizer.zero_grad()
        total_loss = 0

        for t in range(steps):
            if rainfall_type == "ramp":
                r = torch.full_like(h, 0.1 * (1 - abs((t - steps / 2) / (steps / 2))))
            else:
                r = torch.full_like(h, 0.1)

            h_new = model(x, h)
            h = torch.clamp(h_new + r - infiltration, min=0)

            loss = F.mse_loss(h_new[:, :, 1:, :], h_new[:, :, :-1, :]) + F.mse_loss(
                h_new[:, :, :, 1:], h_new[:, :, :, :-1]
            )
            total_loss += loss

        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())
        print(f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}")

    return model, h.detach(), losses


# %%
def visualize_final_water_height(h):
    import matplotlib.pyplot as plt

    h_np = h[0, 0].detach().cpu().numpy()
    plt.imshow(h_np, cmap="Blues")
    plt.title("Final Water Height After Simulation")
    plt.colorbar()
    plt.axis("off")
    plt.show()


# %%
# Load dataset and initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FloodNet().to(device)
dataset = FloodDataset("taree", rainfall_pattern="constant")

# Train model with time-stepped simulation
model, h_final, losses = simulate_flood(
    model=model,
    dataset=dataset,
    steps=288,  # 2 days x 24 hrs x 6 (10-min)
    rainfall_type="constant",
    infiltration=0.001,  # mm/min or equivalent
)

# Plot final water height map
visualize_final_water_height(h_final)

# Plot training loss
import matplotlib.pyplot as plt

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.show()

# %%
