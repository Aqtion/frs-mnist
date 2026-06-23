# Sample from Gaussian Noise
# x0 ~ N(0,1)
# x1 ~ D
# Linearly interpolate between x0 and x1
# Get U-Net predict the velocity vector field

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image


# Timestep Embedding
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, device=device) * (torch.log(torch.tensor(10000.0)) / (half - 1)))
        comb = t[:, None] * freqs[None]
        return torch.cat([comb.sin(), comb.cos()], dim=-1)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, td):
        super().__init__()
        self.ln1 = nn.GroupNorm(min(8, inc), inc)
        self.conv1 = nn.Conv2d(inc, outc, 3, padding=1)
        self.ln2 = nn.GroupNorm(min(8, outc), outc)
        self.conv2 = nn.Conv2d(outc, outc, 3, padding=1)
        self.tp = nn.Linear(td, outc * 2)
        self.skip = nn.Conv2d(inc, outc, 1) if inc != outc else nn.Identity()
    
    def forward(self, x, te):
        h = F.silu(self.ln1(x))
        h = self.conv1(h)
        sc, sh = self.tp(F.silu(te)).chunk(2, dim=-1)
        h = self.ln2(h) * (1 + sc[..., None, None]) + sh[..., None, None]
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


# U-Net
class UNet(nn.Module):
    def __init__(self, inc=3, bc=64, td=256):
        super().__init__()
        self.te = nn.Sequential(
            SinusoidalEmbedding(td),
            nn.Linear(td, td*4),
            nn.SiLU(),
            nn.Linear(td*4, td),
        )
         
        self.enc1 = ResidualBlock(inc, bc, td)
        self.enc2 = ResidualBlock(bc, bc*2, td)
        self.down = nn.AvgPool2d(2)

        self.mid1 = ResidualBlock(bc * 2, bc * 4, td)
        self.mid2 = ResidualBlock(bc * 4, bc * 2, td)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = ResidualBlock(bc * 4, bc * 2, td)  
        self.dec1 = ResidualBlock(bc * 2 + bc, bc, td)

        self.out = nn.Conv2d(bc, inc, 1)
    
    def forward(self, x, t):
        te = self.te(t)
        enc1 = self.enc1(x, te)
        enc2 = self.enc2(self.down(enc1), te)

        b = self.mid1(self.down(enc2), te)
        b = self.mid2(b, te)

        dec2 = self.dec2(torch.cat([self.up(b),  enc2], dim=1), te)
        dec1 = self.dec1(torch.cat([self.up(dec2),  enc1], dim=1), te)

        return self.out(dec1)

# Loss
def fml(model, x1):
    B = x1.shape[0]
    device = x1.device

    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device)

    ti = t[:, None, None, None]
    xt = (1 - ti) * x0 + ti * x1

    target = x1 - x0
    predicted = model(xt, t)

    return F.mse_loss(predicted, target)

# Inference
@torch.no_grad()
def sample(model, shape, steps=50, device="mps"):
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full((shape[0],), i * dt, device=device)
        v = model(x, t)
        x = x + v * dt
    
    return x.clamp(-1, 1)

# Reverse Flow
@torch.no_grad()
def reverse_flow(model, xr, kr=10):
    device=xr.device
    b = xr.shape[0]
    x = xr.clone()
    dt = 1.0 / kr

    for i in reversed(range(kr)):
        tv = (i+1)*dt
        # timestep for batch
        t = torch.full((b,), tv, device=device)
        pred_v = model(x,t)
        # travel in opposite direction of predicted velocity from flow policy
        x = x - dt * pred_v
    
    return x

@torch.no_grad()
def forward_flow(model, x0, kf=50):
    device = x0.device
    B = x0.shape[0]
    x = x0.clone()
    dt = 1.0 / kf

    for i in range(kf):
        tval = i * dt
        t = torch.full((B,), tval, device=device)
        v = model(x, t)
        x = x + dt * v

    return x.clamp(-1, 1)

@torch.no_grad()
def frs(model, xr, rs=10, fs=50):
    # coarse reference -> predicted noise -> refined in-distribution point
    x0r = reverse_flow(model, xr, kr=rs)
    xfrs = forward_flow(model, x0r, kf=fs)
    return xfrs


@torch.no_grad()
def generate_coarse_reference(x, noise_std=0.0, downsample=7):
    crude = F.interpolate(x, size=(downsample, downsample), mode="area")
    crude = F.interpolate(crude, size=(28, 28), mode="nearest")

    # Add noise
    crude = crude + noise_std * torch.randn_like(crude)

    return crude.clamp(-1, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "sample", "frs"], default="train")
    parser.add_argument("--checkpoint", type=str, default="flow_mnist.pt")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"using device: {device}")

    if args.mode == "train":
        model  = UNet(inc=1, bc=64).to(device)
        optimizer = Adam(model.parameters(), lr=1e-4)

        # Data
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # maps to [-1, 1]
        ])
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=tf)
        loader  = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

        model = UNet(inc=1, bc=64).to(device)
        optimizer = Adam(model.parameters(), lr=1e-4)

        print("data loaded")
        epochs = 25
        for epoch in range(epochs):
            total_loss = 0
            for x1, _ in loader:
                x1 = x1.to(device)
                optimizer.zero_grad()
                loss = fml(model, x1)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg = total_loss / len(loader)
            print(f"epoch {epoch+1}/{epochs}  loss {avg:.4f}")
        
        torch.save(model.state_dict(), "flow_mnist.pt")

        imgs = sample(model, (16, 1, 28, 28), device=device) 
        print("generated:", imgs.shape)

    elif args.mode == "sample":
        model = UNet(inc=1, bc=64).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        imgs = sample(model, (16, 1, 28, 28), device=device)
        imgs = (imgs + 1) / 2
        save_image(imgs, "samples.png", nrow=4)
        print("saved samples.png")
    
    elif args.mode == "frs":
        model = UNet(inc=1, bc=64).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()

        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        dataset = datasets.MNIST(root="./data", train=False, download=True, transform=tf)
        loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

        x_real, labels = next(iter(loader))
        x_real = x_real.to(device)

        # Crude "VLM-like" reference
        x_ref = generate_coarse_reference(x_real, noise_std=0.5, downsample=7)

        # FRS refinement
        x_frs = frs(
            model,
            x_ref,
            rs=10,
            fs=50,
        )

        # Also compare normal random samples
        x_rand = sample(model, (16, 1, 28, 28), steps=50, device=device)

        # Save comparison grid:
        # row 1: real
        # row 2: crude references
        # row 3: FRS outputs
        # row 4: normal random samples
        grid = torch.cat([x_real, x_ref, x_frs, x_rand], dim=0)
        grid = (grid + 1) / 2
        save_image(grid, "frsc_.png", nrow=16)

        print("saved frs_comparison.png")
        print("rows: real | crude reference | FRS refined | random samples")
