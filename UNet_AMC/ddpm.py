"""
Proper DDPM (Denoising Diffusion Probabilistic Model) for RF signals.

Key idea: Learn to denoise by training on REAL RF noise pairs:
- low_SNR sample (noisy) → high_SNR sample (clean) of the SAME modulation class

The diffusion model learns the distribution of clean signals and how to
iteratively refine noisy signals toward that distribution.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in "Improved DDPM" paper.
    Returns betas for each timestep.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """Linear noise schedule"""
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionSchedule:
    """Precomputes all diffusion schedule quantities"""
    
    def __init__(self, timesteps=1000, schedule='cosine', device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
        if schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)
        
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: sample x_t from q(x_t | x_0)
        
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise, noise
    
    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
            self.posterior_mean_coef1[t].view(-1, 1, 1) * x_0 +
            self.posterior_mean_coef2[t].view(-1, 1, 1) * x_t
        )
        posterior_variance = self.posterior_variance[t].view(-1, 1, 1)
        posterior_log_variance = self.posterior_log_variance_clipped[t].view(-1, 1, 1)
        return posterior_mean, posterior_variance, posterior_log_variance


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for timestep"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """Conv block with time embedding"""
    
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()
        
        if in_ch != out_ch:
            self.residual = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.act(self.bn1(self.conv1(x)))
        # Add time embedding
        t = self.time_mlp(t_emb)[:, :, None]
        h = h + t
        h = self.act(self.bn2(self.conv2(h)))
        return h + self.residual(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, time_dim)
        self.pool = nn.MaxPool1d(2)
    
    def forward(self, x, t_emb):
        skip = x  # Save BEFORE conv (has in_ch channels)
        x = self.conv(x, t_emb)
        return self.pool(x), skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, time_dim)
    
    def forward(self, x, skip, t_emb):
        x = self.up(x)
        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            x = F.pad(x, (diff // 2, diff - diff // 2))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, t_emb)


class UNetDDPM(nn.Module):
    """
    U-Net for DDPM that predicts noise (epsilon) given noisy signal and timestep.
    
    Input: x_t (noisy signal), t (timestep)
    Output: predicted noise epsilon
    """
    
    def __init__(self, in_channels=2, base_channels=64, time_dim=256, depth=4):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Encoder
        # inc: in_channels -> base_channels
        self.inc = ConvBlock(in_channels, base_channels, time_dim)
        
        # Track channel sizes for skip connections
        # After inc: base_channels
        # Each down: in_ch -> out_ch (doubles)
        self.downs = nn.ModuleList()
        self.encoder_channels = [base_channels]  # Channels at each level (for skips)
        
        ch = base_channels
        for i in range(depth):
            self.downs.append(DownBlock(ch, ch * 2, time_dim))
            self.encoder_channels.append(ch)  # Skip has 'ch' channels (before conv)
            ch *= 2
        
        # Bottleneck
        self.mid = ConvBlock(ch, ch, time_dim)
        
        # Decoder
        # Need to match skip channels correctly
        # Skips are stored in order: [base, base, base*2, base*4, ...]
        # Reversed: [..., base*4, base*2, base, base]
        self.ups = nn.ModuleList()
        skip_channels = list(reversed(self.encoder_channels[1:]))  # Exclude first (used at end)
        
        for i in range(depth):
            skip_ch = skip_channels[i]
            out_ch = ch // 2
            self.ups.append(UpBlock(ch, skip_ch, out_ch, time_dim))
            ch = out_ch
        
        # Final: concatenate with inc output (base_channels)
        self.final_conv = ConvBlock(ch + base_channels, base_channels, time_dim)
        
        # Output
        self.outc = nn.Conv1d(base_channels, in_channels, 1)
    
    def forward(self, x, t):
        """
        Args:
            x: (B, 2, 128) noisy signal
            t: (B,) timesteps (integers)
        Returns:
            predicted noise: (B, 2, 128)
        """
        t_emb = self.time_mlp(t.float())
        
        # Encoder
        x = self.inc(x, t_emb)
        inc_skip = x  # Save for final concat
        
        skips = []
        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)
        
        # Bottleneck
        x = self.mid(x, t_emb)
        
        # Decoder
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, t_emb)
        
        # Final concat with inc output
        if x.size(-1) != inc_skip.size(-1):
            diff = inc_skip.size(-1) - x.size(-1)
            x = F.pad(x, (diff // 2, diff - diff // 2))
        x = torch.cat([x, inc_skip], dim=1)
        x = self.final_conv(x, t_emb)
        
        return self.outc(x)


class DDPM(nn.Module):
    """
    Full DDPM system: combines U-Net model with diffusion schedule.
    
    Provides training loss computation and sampling (denoising) methods.
    """
    
    def __init__(
        self,
        in_channels=2,
        base_channels=64,
        time_dim=256,
        depth=4,
        timesteps=1000,
        schedule='cosine',
        device='cuda'
    ):
        super().__init__()
        self.device = device
        self.timesteps = timesteps
        
        self.model = UNetDDPM(in_channels, base_channels, time_dim, depth)
        self.schedule = DiffusionSchedule(timesteps, schedule, device)
    
    def training_loss(self, x_0, noise=None):
        """
        Compute training loss.
        
        Args:
            x_0: (B, 2, 128) clean signals (high-SNR)
            noise: optional pre-sampled noise
        
        Returns:
            loss: MSE between predicted and actual noise
        """
        batch_size = x_0.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_0.device)
        
        # Sample noise
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Create noisy samples
        x_t, _ = self.schedule.q_sample(x_0, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, x_t, t, deterministic: bool = False):
        """
        Sample x_{t-1} from p(x_{t-1} | x_t) using the model.
        """
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # Get alpha values
        alpha = self.schedule.alphas[t].view(-1, 1, 1)
        alpha_cumprod = self.schedule.alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # Predict x_0
        x_0_pred = (x_t - sqrt_one_minus_alpha_cumprod * predicted_noise) / torch.sqrt(alpha_cumprod)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)  # Clip for stability
        
        # Get posterior mean
        posterior_mean, posterior_variance, _ = self.schedule.q_posterior_mean_variance(
            x_0_pred, x_t, t
        )
        
        # Add noise (except for t=0). For denoising (restoration), a deterministic
        # variant (noise=0) is often more stable and preserves content better.
        if deterministic:
            noise = torch.zeros_like(x_t)
        else:
            noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, 1, 1)
        
        return posterior_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise
    
    @torch.no_grad()
    def sample(self, x_t_init, num_steps=None, start_t=None, deterministic: bool = False):
        """
        Reverse diffusion starting from a provided x_t (not necessarily pure noise).
        
        Args:
            x_t_init: (B, 2, 128) signal assumed to correspond to timestep start_t
            num_steps: number of reverse steps to run (default: start_t+1)
            start_t: starting timestep (required for correct denoising of observations;
                     default: timesteps-1, which is appropriate ONLY when starting from
                     pure Gaussian noise)
            deterministic: if True, use posterior mean (no sampling noise)
        
        Returns:
            denoised signal: (B, 2, 128)
        """
        if start_t is None:
            start_t = self.timesteps - 1
        start_t = int(start_t)
        if start_t < 0:
            return x_t_init
        if num_steps is None:
            num_steps = start_t + 1
        
        batch_size = x_t_init.shape[0]
        x = x_t_init
        
        # Iterate from start_t to 0
        step_size = max(1, (start_t + 1) // max(1, int(num_steps)))
        timesteps = list(range(start_t, -1, -step_size))
        if len(timesteps) == 0 or timesteps[-1] != 0:
            timesteps.append(0)
        
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=x.device, dtype=torch.long)
            x = self.p_sample(x, t_batch, deterministic=deterministic)
        
        return x
    
    @torch.no_grad()
    def denoise(self, x_noisy, num_steps=50):
        """
        Fast denoising with reduced steps.
        
        Args:
            x_noisy: noisy signal
            num_steps: number of denoising steps (fewer = faster, more = better)
        
        Returns:
            denoised signal
        """
        # NOTE: This treats x_noisy as x_{T} (very noisy). This is NOT the right
        # way to denoise a real observation. Use `denoise_observation(...)` instead.
        return self.sample(x_noisy, num_steps=num_steps)

    @torch.no_grad()
    def snr_db_to_t(self, snr_db: torch.Tensor) -> torch.Tensor:
        """
        Map an SNR (dB) to an approximate diffusion timestep t by matching
        SNR_linear ≈ alpha_bar / (1 - alpha_bar).

        Assumes signals are normalized and that the dominant corruption is
        approximately additive Gaussian noise.
        """
        snr_db = snr_db.to(self.schedule.alphas_cumprod.device).float()
        snr_lin = torch.pow(10.0, snr_db / 10.0)
        alpha_bar_target = snr_lin / (1.0 + snr_lin)  # in (0,1)

        # Find nearest alpha_bar in the precomputed schedule
        alphas_cumprod = self.schedule.alphas_cumprod  # (T,)
        diffs = (alphas_cumprod[None, :] - alpha_bar_target[:, None]).abs()
        t = diffs.argmin(dim=1)
        return t

    @torch.no_grad()
    def denoise_observation(
        self,
        x_obs: torch.Tensor,
        snr_db: torch.Tensor,
        num_steps: int = 50,
        deterministic: bool = True,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Denoise a real observation x_obs by treating it as x_t for a timestep
        chosen from SNR, then running reverse diffusion from that t to 0.

        This is closer to how diffusion denoisers are typically used for
        restoration. It avoids the (wrong) assumption that x_obs is x_T.

        Args:
            x_obs: (B, 2, L)
            snr_db: (B,) SNR in dB (can be int tensor)
            num_steps: reverse steps budget (accelerated sampling)
            deterministic: True => posterior mean (stable, content-preserving)
            strength: multiply the inferred t by this factor (<=1.0 reduces denoising)
        """
        if x_obs.ndim != 3:
            raise ValueError(f"x_obs must be (B, C, L), got {tuple(x_obs.shape)}")

        t_start = self.snr_db_to_t(snr_db)
        if strength != 1.0:
            t_start = torch.clamp((t_start.float() * float(strength)).round().long(), 0, self.timesteps - 1)

        # Group by unique t to avoid per-sample loops (SNR levels are discrete)
        x_out = x_obs.clone()
        for t in torch.unique(t_start):
            t_int = int(t.item())
            mask = (t_start == t)
            if mask.sum() == 0:
                continue
            # Limit steps for this group
            group_steps = min(int(num_steps), t_int + 1)
            x_out[mask] = self.sample(
                x_obs[mask],
                num_steps=group_steps,
                start_t=t_int,
                deterministic=deterministic,
            )

        return x_out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DDPM(
        in_channels=2,
        base_channels=64,
        time_dim=256,
        depth=4,
        timesteps=1000,
        device=device
    ).to(device)
    
    print(f"DDPM parameters: {count_parameters(model):,}")
    
    # Test training loss
    x_0 = torch.randn(4, 2, 128).to(device)
    loss = model.training_loss(x_0)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test sampling
    x_noisy = torch.randn(4, 2, 128).to(device)
    x_denoised = model.denoise(x_noisy, num_steps=10)
    print(f"Denoised shape: {x_denoised.shape}")
