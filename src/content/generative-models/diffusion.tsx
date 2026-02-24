"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function DiffusionModels() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Diffusion models generate data by learning to <strong>reverse a gradual noising process</strong>.
          The idea is beautifully simple: take a clean image, add noise step by step until it&apos;s pure
          static, then train a neural network to undo each step. At generation time, start with pure noise
          and iteratively denoise it to get a realistic image.
        </p>
        <p>
          Think of it like this: imagine slowly dissolving a sugar sculpture in water (the forward process).
          If you recorded exactly how the sugar dissolved at each step, you could play the video in reverse
          to reconstruct the sculpture from a glass of sugar water. A diffusion model learns to play this
          reverse video — but for images, audio, or any kind of data.
        </p>
        <p>
          Diffusion models have become the dominant generative paradigm, powering Stable Diffusion, DALL-E 2,
          Midjourney, and Sora. They produce <strong>higher-quality and more diverse outputs</strong> than
          GANs, with much more stable training. The tradeoff is speed — they require many denoising steps
          at inference time (though techniques like DDIM and consistency models have dramatically reduced this).
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Forward Process (Adding Noise)</h3>
        <p>
          Given a data point <InlineMath math="x_0 \sim q(x_0)" />, the forward process adds Gaussian noise
          over <InlineMath math="T" /> steps according to a variance schedule <InlineMath math="\beta_1, \ldots, \beta_T" />:
        </p>
        <BlockMath math="q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t I)" />
        <p>
          Key insight: we can jump directly to any timestep <InlineMath math="t" /> without iterating. Let
          <InlineMath math="\alpha_t = 1 - \beta_t" /> and <InlineMath math="\bar{\alpha}_t = \prod_{s=1}^t \alpha_s" />:
        </p>
        <BlockMath math="q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\, x_0, (1 - \bar{\alpha}_t) I)" />
        <BlockMath math="x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)" />

        <h3>Reverse Process (Denoising)</h3>
        <p>
          The reverse process is also Gaussian when <InlineMath math="\beta_t" /> is small:
        </p>
        <BlockMath math="p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)" />
        <p>
          The model learns to predict the mean <InlineMath math="\mu_\theta" />. In practice, we reparameterize
          the model to predict the <strong>noise</strong> <InlineMath math="\epsilon_\theta(x_t, t)" /> instead:
        </p>
        <BlockMath math="\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)" />

        <h3>Training Objective (Simplified DDPM Loss)</h3>
        <p>
          The variational bound simplifies to a remarkably clean objective — just predict the noise:
        </p>
        <BlockMath math="\mathcal{L}_{\text{simple}} = E_{t, x_0, \epsilon}\left[\| \epsilon - \epsilon_\theta(x_t, t) \|^2\right]" />
        <p>
          where <InlineMath math="t \sim \text{Uniform}(1, T)" />, <InlineMath math="\epsilon \sim \mathcal{N}(0, I)" />,
          and <InlineMath math="x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon" />.
          This is just MSE between the true noise and the predicted noise.
        </p>

        <h3>DDIM (Faster Sampling)</h3>
        <p>
          DDPM requires <InlineMath math="T" /> steps (typically 1000). DDIM makes the process <strong>deterministic</strong> and
          allows skipping steps via a subsequence <InlineMath math="\tau_1, \tau_2, \ldots, \tau_S" /> with <InlineMath math="S \ll T" />:
        </p>
        <BlockMath math="x_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}}\, \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}}}\, \frac{x_{\tau_i} - \sqrt{\bar{\alpha}_{\tau_i}}\, \hat{x}_0}{\sqrt{1 - \bar{\alpha}_{\tau_i}}}" />
        <p>where <InlineMath math="\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}" /> is the predicted clean image.</p>
      </TopicSection>

      <TopicSection type="code">
        <h3>DDPM Training Loop</h3>
        <CodeBlock
          language="python"
          title="ddpm_train.py"
          code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.T = T
        self.device = device

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, T).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self, x0, t):
        """Add noise to x0 at timestep t. Returns noisy image and noise."""
        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

        # x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    def train_step(self, model, x0, optimizer):
        """One training step: predict noise from noisy input."""
        batch_size = x0.size(0)

        # Sample random timesteps
        t = torch.randint(0, self.T, (batch_size,), device=self.device)

        # Forward diffusion
        x_t, noise = self.forward_diffusion(x0, t)

        # Predict noise
        noise_pred = model(x_t, t)

        # Simple MSE loss
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, model, shape):
        """Generate samples via reverse diffusion (DDPM)."""
        x = torch.randn(shape, device=self.device)

        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            noise_pred = model(x, t_batch)

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.betas[t]

            # Compute mean
            mean = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
            )

            # Add noise (except at t=0)
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = mean + sigma * noise
            else:
                x = mean

        return x`}
        />

        <h3>U-Net Noise Predictor (Simplified)</h3>
        <CodeBlock
          language="python"
          title="simple_unet.py"
          code={`import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    """Timestep embedding using sinusoidal positional encoding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class SimpleUNet(nn.Module):
    """Minimal U-Net for diffusion (28x28 images)."""
    def __init__(self, channels=1, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
        )
        # Encoder
        self.enc1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        # Bottleneck
        self.bot = nn.Conv2d(128, 128, 3, padding=1)
        # Decoder
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(128, channels, 3, padding=1)  # skip connection

        self.time_proj = nn.Linear(time_dim, 128)
        self.act = nn.GELU()

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Encoder
        h1 = self.act(self.enc1(x))           # (B, 64, 28, 28)
        h2 = self.act(self.enc2(h1))          # (B, 128, 14, 14)

        # Add time embedding
        h2 = h2 + self.time_proj(t_emb)[:, :, None, None]

        # Bottleneck
        h = self.act(self.bot(h2))            # (B, 128, 14, 14)

        # Decoder with skip connection
        h = self.act(self.dec2(h))            # (B, 64, 28, 28)
        h = torch.cat([h, h1], dim=1)        # (B, 128, 28, 28)
        return self.dec1(h)                   # (B, 1, 28, 28)

# Putting it together
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleUNet().to(device)
ddpm = SimpleDDPM(T=1000, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# Training (using a DataLoader for MNIST, as in VAE example)
# for epoch in range(50):
#     for batch, _ in train_loader:
#         loss = ddpm.train_step(model, batch.to(device), optimizer)

# Generate
# samples = ddpm.sample(model, shape=(16, 1, 28, 28))`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Latent diffusion (Stable Diffusion)</strong>: Running diffusion in pixel space is expensive. Encode images to a lower-dimensional latent space with a pretrained VAE, run diffusion there, then decode. This is 10-100x faster.</li>
          <li><strong>Classifier-free guidance</strong>: Train with both conditional and unconditional objectives. At inference, interpolate: <InlineMath math="\hat{\epsilon} = (1 + w)\epsilon_\theta(x_t, t, c) - w \cdot \epsilon_\theta(x_t, t, \emptyset)" />. Guidance scale <InlineMath math="w" /> controls quality vs diversity.</li>
          <li><strong>Noise schedules matter</strong>: Linear schedules work for small images. Cosine schedules (improved DDPM) work better for higher resolutions by spending more steps at low noise levels.</li>
          <li><strong>Fast sampling</strong>: DDIM reduces 1000 steps to 50. Consistency models and distillation can get it down to 1-4 steps.</li>
          <li><strong>Text conditioning</strong>: Use a frozen text encoder (CLIP, T5) to embed prompts. Inject text embeddings into the U-Net via cross-attention layers.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Confusing the model&apos;s prediction target</strong>: DDPM predicts the noise <InlineMath math="\epsilon" />. Some formulations predict <InlineMath math="x_0" /> directly or the &quot;velocity&quot; <InlineMath math="v = \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1-\bar{\alpha}_t}x_0" />. Mixing these up silently breaks generation.</li>
          <li><strong>Wrong normalization</strong>: Images must be scaled to [-1, 1] (not [0, 1]) for the Gaussian noise assumptions to hold. Using [0, 255] will completely fail.</li>
          <li><strong>Forgetting that sampling is stochastic</strong>: DDPM adds noise at each reverse step (except the last). Omitting this noise produces blurry, averaged outputs. DDIM makes this deterministic by design.</li>
          <li><strong>Not using EMA weights</strong>: Always maintain an exponential moving average of model weights for generation. Training weights are noisy; EMA weights produce much better samples.</li>
          <li><strong>Ignoring the variance schedule</strong>: The schedule controls SNR at each timestep. A bad schedule means the model spends too many steps on trivially easy or impossibly hard denoising.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Compare diffusion models to GANs and VAEs. When would you choose each?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Sample quality</strong>:
            <ul>
              <li>Diffusion models produce the <strong>highest quality</strong> and most diverse samples. They now dominate image/video generation.</li>
              <li>GANs can produce sharp images but suffer from mode collapse (less diversity).</li>
              <li>VAEs tend to produce blurry outputs due to the Gaussian decoder averaging over modes.</li>
            </ul>
          </li>
          <li><strong>Training stability</strong>:
            <ul>
              <li>Diffusion: Very stable — it&apos;s just denoising regression (MSE loss). No adversarial dynamics.</li>
              <li>GANs: Notoriously unstable. Requires careful architecture, hyperparameter tuning, and tricks.</li>
              <li>VAEs: Stable (standard gradient descent on ELBO) but can suffer from posterior collapse.</li>
            </ul>
          </li>
          <li><strong>Inference speed</strong>:
            <ul>
              <li>GANs: Single forward pass (fastest).</li>
              <li>VAEs: Single forward pass through decoder.</li>
              <li>Diffusion: Requires 20-1000 forward passes (slowest, but improving with distillation).</li>
            </ul>
          </li>
          <li><strong>When to choose each</strong>:
            <ul>
              <li><strong>Diffusion</strong>: Best quality, text-to-image, video generation, when inference budget allows.</li>
              <li><strong>GANs</strong>: Real-time applications needing fast inference (e.g., super-resolution, style transfer).</li>
              <li><strong>VAEs</strong>: When you need a structured latent space (representation learning, anomaly detection, drug discovery).</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Ho et al. (2020) &quot;Denoising Diffusion Probabilistic Models&quot;</strong> — The DDPM paper that started the diffusion revolution.</li>
          <li><strong>Song et al. (2021) &quot;Denoising Diffusion Implicit Models (DDIM)&quot;</strong> — Deterministic sampling in far fewer steps.</li>
          <li><strong>Rombach et al. (2022) &quot;High-Resolution Image Synthesis with Latent Diffusion Models&quot;</strong> — Stable Diffusion. Diffusion in latent space for practical high-res generation.</li>
          <li><strong>Lilian Weng &quot;What are Diffusion Models?&quot;</strong> — Outstanding blog post with full derivations and intuitions.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
