"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function GANs() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          A Generative Adversarial Network (GAN) is a framework where two neural networks <strong>compete
          against each other</strong> to produce realistic data. The <strong>generator</strong> creates fake
          samples from random noise, and the <strong>discriminator</strong> tries to tell real samples from
          fake ones. Through this adversarial game, the generator gets better and better at producing
          realistic outputs until the discriminator can&apos;t tell the difference.
        </p>
        <p>
          Think of it as a counterfeiter (generator) versus a detective (discriminator). The counterfeiter
          learns to make increasingly convincing fake bills. The detective learns to spot increasingly subtle
          flaws. Over time, both get better — and the counterfeiter eventually produces near-perfect fakes.
          At equilibrium, the discriminator outputs 0.5 for everything (pure guessing).
        </p>
        <p>
          GANs were revolutionary because they produce <strong>sharp, high-quality outputs</strong> without
          explicit density estimation. Unlike VAEs, which optimize a lower bound and tend to produce blurry
          results, GANs directly optimize for realism. The downside is that they&apos;re notoriously hard
          to train — mode collapse, training instability, and vanishing gradients are constant challenges.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>The Minimax Objective</h3>
        <p>
          The generator <InlineMath math="G" /> and discriminator <InlineMath math="D" /> play a two-player minimax game:
        </p>
        <BlockMath math="\min_G \max_D \; V(D, G) = E_{x \sim p_{\text{data}}}[\log D(x)] + E_{z \sim p_z}[\log(1 - D(G(z)))]" />
        <p>The discriminator wants to <strong>maximize</strong> this: assign high <InlineMath math="D(x)" /> to real data and low <InlineMath math="D(G(z))" /> to fakes. The generator wants to <strong>minimize</strong> it: make <InlineMath math="D(G(z))" /> as high as possible.</p>

        <h3>Optimal Discriminator</h3>
        <p>For a fixed generator, the optimal discriminator is:</p>
        <BlockMath math="D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}" />
        <p>When the generator perfectly matches the data distribution (<InlineMath math="p_g = p_{\text{data}}" />), we get <InlineMath math="D^*(x) = \frac{1}{2}" /> everywhere.</p>

        <h3>Global Optimum and Jensen-Shannon Divergence</h3>
        <p>Substituting <InlineMath math="D^*" /> back into the objective:</p>
        <BlockMath math="V(D^*, G) = -\log 4 + 2 \cdot D_{JS}(p_{\text{data}} \| p_g)" />
        <p>
          where <InlineMath math="D_{JS}" /> is the Jensen-Shannon divergence. The global minimum is achieved
          when <InlineMath math="p_g = p_{\text{data}}" />, giving <InlineMath math="V = -\log 4" />.
        </p>

        <h3>Non-Saturating Generator Loss</h3>
        <p>
          In practice, <InlineMath math="\log(1 - D(G(z)))" /> saturates when <InlineMath math="D" /> is confident
          (gradient vanishes). Instead, maximize <InlineMath math="\log D(G(z))" />:
        </p>
        <BlockMath math="\mathcal{L}_G = -E_{z \sim p_z}[\log D(G(z))]" />
        <p>This provides stronger gradients early in training when the generator is poor.</p>

        <h3>Wasserstein GAN (WGAN) Objective</h3>
        <p>Replace JS divergence with the Earth Mover (Wasserstein-1) distance:</p>
        <BlockMath math="W(p_{\text{data}}, p_g) = \sup_{\|f\|_L \leq 1} E_{x \sim p_{\text{data}}}[f(x)] - E_{x \sim p_g}[f(x)]" />
        <p>
          The critic <InlineMath math="f" /> (no longer a classifier) must be 1-Lipschitz. This is enforced
          via weight clipping (WGAN) or gradient penalty (WGAN-GP):
        </p>
        <BlockMath math="\lambda \, E_{\hat{x}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]" />
      </TopicSection>

      <TopicSection type="code">
        <h3>DCGAN on MNIST in PyTorch</h3>
        <CodeBlock
          language="python"
          title="dcgan.py"
          code={`import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Hyperparameters
latent_dim = 100
lr = 2e-4
batch_size = 128
epochs = 20

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
D = Discriminator().to(device)

opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
])
loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

for epoch in range(epochs):
    for real_imgs, _ in loader:
        real_imgs = real_imgs.to(device)
        bs = real_imgs.size(0)
        real_labels = torch.ones(bs, 1, device=device)
        fake_labels = torch.zeros(bs, 1, device=device)

        # --- Train Discriminator ---
        z = torch.randn(bs, latent_dim, device=device)
        fake_imgs = G(z).detach()

        loss_D = (criterion(D(real_imgs), real_labels)
                + criterion(D(fake_imgs), fake_labels)) / 2

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # --- Train Generator ---
        z = torch.randn(bs, latent_dim, device=device)
        fake_imgs = G(z)
        loss_G = criterion(D(fake_imgs), real_labels)  # Non-saturating loss

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch {epoch+1:2d} | D Loss: {loss_D:.4f} | G Loss: {loss_G:.4f}")

# Generate samples
G.eval()
with torch.no_grad():
    z = torch.randn(16, latent_dim, device=device)
    samples = G(z).cpu()

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i, 0], cmap='gray')
    ax.axis('off')
plt.suptitle("GAN-Generated Digits")
plt.show()`}
        />

        <h3>WGAN-GP Training Loop</h3>
        <CodeBlock
          language="python"
          title="wgan_gp.py"
          code={`def gradient_penalty(D, real, fake, device):
    """Compute gradient penalty for WGAN-GP."""
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_out = D(interpolated)
    grads = torch.autograd.grad(
        outputs=d_out, inputs=interpolated,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True, retain_graph=True
    )[0]
    grads = grads.view(grads.size(0), -1)
    penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

# In training loop (critic update, n_critic=5 steps per G step):
# loss_D = D(fake).mean() - D(real).mean() + 10 * gradient_penalty(D, real, fake, device)
# loss_G = -D(G(z)).mean()`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Use WGAN-GP or spectral normalization</strong>: Vanilla GAN training is brittle. WGAN-GP and SN-GAN provide much more stable training with meaningful loss curves you can actually monitor.</li>
          <li><strong>Two timescale update rule</strong>: Use a higher learning rate for the discriminator (e.g., <InlineMath math="2 \times 10^{-4}" /> for D, <InlineMath math="1 \times 10^{-4}" /> for G) or more update steps for D per G step.</li>
          <li><strong>FID score for evaluation</strong>: Don&apos;t trust the loss curves — they don&apos;t correlate well with sample quality. Use Frechet Inception Distance (FID) to measure how close generated and real distributions are. Lower is better.</li>
          <li><strong>Progressive growing</strong>: Start generating at low resolution and progressively add layers (ProGAN/StyleGAN). This stabilizes training and enables high-resolution synthesis.</li>
          <li><strong>Conditional GANs</strong>: Feed class labels to both G and D to control what you generate. Pix2pix and CycleGAN extend this to image-to-image translation.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Mode collapse</strong>: The generator produces only a few types of outputs regardless of input noise. The discriminator can&apos;t fix this since each individual sample looks real. Fix: use WGAN-GP, minibatch discrimination, or unrolled GANs.</li>
          <li><strong>Training imbalance</strong>: If the discriminator gets too strong too fast, gradients to the generator vanish. If too weak, the generator gets no useful signal. Monitor both losses and use learning rate scheduling.</li>
          <li><strong>Using vanilla loss with a strong discriminator</strong>: The original <InlineMath math="\log(1 - D(G(z)))" /> saturates. Always use the non-saturating variant <InlineMath math="-\log D(G(z))" /> or the WGAN objective.</li>
          <li><strong>BatchNorm in the discriminator</strong>: With WGAN-GP, batch normalization violates the per-sample gradient penalty assumption. Use layer normalization or instance normalization instead.</li>
          <li><strong>Evaluating by eye alone</strong>: Cherry-picked samples can look great while the model has severe mode collapse. Always compute FID on a large sample.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Explain mode collapse in GANs. Why does it happen, and describe three methods to mitigate it.</p>
        <p><strong>Answer:</strong></p>
        <p>
          <strong>Mode collapse</strong> occurs when the generator learns to produce only a small subset
          of the data distribution. For example, a GAN trained on MNIST might only generate 3s and 7s,
          ignoring the other digits.
        </p>
        <p><strong>Why it happens</strong>:</p>
        <ul>
          <li>The generator finds a few outputs that consistently fool the discriminator and &quot;collapses&quot; onto them.</li>
          <li>Mathematically, the minimax game has no guarantee of converging to the Nash equilibrium with gradient descent — the generator can cycle between modes.</li>
          <li>The JS divergence used in vanilla GANs provides zero gradient when distributions don&apos;t overlap, allowing the generator to ignore entire modes.</li>
        </ul>
        <p><strong>Three mitigations</strong>:</p>
        <ol>
          <li><strong>Wasserstein loss (WGAN)</strong>: The Earth Mover distance provides gradients even when distributions don&apos;t overlap, encouraging the generator to cover all modes.</li>
          <li><strong>Minibatch discrimination</strong>: Let the discriminator see statistics across a batch (not just individual samples). If all generated samples are similar, the discriminator can detect this.</li>
          <li><strong>Unrolled GANs</strong>: When updating G, unroll k steps of D&apos;s optimization. This gives G a more accurate signal about how D will respond, preventing cycling.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Goodfellow et al. (2014) &quot;Generative Adversarial Nets&quot;</strong> — The original GAN paper with the minimax formulation and theoretical analysis.</li>
          <li><strong>Arjovsky et al. (2017) &quot;Wasserstein GAN&quot;</strong> — Explains why vanilla GANs fail and introduces the Wasserstein distance objective.</li>
          <li><strong>Karras et al. (2019) &quot;A Style-Based Generator Architecture&quot;</strong> — StyleGAN, which produces photorealistic faces at 1024x1024.</li>
          <li><strong>Lucic et al. (2018) &quot;Are GANs Created Equal?&quot;</strong> — Systematic comparison of GAN variants. Spoiler: with enough tuning, many variants perform similarly.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
