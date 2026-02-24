"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function SelfSupervisedLearning() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Self-supervised learning is about <strong>learning representations without labels</strong>. The model
          creates its own supervision by predicting parts of the input from other parts. This is how GPT learns
          from text (predict the next token) and how vision models learn from images (mask a region and
          reconstruct it).
        </p>
        <p>
          Think of it like a jigsaw puzzle: you don&apos;t need someone to tell you the &quot;answer&quot; — the
          puzzle itself provides the learning signal. By solving thousands of puzzles, you develop a deep
          understanding of shapes, colors, and spatial relationships — <strong>a general representation</strong> that
          transfers to other visual tasks.
        </p>
        <p>
          There are three dominant paradigms. <strong>Contrastive learning</strong> (SimCLR) teaches the model that
          two augmented views of the same image should have similar embeddings, while views of different images
          should be pushed apart. <strong>Self-distillation</strong> (BYOL) achieves the same goal without needing
          negative pairs at all — a student network learns to predict the output of a slowly-updated teacher.
          <strong>Masked modeling</strong> (MAE) masks random patches of an image and trains the model to
          reconstruct them, analogous to BERT&apos;s masked language modeling.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Contrastive Loss (InfoNCE)</h3>
        <p>
          Given a batch of <InlineMath math="N" /> images, we create two augmented views of each, giving
          <InlineMath math="2N" /> total views. For a positive pair <InlineMath math="(i, j)" /> (two views
          of the same image), the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss is:
        </p>
        <BlockMath math="\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}" />
        <p>
          where <InlineMath math="\text{sim}(z_i, z_j) = \frac{z_i^\top z_j}{\|z_i\| \|z_j\|}" /> is cosine
          similarity and <InlineMath math="\tau" /> is a temperature parameter that controls the sharpness of
          the distribution. The total SimCLR loss averages over all positive pairs:
        </p>
        <BlockMath math="\mathcal{L}_{\text{SimCLR}} = \frac{1}{2N} \sum_{k=1}^{N} [\ell_{2k-1, 2k} + \ell_{2k, 2k-1}]" />

        <h3>Why Temperature Matters</h3>
        <p>
          Small <InlineMath math="\tau" /> (e.g., 0.07) makes the softmax peakier, focusing on the hardest
          negatives. Large <InlineMath math="\tau" /> treats all negatives more equally. SimCLR uses
          <InlineMath math="\tau = 0.5" /> by default.
        </p>

        <h3>BYOL: No Negatives Needed</h3>
        <p>
          BYOL uses an online network <InlineMath math="f_\theta" /> with a predictor <InlineMath math="q_\theta" /> and
          a target network <InlineMath math="f_\xi" /> updated via exponential moving average (EMA):
        </p>
        <BlockMath math="\xi \leftarrow m \cdot \xi + (1 - m) \cdot \theta, \quad m \in [0.99, 1)" />
        <p>The loss is a simple mean squared error between normalized predictions and targets:</p>
        <BlockMath math="\mathcal{L}_{\text{BYOL}} = \left\| \overline{q_\theta(f_\theta(x_1))} - \overline{f_\xi(x_2)} \right\|_2^2 + \left\| \overline{q_\theta(f_\theta(x_2))} - \overline{f_\xi(x_1)} \right\|_2^2" />
        <p>
          where <InlineMath math="\overline{v} = v / \|v\|" /> denotes L2 normalization. The asymmetry (predictor only
          on the online side, stop-gradient on the target) prevents collapse.
        </p>

        <h3>Masked Autoencoder (MAE)</h3>
        <p>
          MAE randomly masks a large fraction (75%) of image patches and trains an encoder-decoder to reconstruct
          the masked pixels. The loss is mean squared error on the masked patches only:
        </p>
        <BlockMath math="\mathcal{L}_{\text{MAE}} = \frac{1}{|M|} \sum_{i \in M} \| x_i - \hat{x}_i \|_2^2" />
        <p>
          where <InlineMath math="M" /> is the set of masked patch indices, <InlineMath math="x_i" /> is the original
          patch, and <InlineMath math="\hat{x}_i" /> is the reconstructed patch. The encoder only processes
          visible patches (25%), making pre-training highly efficient.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>SimCLR Contrastive Learning from Scratch</h3>
        <CodeBlock
          language="python"
          title="simclr.py"
          code={`import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# --- Data Augmentation (critical for contrastive learning) ---
class SimCLRAugmentation:
    """Apply two random augmentations to create a positive pair."""
    def __init__(self, size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465],
                                 [0.2470, 0.2435, 0.2616]),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

# --- Encoder + Projection Head ---
class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        self.encoder = base_encoder  # e.g., ResNet-18 backbone
        # Replace final FC with identity to get features
        feat_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        # Projection head: maps features to contrastive space
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, projection_dim),
        )

    def forward(self, x):
        features = self.encoder(x)           # h = f(x)
        projections = self.projector(features)  # z = g(h)
        return features, projections

# --- NT-Xent Loss (InfoNCE) ---
def nt_xent_loss(z1, z2, temperature=0.5):
    """Normalized Temperature-scaled Cross Entropy Loss."""
    batch_size = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate: [z1_0, z1_1, ..., z2_0, z2_1, ...]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    # Cosine similarity matrix (2B x 2B)
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    pos_idx = torch.arange(2 * batch_size, device=z.device)
    pos_idx = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])

    # Cross-entropy: each row's positive is at pos_idx
    loss = F.cross_entropy(sim, pos_idx)
    return loss

# --- Training Loop ---
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base = resnet18(weights=None)
model = SimCLR(base, projection_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

train_data = datasets.CIFAR10(
    './data', train=True, download=True,
    transform=SimCLRAugmentation(size=32)
)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True,
                          num_workers=2, drop_last=True)

model.train()
for epoch in range(100):
    total_loss = 0
    for (x1, x2), _ in train_loader:  # labels ignored!
        x1, x2 = x1.to(device), x2.to(device)

        _, z1 = model(x1)
        _, z2 = model(x2)

        loss = nt_xent_loss(z1, z2, temperature=0.5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg = total_loss / len(train_loader)
    print(f"Epoch {epoch+1:3d} | Loss: {avg:.4f}")`}
        />

        <h3>BYOL-Style Self-Distillation with EMA Target Network</h3>
        <CodeBlock
          language="python"
          title="byol.py"
          code={`import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class BYOL(nn.Module):
    """Bootstrap Your Own Latent — no negative pairs needed."""
    def __init__(self, encoder, feature_dim=512, projection_dim=256,
                 hidden_dim=4096, ema_decay=0.996):
        super().__init__()
        self.ema_decay = ema_decay

        # Online network: encoder + projector + predictor
        self.online_encoder = encoder
        self.online_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

        # Target network: encoder + projector (no predictor!)
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Target params are NOT updated by gradient
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        """Exponential Moving Average update of target network."""
        m = self.ema_decay
        for p_online, p_target in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            p_target.data = m * p_target.data + (1 - m) * p_online.data
        for p_online, p_target in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters()
        ):
            p_target.data = m * p_target.data + (1 - m) * p_online.data

    def forward(self, x1, x2):
        # Online branch
        o1 = self.predictor(self.online_projector(self.online_encoder(x1)))
        o2 = self.predictor(self.online_projector(self.online_encoder(x2)))

        # Target branch (stop gradient via no_grad in update + detach)
        with torch.no_grad():
            t1 = self.target_projector(self.target_encoder(x1))
            t2 = self.target_projector(self.target_encoder(x2))

        # Symmetric loss: predict each view from the other
        loss = (
            self._regression_loss(o1, t2.detach()) +
            self._regression_loss(o2, t1.detach())
        )
        return loss

    @staticmethod
    def _regression_loss(x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return 2 - 2 * (x * y).sum(dim=-1).mean()

# --- Training ---
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = resnet18(weights=None)
feat_dim = backbone.fc.in_features
backbone.fc = nn.Identity()

model = BYOL(backbone, feature_dim=feat_dim, projection_dim=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-6)

# Assume train_loader yields (x1, x2), _ from SimCLRAugmentation
model.train()
for epoch in range(100):
    total_loss = 0
    for (x1, x2), _ in train_loader:
        x1, x2 = x1.to(device), x2.to(device)
        loss = model(x1, x2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_target()  # EMA update after each step
        total_loss += loss.item()
    print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f}")`}
        />

        <h3>Masked Autoencoder (MAE) for Vision</h3>
        <CodeBlock
          language="python"
          title="mae.py"
          code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)

class MAE(nn.Module):
    """Masked Autoencoder: mask patches, encode visible, decode all."""
    def __init__(self, img_size=224, patch_size=16, embed_dim=768,
                 encoder_depth=12, decoder_embed_dim=512, decoder_depth=4,
                 num_heads=12, mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        num_patches = (img_size // patch_size) ** 2

        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

        # Decoder (lightweight)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim)
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim, nhead=8,
            dim_feedforward=decoder_embed_dim * 4,
            batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        # Predict pixel values for each patch
        self.pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * 3)

    def random_masking(self, x):
        """Randomly mask patches. Return visible tokens and indices."""
        B, N, D = x.shape
        keep = int(N * (1 - self.mask_ratio))

        # Random shuffle -> keep first 'keep' tokens
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :keep]
        x_visible = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )
        # Binary mask: 1 = masked, 0 = visible
        mask = torch.ones(B, N, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_visible, mask, ids_restore

    def forward(self, imgs):
        # Embed patches
        x = self.patch_embed(imgs)  # (B, N, D)
        x = x + self.pos_embed[:, 1:, :]

        # Mask: only encode visible patches (75% masked = huge speedup)
        x, mask, ids_restore = self.random_masking(x)

        # Add CLS token and encode
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)

        # Decode: project, insert mask tokens, add position
        x = self.decoder_embed(x)
        B, _, D = x.shape
        N = self.patch_embed.num_patches
        mask_tokens = self.mask_token.repeat(B, N + 1 - x.shape[1], 1)
        x_full = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no CLS
        x_full = torch.gather(
            x_full, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )
        x_full = torch.cat([x[:, :1, :], x_full], dim=1)  # add CLS back
        x_full = x_full + self.decoder_pos_embed
        x_full = self.decoder(x_full)

        # Predict pixels (skip CLS)
        pred = self.pred(x_full[:, 1:, :])
        return pred, mask

    def loss(self, imgs, pred, mask):
        """MSE on masked patches only."""
        # Patchify: (B, C, H, W) -> (B, N, patch_size^2 * 3)
        p = self.patch_size
        target = imgs.unfold(2, p, p).unfold(3, p, p)  # (B,C,h,w,p,p)
        target = target.contiguous().view(imgs.shape[0], -1, p * p * 3)
        # Loss only on masked positions
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)              # per-patch MSE
        loss = (loss * mask).sum() / mask.sum()  # average over masked
        return loss

# --- Training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mae = MAE(img_size=224, patch_size=16, mask_ratio=0.75).to(device)
optimizer = torch.optim.AdamW(mae.parameters(), lr=1.5e-4, weight_decay=0.05)

mae.train()
for epoch in range(200):
    for imgs, _ in train_loader:  # labels not used!
        imgs = imgs.to(device)
        pred, mask = mae(imgs)
        loss = mae.loss(imgs, pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1:3d} | Reconstruction Loss: {loss.item():.4f}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Pre-train then fine-tune</strong>: The standard recipe is to pre-train on a large unlabeled dataset (e.g., ImageNet without labels), then fine-tune on a small labeled set. Even 1% of ImageNet labels can achieve strong accuracy after SimCLR pre-training.</li>
          <li><strong>Augmentation strategy matters enormously</strong>: SimCLR showed that the composition of augmentations is critical. Random crop + color jitter is the most important combination. Without color jitter, the model can cheat by matching color histograms instead of learning semantics.</li>
          <li><strong>SimCLR needs large batch sizes</strong>: NT-Xent loss quality depends on the number of negatives. SimCLR uses batch sizes of 4096-8192. With small batches, consider MoCo (maintains a negative queue) or BYOL (no negatives).</li>
          <li><strong>BYOL avoids collapse without negatives</strong>: The combination of (1) the predictor on only the online side, (2) stop-gradient on the target, and (3) EMA updates prevents all embeddings from collapsing to a constant. BatchNorm also plays a subtle stabilizing role.</li>
          <li><strong>MAE is computationally efficient</strong>: By masking 75% of patches and only encoding the visible 25%, MAE pre-training is 3-4x faster than contrastive methods. The heavy decoder is discarded after pre-training.</li>
          <li><strong>Linear probing vs. fine-tuning</strong>: Linear probing (freeze encoder, train a linear classifier) measures representation quality. Fine-tuning all layers gives better downstream accuracy but reveals less about the representation itself.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Representation collapse</strong>: All embeddings converge to the same point, giving zero contrastive loss trivially. This happens with BYOL if you remove BatchNorm from the projector, or with SimCLR if temperature is too high. Monitor embedding standard deviation during training — if it trends toward zero, you have collapse.</li>
          <li><strong>Insufficient augmentation diversity</strong>: If augmentations are too mild (e.g., only small crops), the model learns trivial shortcuts like matching low-level texture rather than semantic content. Always include both geometric (crop, flip) and photometric (color jitter, grayscale) transforms.</li>
          <li><strong>Not scaling batch size for contrastive methods</strong>: Using SimCLR with batch size 64 gives dramatically worse results than batch size 4096. If you can&apos;t use large batches, switch to MoCo (memory bank of negatives) or BYOL (no negatives required).</li>
          <li><strong>Fine-tuning all layers when few labels</strong>: With only a few hundred labeled examples, fine-tuning the entire encoder can overfit rapidly. Start with linear probing, then gradually unfreeze layers from the top. Use a much lower learning rate for pre-trained layers than for the new classification head.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You have 1M unlabeled images and 1K labeled ones — design a training pipeline.</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Phase 1: Self-supervised pre-training on 1M unlabeled images</strong>
            <ul>
              <li>Choose method based on compute budget: MAE if you have limited GPUs (efficient due to 75% masking), SimCLR if you can use large batch sizes (4096+), BYOL if batch size is constrained.</li>
              <li>Use a ResNet-50 or ViT-B backbone. Train for 200-800 epochs on the unlabeled set with strong augmentations.</li>
              <li>Monitor for collapse: track embedding variance and loss curves.</li>
            </ul>
          </li>
          <li><strong>Phase 2: Fine-tuning on 1K labeled images</strong>
            <ul>
              <li>Freeze the pre-trained encoder. Train only a linear classification head first (linear probing) to verify representation quality.</li>
              <li>Then fine-tune with a low learning rate (<InlineMath math="10^{-4}" /> to <InlineMath math="10^{-5}" />) and strong regularization (weight decay, dropout, mixup).</li>
              <li>Use layerwise learning rate decay: lower layers get smaller learning rates to preserve learned features.</li>
            </ul>
          </li>
          <li><strong>Phase 3: Semi-supervised refinement (optional)</strong>
            <ul>
              <li>Use the fine-tuned model to pseudo-label high-confidence unlabeled examples.</li>
              <li>Retrain on labeled + pseudo-labeled data (self-training / FixMatch approach).</li>
            </ul>
          </li>
          <li><strong>Key considerations</strong>: Data augmentation pipeline should match the pre-training augmentations. Evaluate with stratified k-fold given the small labeled set. Compare against a fully supervised baseline to quantify the benefit of self-supervised pre-training.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Chen et al. (2020) &quot;A Simple Framework for Contrastive Learning of Visual Representations&quot;</strong> — The SimCLR paper. Demonstrates that strong augmentations + large batch + simple architecture achieves state-of-the-art self-supervised results.</li>
          <li><strong>Grill et al. (2020) &quot;Bootstrap Your Own Latent&quot;</strong> — BYOL. Shows contrastive learning works without negative pairs, using an EMA target network and asymmetric architecture.</li>
          <li><strong>He et al. (2022) &quot;Masked Autoencoders Are Scalable Vision Learners&quot;</strong> — MAE. Masks 75% of patches for efficient pre-training. Inspired by BERT&apos;s masked language modeling.</li>
          <li><strong>Jing &amp; Tian (2021) &quot;Self-Supervised Visual Feature Learning with Deep Neural Networks: A Survey&quot;</strong> — Comprehensive survey covering pretext tasks, contrastive methods, and evaluation protocols.</li>
          <li><strong>Ericsson et al. (2022) &quot;Self-Supervised Representation Learning: Introduction, Advances, and Challenges&quot;</strong> — Broader survey including NLP and multi-modal self-supervision.</li>
        </ul>
      </TopicSection>
    </div>
  );
}