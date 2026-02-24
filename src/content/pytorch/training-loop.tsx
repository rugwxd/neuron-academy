"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function TrainingLoop() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          The training loop is the heartbeat of deep learning. It&apos;s the code that actually
          teaches a model, repeating the same cycle thousands or millions of times: <strong>forward
          pass</strong> (make a prediction), <strong>compute loss</strong> (measure the error),
          <strong>backward pass</strong> (compute gradients), <strong>update weights</strong> (improve
          the model).
        </p>
        <p>
          In PyTorch, the training loop is explicit — you write every step yourself. This is both
          PyTorch&apos;s greatest strength (total control, easy to debug, easy to customize) and its
          biggest annoyance (lots of boilerplate for standard setups). A production training loop includes
          much more than the four basic steps: data loading, learning rate scheduling, gradient clipping,
          validation, checkpointing, logging, mixed precision, distributed training, and early stopping.
        </p>
        <p>
          <strong>PyTorch Lightning</strong> (and similar libraries like Hugging Face Trainer) exist to
          handle the boilerplate while keeping the flexibility. You define <em>what</em> to compute
          (the model, the loss, the optimizer) and Lightning handles <em>how</em> to run it (multi-GPU,
          mixed precision, logging, checkpointing). The key insight: start by understanding the raw
          training loop, then use Lightning to avoid rewriting it for every project.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>The Training Objective</h3>
        <p>
          Training seeks to minimize the expected loss over the data distribution:
        </p>
        <BlockMath math="\theta^* = \arg\min_\theta \mathbb{E}_{(x, y) \sim \mathcal{D}} [L(f_\theta(x), y)]" />
        <p>
          Since we can&apos;t compute the expectation over the true distribution, we approximate with
          a mini-batch of <InlineMath math="B" /> samples:
        </p>
        <BlockMath math="\nabla_\theta L \approx \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta L(f_\theta(x_i), y_i)" />

        <h3>Generalization and Overfitting</h3>
        <p>
          We monitor two losses during training:
        </p>
        <BlockMath math="\mathcal{L}_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{i} L(f_\theta(x_i), y_i)" />
        <BlockMath math="\mathcal{L}_{\text{val}} = \frac{1}{N_{\text{val}}} \sum_{j} L(f_\theta(x_j), y_j)" />
        <p>
          When <InlineMath math="\mathcal{L}_{\text{val}}" /> stops decreasing while <InlineMath math="\mathcal{L}_{\text{train}}" /> continues to decrease, the model is overfitting.
          Early stopping saves the model at the epoch where <InlineMath math="\mathcal{L}_{\text{val}}" /> was
          lowest.
        </p>

        <h3>Learning Rate Schedule: Cosine Annealing</h3>
        <p>
          The most common modern schedule, used in nearly all Transformer training:
        </p>
        <BlockMath math="\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)" />
        <p>
          Often combined with linear warmup for the first <InlineMath math="T_w" /> steps:
        </p>
        <BlockMath math="\eta_t = \begin{cases} \eta_{\max} \cdot \frac{t}{T_w} & \text{if } t < T_w \\ \text{cosine decay} & \text{otherwise} \end{cases}" />
      </TopicSection>

      <TopicSection type="code">
        <h3>The Complete Raw Training Loop</h3>
        <CodeBlock
          language="python"
          title="training_loop_raw.py"
          code={`import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ============================================================
# 1. DATA
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

full_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, transform=transform)

# Split training into train + validation
train_dataset, val_dataset = random_split(full_dataset, [55000, 5000])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# ============================================================
# 2. MODEL
# ============================================================
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10),
)

# ============================================================
# 3. OPTIMIZER + SCHEDULER
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
criterion = nn.CrossEntropyLoss()

# ============================================================
# 4. TRAINING LOOP
# ============================================================
best_val_loss = float("inf")
patience, patience_counter = 5, 0

for epoch in range(20):
    # ---- Training ----
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()               # Clear gradients
        output = model(data)                # Forward pass
        loss = criterion(output, target)    # Compute loss
        loss.backward()                     # Backward pass

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()                    # Update weights

        train_loss += loss.item() * data.size(0)
        train_correct += (output.argmax(1) == target).sum().item()
        train_total += data.size(0)

    scheduler.step()

    # ---- Validation ----
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            val_loss += loss.item() * data.size(0)
            val_correct += (output.argmax(1) == target).sum().item()
            val_total += data.size(0)

    train_loss /= train_total
    val_loss /= val_total
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1:2d} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | "
          f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # ---- Early stopping + checkpointing ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ---- Test evaluation ----
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        test_correct += (model(data).argmax(1) == target).sum().item()
        test_total += data.size(0)

print(f"\\nTest Accuracy: {test_correct/test_total:.2%}")`}
        />

        <h3>The Same Model with PyTorch Lightning</h3>
        <CodeBlock
          language="python"
          title="training_loop_lightning.py"
          code={`import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchmetrics import Accuracy

class MNISTClassifier(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def setup(self, stage=None):
        full = datasets.MNIST("./data", train=True, download=True,
                              transform=self.transform)
        self.train_ds, self.val_ds = random_split(full, [55000, 5000])
        self.test_ds = datasets.MNIST("./data", train=False,
                                      transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                         shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=256, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=256, num_workers=2)


# ---- Training: 4 lines vs 50+ ----
model = MNISTClassifier(lr=1e-3)
data = MNISTDataModule(batch_size=128)

trainer = L.Trainer(
    max_epochs=20,
    accelerator="auto",           # Automatically uses GPU if available
    callbacks=[
        L.pytorch.callbacks.EarlyStopping(monitor="val/loss", patience=5),
        L.pytorch.callbacks.ModelCheckpoint(monitor="val/loss"),
    ],
    gradient_clip_val=1.0,        # Gradient clipping built-in
    precision="16-mixed",         # Mixed precision built-in
)

trainer.fit(model, data)
trainer.test(model, data)`}
        />

        <h3>Advanced Patterns: Gradient Accumulation and Multi-GPU</h3>
        <CodeBlock
          language="python"
          title="advanced_training.py"
          code={`import torch
import torch.nn as nn

# ---- Gradient Accumulation ----
# Simulate batch_size=256 when GPU only fits batch_size=64
model = nn.Linear(784, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
accumulation_steps = 4  # 64 * 4 = 256 effective batch size

for step, (data, target) in enumerate(train_loader):
    output = model(data.flatten(1))
    loss = nn.functional.cross_entropy(output, target)
    loss = loss / accumulation_steps      # Scale loss by accumulation steps
    loss.backward()                        # Gradients accumulate

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()                   # Update only every N steps
        optimizer.zero_grad()

# ---- Mixed Precision Training ----
scaler = torch.amp.GradScaler("cuda")

for data, target in train_loader:
    data, target = data.to("cuda"), target.to("cuda")
    optimizer.zero_grad()

    # Forward pass in float16 (where safe)
    with torch.autocast("cuda"):
        output = model(data.flatten(1))
        loss = nn.functional.cross_entropy(output, target)

    # Backward pass: scaler handles float16 -> float32 gradients
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# ---- With Lightning, all of this is just flags: ----
# trainer = L.Trainer(
#     accumulate_grad_batches=4,
#     precision="16-mixed",
#     devices=4,                     # Multi-GPU
#     strategy="ddp",                # Distributed Data Parallel
# )`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>Start with the raw loop for learning, use Lightning for projects</strong>: understanding
            every step of the training loop is essential for debugging. But once you&apos;re comfortable,
            Lightning eliminates bugs from boilerplate (device management, gradient zeroing, eval mode).
          </li>
          <li>
            <strong>Always use a validation set</strong>: never tune hyperparameters on the test set.
            Split your data into train/val/test (e.g., 80/10/10). Monitor validation loss for early
            stopping and model selection.
          </li>
          <li>
            <strong>Checkpoint the best model</strong>: training loss is not the metric you care about.
            Save the model with the best <em>validation</em> metric, then load it for final evaluation
            on the test set.
          </li>
          <li>
            <strong>Log everything</strong>: use Weights &amp; Biases (wandb) or TensorBoard to track
            loss curves, learning rates, gradient norms, and metrics. You can&apos;t debug what you
            can&apos;t see.
          </li>
          <li>
            <strong>Gradient clipping is not optional for Transformers</strong>: without it, occasional
            large gradients (from attention score explosion) can corrupt the entire model. Use
            <code>clip_grad_norm_</code> with max_norm=1.0.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Not calling model.eval() during validation</strong>: Dropout and BatchNorm behave
            differently during training and evaluation. Forgetting <code>model.eval()</code> means your
            validation metrics are wrong. Always pair with <code>model.train()</code> before the
            training loop.
          </li>
          <li>
            <strong>Not wrapping validation in torch.no_grad()</strong>: without it, PyTorch builds
            a computational graph during validation, wasting memory and potentially causing OOM errors.
          </li>
          <li>
            <strong>Using loss.item() inside backward()</strong>: <code>loss.item()</code> detaches
            the tensor from the graph. Call <code>loss.backward()</code> first, then use
            <code>loss.item()</code> for logging.
          </li>
          <li>
            <strong>Shuffling the validation/test set</strong>: only shuffle the training data.
            Shuffling validation data doesn&apos;t help and makes results harder to reproduce.
          </li>
          <li>
            <strong>Computing accuracy on the last batch only</strong>: the last batch may have a
            different size. Accumulate total correct predictions and total samples across all batches,
            then divide at the end.
          </li>
          <li>
            <strong>Not normalizing the accumulated loss</strong>: when summing loss across batches,
            multiply <code>loss.item()</code> by <code>batch_size</code> and divide by total samples
            at the end. Otherwise, the last (possibly smaller) batch has disproportionate weight.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> Walk me through a PyTorch training loop. What happens at each
          step? Where are the common failure points? How would you scale this to multiple GPUs?
        </p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>The 5 steps per batch</strong>:
            <ul>
              <li><code>optimizer.zero_grad()</code> — clear accumulated gradients from the last step.</li>
              <li><code>output = model(data)</code> — forward pass, builds the computational graph.</li>
              <li><code>loss = criterion(output, target)</code> — compute the scalar loss.</li>
              <li><code>loss.backward()</code> — reverse-mode AD computes <InlineMath math="\partial L / \partial \theta" /> for every parameter.</li>
              <li><code>optimizer.step()</code> — updates parameters using the optimizer&apos;s update rule (Adam, SGD, etc.).</li>
            </ul>
          </li>
          <li>
            <strong>Common failure points</strong>:
            <ul>
              <li>Forgetting <code>zero_grad()</code> — gradients accumulate, weights blow up.</li>
              <li>Forgetting <code>model.eval()</code> during validation — dropout/batchnorm give wrong results.</li>
              <li>Tensors on different devices — crashes at the first operation.</li>
              <li>NaN loss — usually learning rate too high or numerical instability. Check for inf/nan in inputs.</li>
            </ul>
          </li>
          <li>
            <strong>Multi-GPU scaling</strong>:
            <ul>
              <li><strong>DataParallel (DP)</strong>: simple but inefficient — one GPU does all the gradient aggregation.</li>
              <li><strong>DistributedDataParallel (DDP)</strong>: the standard approach. Each GPU runs an independent process with its own data shard. Gradients are synchronized via all-reduce. Near-linear scaling.</li>
              <li><strong>FSDP (Fully Sharded Data Parallel)</strong>: shards model parameters across GPUs for models that don&apos;t fit on a single GPU.</li>
              <li>Lightning makes this a one-line change: <code>Trainer(devices=4, strategy=&quot;ddp&quot;)</code>.</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>PyTorch official tutorials</strong> — Start with &quot;Learning PyTorch with Examples&quot; and &quot;What is torch.nn really?&quot;</li>
          <li><strong>PyTorch Lightning documentation</strong> — Structured approach to organizing training code.</li>
          <li><strong>Karpathy &quot;nanoGPT&quot;</strong> — A clean, minimal training loop for GPT-2 from scratch.</li>
          <li><strong>Li et al. (2020) &quot;PyTorch Distributed: Experiences on Accelerating Data Parallel Training&quot;</strong> — How DDP works internally.</li>
          <li><strong>Weights &amp; Biases guides</strong> — Practical tips for experiment tracking and hyperparameter sweeps.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
