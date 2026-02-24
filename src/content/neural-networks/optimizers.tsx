"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function Optimizers() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          An optimizer is the algorithm that decides <strong>how to update the weights</strong> given
          the gradients. Vanilla gradient descent just takes a step proportional to the gradient, but
          real loss surfaces are messy — they have ravines, saddle points, flat regions, and noisy
          gradients from mini-batches. A good optimizer navigates all of this.
        </p>
        <p>
          <strong>SGD with momentum</strong> is like a ball rolling downhill — it accumulates velocity
          in consistent directions and dampens oscillations. This dramatically improves convergence in
          loss landscapes with elongated ravines (which is most loss landscapes).
        </p>
        <p>
          <strong>Adam</strong> (Adaptive Moment Estimation) is the most popular optimizer in deep
          learning. It combines momentum with <strong>per-parameter learning rates</strong> — parameters
          that get large gradients get smaller steps, and parameters with small gradients get larger steps.
          This makes it very forgiving of the initial learning rate choice and is why it&apos;s the default
          &quot;just works&quot; optimizer.
        </p>
        <p>
          The optimizer debate is real: Adam converges faster but sometimes generalizes worse than
          well-tuned SGD with momentum. In practice, Adam (or its variant AdamW with decoupled weight
          decay) is the standard for Transformers and most modern architectures. SGD with momentum is
          still preferred for some vision tasks (ResNets) and when you can afford extensive hyperparameter
          tuning.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Stochastic Gradient Descent (SGD)</h3>
        <BlockMath math="\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)" />
        <p>
          where <InlineMath math="\eta" /> is the learning rate and <InlineMath math="\nabla L" /> is
          computed on a random mini-batch (hence &quot;stochastic&quot;). The noise from mini-batches
          actually <em>helps</em> — it acts as regularization and helps escape local minima.
        </p>

        <h3>SGD with Momentum</h3>
        <BlockMath math="v_t = \beta v_{t-1} + \nabla L(\theta_t)" />
        <BlockMath math="\theta_{t+1} = \theta_t - \eta v_t" />
        <p>
          Typical <InlineMath math="\beta = 0.9" />. The velocity <InlineMath math="v_t" /> is an
          exponential moving average of past gradients. In a ravine, oscillating gradients cancel out
          while the consistent downhill direction accumulates.
        </p>

        <h3>Nesterov Accelerated Gradient (NAG)</h3>
        <BlockMath math="v_t = \beta v_{t-1} + \nabla L(\theta_t - \eta \beta v_{t-1})" />
        <BlockMath math="\theta_{t+1} = \theta_t - \eta v_t" />
        <p>
          The key insight: compute the gradient at the <strong>look-ahead position</strong> <InlineMath math="\theta_t - \eta \beta v_{t-1}" />,
          not the current position. This provides a &quot;correction&quot; that reduces overshooting.
        </p>

        <h3>RMSProp (Adaptive Learning Rates)</h3>
        <BlockMath math="s_t = \beta s_{t-1} + (1 - \beta)(\nabla L(\theta_t))^2" />
        <BlockMath math="\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}} \nabla L(\theta_t)" />
        <p>
          Divides the learning rate by the running average of gradient magnitudes. Parameters with
          large gradients get smaller effective learning rates. <InlineMath math="\epsilon \approx 10^{-8}" /> prevents division by zero.
        </p>

        <h3>Adam (Adaptive Moment Estimation)</h3>
        <p>Combines momentum (first moment) with RMSProp (second moment):</p>
        <BlockMath math="m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta_t)" />
        <BlockMath math="v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L(\theta_t))^2" />
        <p>Bias correction (critical for early steps when <InlineMath math="m_t, v_t" /> are biased toward zero):</p>
        <BlockMath math="\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}" />
        <p>Update:</p>
        <BlockMath math="\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t" />
        <p>
          Defaults: <InlineMath math="\beta_1 = 0.9" />, <InlineMath math="\beta_2 = 0.999" />, <InlineMath math="\epsilon = 10^{-8}" />.
        </p>

        <h3>AdamW (Decoupled Weight Decay)</h3>
        <p>
          Standard Adam applies weight decay inside the adaptive gradient, which is incorrect.
          AdamW separates weight decay from the gradient update:
        </p>
        <BlockMath math="\theta_{t+1} = \theta_t - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)" />
        <p>
          This is the optimizer used by almost all modern Transformer training (GPT, LLaMA, etc.).
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Optimizers from Scratch</h3>
        <CodeBlock
          language="python"
          title="optimizers_scratch.py"
          code={`import numpy as np

class SGDMomentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def step(self, params, grads):
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            self.velocity[key] = (
                self.momentum * self.velocity[key] + grads[key]
            )
            params[key] -= self.lr * self.velocity[key]

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # first moment
        self.v = {}  # second moment
        self.t = 0   # timestep

    def step(self, params, grads):
        self.t += 1
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            # Update biased moments
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2

            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            # Update
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# Demo: optimize Rosenbrock function f(x,y) = (1-x)^2 + 100(y-x^2)^2
def rosenbrock_grad(x, y):
    dx = -2*(1 - x) + 200*(y - x**2)*(-2*x)
    dy = 200*(y - x**2)
    return {"x": dx, "y": dy}

for OptimizerClass, name in [(SGDMomentum, "SGD+Momentum"), (Adam, "Adam")]:
    params = {"x": -1.0, "y": -1.0}
    opt = OptimizerClass(lr=0.001)

    for step in range(5000):
        grads = rosenbrock_grad(params["x"], params["y"])
        opt.step(params, grads)

    print(f"{name:15s} -> x={params['x']:.4f}, y={params['y']:.4f} "
          f"(optimal: 1.0, 1.0)")`}
        />

        <h3>PyTorch Optimizers — Practical Usage</h3>
        <CodeBlock
          language="python"
          title="pytorch_optimizers.py"
          code={`import torch
import torch.nn as nn
import torch.optim as optim

# Simple model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# ---- SGD with momentum + learning rate schedule ----
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,       # L2 regularization
    nesterov=True,           # Nesterov momentum
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100     # Cosine decay over 100 epochs
)

# ---- AdamW (the modern default) ----
optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,                 # The "Karpathy constant" — a good default
    betas=(0.9, 0.999),
    weight_decay=0.01,       # Decoupled weight decay
)

# ---- Per-parameter groups (different LR for backbone vs head) ----
optimizer = optim.AdamW([
    {"params": model[0].parameters(), "lr": 1e-5},   # backbone: low lr
    {"params": model[2].parameters(), "lr": 1e-3},   # head: high lr
], weight_decay=0.01)

# ---- Learning rate warmup + cosine decay (Transformer standard) ----
def lr_lambda(step):
    warmup_steps = 1000
    total_steps = 50000
    if step < warmup_steps:
        return step / warmup_steps  # Linear warmup
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Training loop with optimizer
criterion = nn.CrossEntropyLoss()
x = torch.randn(32, 784)  # dummy batch
y = torch.randint(0, 10, (32,))

for step in range(100):
    optimizer.zero_grad()          # 1. Clear gradients
    logits = model(x)              # 2. Forward pass
    loss = criterion(logits, y)    # 3. Compute loss
    loss.backward()                # 4. Backward pass (compute gradients)

    # Optional: gradient clipping (prevents exploding gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()               # 5. Update weights
    scheduler.step()               # 6. Update learning rate`}
        />

        <h3>Comparing Optimizers on the Same Problem</h3>
        <CodeBlock
          language="python"
          title="optimizer_comparison.py"
          code={`import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Create a synthetic problem
torch.manual_seed(42)
X = torch.randn(1000, 20)
W_true = torch.randn(20, 1)
y = X @ W_true + 0.1 * torch.randn(1000, 1)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

def train(optimizer_class, kwargs, epochs=50):
    model = nn.Linear(20, 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    opt = optimizer_class(model.parameters(), **kwargs)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

    return losses

results = {
    "SGD (lr=0.01)": train(torch.optim.SGD, {"lr": 0.01}),
    "SGD+Momentum":  train(torch.optim.SGD, {"lr": 0.01, "momentum": 0.9}),
    "Adam":          train(torch.optim.Adam, {"lr": 0.001}),
    "AdamW":         train(torch.optim.AdamW, {"lr": 0.001, "weight_decay": 0.01}),
}

for name, losses in results.items():
    print(f"{name:20s} | Final loss: {losses[-1]:.6f} | "
          f"Loss at epoch 5: {losses[4]:.6f}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>Default choice: AdamW with lr=3e-4</strong>. This works well for most problems
            out of the box. Adjust from there.
          </li>
          <li>
            <strong>Learning rate is still the most important hyperparameter</strong>. Even with adaptive
            optimizers, a 10x wrong learning rate will fail. Use a learning rate finder (sweep from 1e-7
            to 1 and plot loss vs. lr) to find the right ballpark.
          </li>
          <li>
            <strong>Learning rate warmup is critical for Transformers</strong>: start with a very small
            learning rate and linearly increase to the target over 1-5% of training steps. Without warmup,
            early large gradients can corrupt the randomly initialized parameters.
          </li>
          <li>
            <strong>Weight decay in AdamW is NOT the same as L2 regularization in Adam</strong>. In
            standard Adam, weight decay interacts with the adaptive learning rate, making it less
            effective. AdamW decouples them, which is why it&apos;s preferred.
          </li>
          <li>
            <strong>SGD generalizes better than Adam in some cases</strong>: for ResNet on ImageNet,
            SGD with momentum + cosine schedule still slightly outperforms Adam. The sharper minima
            found by Adam can sometimes generalize worse.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Using the same learning rate for Adam and SGD</strong>: Adam typically needs a much
            smaller learning rate (1e-4 to 3e-4) than SGD (0.01 to 0.1). Using SGD&apos;s learning rate
            with Adam will cause divergence.
          </li>
          <li>
            <strong>Forgetting bias correction in Adam</strong>: without the <InlineMath math="1/(1-\beta^t)" /> correction,
            the first few updates are too small because <InlineMath math="m" /> and <InlineMath math="v" /> are
            initialized to zero. All standard implementations include this, but be careful if implementing from scratch.
          </li>
          <li>
            <strong>Not decaying the learning rate</strong>: a constant learning rate will oscillate around
            the minimum forever. Use cosine annealing, step decay, or reduce-on-plateau.
          </li>
          <li>
            <strong>Applying gradient clipping after optimizer.step()</strong>: clip <em>before</em> the
            optimizer update. The correct order is: <code>loss.backward()</code> then <code>clip_grad_norm_</code> then <code>optimizer.step()</code>.
          </li>
          <li>
            <strong>Thinking Adam doesn&apos;t need tuning</strong>: while more forgiving than SGD,
            Adam still benefits from tuning <InlineMath math="\beta_1" /> (try 0.9 vs 0.95),
            <InlineMath math="\beta_2" /> (0.999 vs 0.98), and especially the learning rate.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> Explain the Adam optimizer. Why does it use both first and second
          moments? What problem does bias correction solve?
        </p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>First moment (momentum)</strong>: <InlineMath math="m_t" /> is an exponential moving
            average of the gradients. It smooths out noise and accumulates velocity in consistent
            directions, like a ball rolling downhill. This accelerates convergence in ravines.
          </li>
          <li>
            <strong>Second moment (adaptive LR)</strong>: <InlineMath math="v_t" /> is an exponential
            moving average of the squared gradients. Dividing by <InlineMath math="\sqrt{v_t}" /> gives
            each parameter its own effective learning rate — parameters with large/frequent gradients
            get smaller steps, and parameters with small/rare gradients get larger steps. This is
            especially useful for sparse features.
          </li>
          <li>
            <strong>Bias correction</strong>: since <InlineMath math="m_0 = v_0 = 0" />, the initial
            estimates are biased toward zero. For example, <InlineMath math="m_1 = (1 - \beta_1) g_1" />,
            which is much smaller than <InlineMath math="g_1" />. Dividing
            by <InlineMath math="(1 - \beta_1^t)" /> corrects this:
            at <InlineMath math="t=1" />, <InlineMath math="\hat{m}_1 = m_1 / (1 - 0.9) = m_1 / 0.1 = g_1" />.
            Without this, early training steps would be way too small.
          </li>
          <li>
            <strong>AdamW improvement</strong>: standard Adam incorrectly scales weight decay by the
            adaptive learning rate. AdamW decouples weight decay from the gradient, applying it directly
            to the parameters. This makes the regularization consistent regardless of the adaptive step size.
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Kingma &amp; Ba (2014) &quot;Adam: A Method for Stochastic Optimization&quot;</strong> — The Adam paper.</li>
          <li><strong>Loshchilov &amp; Hutter (2019) &quot;Decoupled Weight Decay Regularization&quot;</strong> — The AdamW paper explaining why L2 and weight decay differ in adaptive optimizers.</li>
          <li><strong>Ruder (2016) &quot;An overview of gradient descent optimization algorithms&quot;</strong> — The definitive survey of SGD variants.</li>
          <li><strong>Smith (2018) &quot;A disciplined approach to neural network hyper-parameters&quot;</strong> — The 1-cycle learning rate policy and super-convergence.</li>
          <li><strong>Zhang et al. (2019) &quot;Which Algorithmic Choices Matter?&quot;</strong> — Empirical study comparing Adam vs. SGD across tasks.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
