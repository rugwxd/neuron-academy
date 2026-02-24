"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function OptimizationTheory() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Optimization is the engine that makes machine learning work. You have a loss function that
          measures how wrong your model is, and optimization finds the parameters that make it as small
          as possible. The core idea is beautifully simple: <strong>compute the gradient, take a step
          downhill, repeat</strong>.
        </p>
        <p>
          <strong>Gradient descent</strong> is the vanilla version. <strong>SGD</strong> (stochastic
          gradient descent) uses random mini-batches instead of the full dataset &mdash; it&apos;s noisier
          but much faster and actually helps escape bad local minima. <strong>Momentum</strong> adds a
          &quot;velocity&quot; so the optimizer doesn&apos;t zig-zag in narrow valleys.{" "}
          <strong>Adam</strong> goes further by adapting the learning rate for each parameter individually
          based on the history of gradients.
        </p>
        <p>
          <strong>Convexity</strong> is the dream scenario: if your loss function is convex, every local
          minimum is a global minimum, and gradient descent is guaranteed to find it. Linear regression
          and logistic regression are convex. Neural networks are not &mdash; their loss landscapes are
          full of saddle points and local minima, which is why we need all these clever optimizer tricks.
        </p>
        <p>
          <strong>Learning rate schedules</strong> let you start bold and finish careful. Cosine annealing,
          warmup, and step decay are the most common. Getting the learning rate right is often more
          important than choosing the right optimizer.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Convexity</h3>
        <p>
          A function <InlineMath math="f" /> is <strong>convex</strong> if for all{" "}
          <InlineMath math="x, y" /> and <InlineMath math="\lambda \in [0, 1]" />:
        </p>
        <BlockMath math="f(\lambda x + (1-\lambda) y) \le \lambda f(x) + (1-\lambda) f(y)" />
        <p>
          Equivalently, the Hessian <InlineMath math="H = \nabla^2 f" /> is positive semi-definite
          everywhere. For convex functions, any local minimum is a global minimum.{" "}
          <strong>Strictly convex</strong> functions have a <em>unique</em> global minimum. Key properties:
        </p>
        <ul>
          <li>Sum of convex functions is convex</li>
          <li><InlineMath math="f(x) = \|Ax - b\|^2" /> (MSE) is always convex</li>
          <li>Negative log-likelihood of logistic regression is convex</li>
        </ul>

        <h3>Gradient Descent</h3>
        <p>The fundamental update rule:</p>
        <BlockMath math="\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)" />
        <p>
          For an <InlineMath math="L" />-smooth convex function, gradient descent with{" "}
          <InlineMath math="\eta = 1/L" /> converges at rate <InlineMath math="O(1/t)" />.
        </p>

        <h3>Stochastic Gradient Descent (SGD)</h3>
        <p>
          Replace the full gradient with a mini-batch estimate:
        </p>
        <BlockMath math="\theta_{t+1} = \theta_t - \eta \nabla_\theta L_{\mathcal{B}}(\theta_t), \quad \mathcal{B} \subset \text{data}" />
        <p>
          The mini-batch gradient is an unbiased estimate:{" "}
          <InlineMath math="\mathbb{E}[\nabla L_\mathcal{B}] = \nabla L" />. The noise actually helps
          escape sharp local minima and saddle points. For convex <InlineMath math="L" />, SGD converges
          at rate <InlineMath math="O(1/\sqrt{T})" /> with decaying learning rate{" "}
          <InlineMath math="\eta_t = O(1/\sqrt{t})" />.
        </p>

        <h3>SGD with Momentum</h3>
        <p>Accumulate an exponentially weighted moving average of past gradients:</p>
        <BlockMath math="v_t = \beta v_{t-1} + \nabla_\theta L(\theta_t)" />
        <BlockMath math="\theta_{t+1} = \theta_t - \eta v_t" />
        <p>
          Typical <InlineMath math="\beta = 0.9" />. Momentum smooths out oscillations in steep
          dimensions and accelerates progress in consistent directions. Geometrically, it&apos;s like a
          ball rolling downhill with inertia.
        </p>

        <h3>RMSProp</h3>
        <p>Adapt per-parameter learning rates using the second moment of gradients:</p>
        <BlockMath math="s_t = \beta s_{t-1} + (1-\beta) g_t^2" />
        <BlockMath math="\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}} g_t" />
        <p>
          where <InlineMath math="g_t = \nabla_\theta L(\theta_t)" />. Parameters with large
          historical gradients get smaller effective learning rates. This handles sparse gradients
          and varying curvature across dimensions.
        </p>

        <h3>Adam (Adaptive Moment Estimation)</h3>
        <p>Combines momentum (first moment) with RMSProp (second moment) plus bias correction:</p>
        <BlockMath math="m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(1st moment)}" />
        <BlockMath math="v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(2nd moment)}" />
        <BlockMath math="\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(bias correction)}" />
        <BlockMath math="\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t" />
        <p>
          Default hyperparameters: <InlineMath math="\beta_1 = 0.9" />,{" "}
          <InlineMath math="\beta_2 = 0.999" />, <InlineMath math="\epsilon = 10^{-8}" />.
          Bias correction compensates for the zero-initialization of <InlineMath math="m" /> and{" "}
          <InlineMath math="v" />, which matters most in the first few steps.
        </p>

        <h3>Learning Rate Schedules</h3>
        <p><strong>Step decay:</strong></p>
        <BlockMath math="\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}" />
        <p><strong>Cosine annealing:</strong></p>
        <BlockMath math="\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right)" />
        <p><strong>Linear warmup</strong> (used with transformers):</p>
        <BlockMath math="\eta_t = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}, \quad t \le T_{\text{warmup}}" />

        <h3>Second-Order Methods</h3>
        <p>
          Newton&apos;s method uses curvature information from the Hessian:
        </p>
        <BlockMath math="\theta_{t+1} = \theta_t - H^{-1} \nabla_\theta L(\theta_t)" />
        <p>
          This converges quadratically near the optimum but is impractical for neural networks because
          the Hessian is <InlineMath math="O(n^2)" /> to store and <InlineMath math="O(n^3)" /> to invert
          for <InlineMath math="n" /> parameters. Approximations like L-BFGS, natural gradient, and
          K-FAC make this tractable for smaller models.
        </p>

        <h3>KKT Conditions (Constrained Optimization)</h3>
        <p>
          For a constrained problem <InlineMath math="\min f(x)" /> subject to{" "}
          <InlineMath math="g_i(x) \le 0" /> and <InlineMath math="h_j(x) = 0" />, the
          Karush-Kuhn-Tucker conditions are necessary for optimality:
        </p>
        <BlockMath math="\nabla f(x^*) + \sum_i \mu_i \nabla g_i(x^*) + \sum_j \lambda_j \nabla h_j(x^*) = 0" />
        <BlockMath math="\mu_i \ge 0, \quad \mu_i g_i(x^*) = 0 \quad \text{(complementary slackness)}" />
        <p>
          KKT conditions are central to SVMs: the support vectors are exactly the points where the
          inequality constraint is active (<InlineMath math="\mu_i > 0" />).
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Gradient Descent from Scratch</h3>
        <CodeBlock
          language="python"
          title="gradient_descent.py"
          code={`import numpy as np

# ── Gradient descent on f(x,y) = x² + 10y² (elongated bowl) ──
def f(x, y):
    return x**2 + 10 * y**2

def grad_f(x, y):
    return np.array([2 * x, 20 * y])

def gradient_descent(x0, y0, lr, n_steps):
    """Vanilla gradient descent."""
    params = np.array([x0, y0], dtype=float)
    path = [params.copy()]
    for _ in range(n_steps):
        g = grad_f(params[0], params[1])
        params -= lr * g
        path.append(params.copy())
    return np.array(path)

def gd_momentum(x0, y0, lr, beta, n_steps):
    """SGD with momentum."""
    params = np.array([x0, y0], dtype=float)
    v = np.zeros(2)
    path = [params.copy()]
    for _ in range(n_steps):
        g = grad_f(params[0], params[1])
        v = beta * v + g
        params -= lr * v
        path.append(params.copy())
    return np.array(path)

# Compare
path_gd = gradient_descent(5.0, 2.0, lr=0.05, n_steps=50)
path_mom = gd_momentum(5.0, 2.0, lr=0.05, beta=0.9, n_steps=50)

print("Vanilla GD final point:",
      f"({path_gd[-1, 0]:.6f}, {path_gd[-1, 1]:.6f})")
print("Momentum final point:  ",
      f"({path_mom[-1, 0]:.6f}, {path_mom[-1, 1]:.6f})")
print(f"Vanilla GD final loss:  {f(*path_gd[-1]):.8f}")
print(f"Momentum final loss:    {f(*path_mom[-1]):.8f}")
# Momentum converges much faster on this elongated surface!`}
        />

        <h3>Comparing Optimizers on the Rosenbrock Function</h3>
        <CodeBlock
          language="python"
          title="optimizer_comparison.py"
          code={`import numpy as np
import matplotlib.pyplot as plt

# Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
# Minimum at (1, 1). Notoriously hard for gradient methods.
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# ── Optimizer implementations ─────────────────────────────────
def run_sgd(lr, n_steps, x0=(-1.0, -1.0)):
    p = np.array(x0, dtype=float)
    path = [p.copy()]
    for _ in range(n_steps):
        g = rosenbrock_grad(p[0], p[1])
        g = np.clip(g, -10, 10)  # gradient clipping
        p -= lr * g
        path.append(p.copy())
    return np.array(path)

def run_momentum(lr, beta, n_steps, x0=(-1.0, -1.0)):
    p = np.array(x0, dtype=float)
    v = np.zeros(2)
    path = [p.copy()]
    for _ in range(n_steps):
        g = rosenbrock_grad(p[0], p[1])
        g = np.clip(g, -10, 10)
        v = beta * v + g
        p -= lr * v
        path.append(p.copy())
    return np.array(path)

def run_rmsprop(lr, beta, eps, n_steps, x0=(-1.0, -1.0)):
    p = np.array(x0, dtype=float)
    s = np.zeros(2)
    path = [p.copy()]
    for _ in range(n_steps):
        g = rosenbrock_grad(p[0], p[1])
        g = np.clip(g, -10, 10)
        s = beta * s + (1 - beta) * g**2
        p -= lr * g / (np.sqrt(s) + eps)
        path.append(p.copy())
    return np.array(path)

def run_adam(lr, b1, b2, eps, n_steps, x0=(-1.0, -1.0)):
    p = np.array(x0, dtype=float)
    m, v = np.zeros(2), np.zeros(2)
    path = [p.copy()]
    for t in range(1, n_steps + 1):
        g = rosenbrock_grad(p[0], p[1])
        g = np.clip(g, -10, 10)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        m_hat = m / (1 - b1**t)  # bias correction
        v_hat = v / (1 - b2**t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(p.copy())
    return np.array(path)

# ── Run all optimizers ────────────────────────────────────────
n_steps = 5000
paths = {
    "SGD":      run_sgd(lr=0.0005, n_steps=n_steps),
    "Momentum": run_momentum(lr=0.0003, beta=0.9, n_steps=n_steps),
    "RMSProp":  run_rmsprop(lr=0.001, beta=0.9, eps=1e-8, n_steps=n_steps),
    "Adam":     run_adam(lr=0.005, b1=0.9, b2=0.999, eps=1e-8, n_steps=n_steps),
}

# Report final positions and losses
print("Optimizer comparison on Rosenbrock (minimum at (1,1)):")
print("-" * 55)
for name, path in paths.items():
    final = path[-1]
    loss = rosenbrock(final[0], final[1])
    print(f"{name:10s}: final=({final[0]:7.4f}, {final[1]:7.4f}), "
          f"loss={loss:.6f}")

# ── Visualize convergence ─────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: optimization paths on contour plot
x_grid = np.linspace(-2, 2, 200)
y_grid = np.linspace(-1, 3, 200)
X, Y = np.meshgrid(x_grid, y_grid)
Z = rosenbrock(X, Y)

ax1.contour(X, Y, Z, levels=np.logspace(0, 3.5, 20), cmap="viridis")
colors = {"SGD": "red", "Momentum": "blue",
          "RMSProp": "orange", "Adam": "green"}
for name, path in paths.items():
    ax1.plot(path[:500, 0], path[:500, 1], label=name,
             color=colors[name], linewidth=1.5, alpha=0.8)
ax1.plot(1, 1, "k*", markersize=15, label="Minimum")
ax1.set_title("Optimization Paths on Rosenbrock")
ax1.legend(fontsize=9)
ax1.set_xlabel("x"); ax1.set_ylabel("y")

# Right: loss over iterations
for name, path in paths.items():
    losses = [rosenbrock(p[0], p[1]) for p in path]
    ax2.semilogy(losses, label=name, color=colors[name], linewidth=1.5)
ax2.set_title("Loss Convergence (log scale)")
ax2.set_xlabel("Iteration"); ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("optimizer_comparison.png", dpi=150)
plt.show()`}
        />

        <h3>Learning Rate Schedules</h3>
        <CodeBlock
          language="python"
          title="lr_schedules.py"
          code={`import numpy as np
import matplotlib.pyplot as plt

total_steps = 1000

# ── Step Decay ────────────────────────────────────────────────
def step_decay(t, lr0=0.1, gamma=0.5, step_size=200):
    return lr0 * gamma ** (t // step_size)

# ── Cosine Annealing ─────────────────────────────────────────
def cosine_anneal(t, lr_max=0.1, lr_min=1e-5, T=1000):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / T))

# ── Linear Warmup + Cosine Decay (transformer default) ───────
def warmup_cosine(t, lr_max=0.1, warmup=100, T=1000):
    if t < warmup:
        return lr_max * t / warmup
    return 0.5 * lr_max * (1 + np.cos(
        np.pi * (t - warmup) / (T - warmup)))

# ── Exponential Decay ────────────────────────────────────────
def exponential_decay(t, lr0=0.1, decay_rate=0.005):
    return lr0 * np.exp(-decay_rate * t)

steps = np.arange(total_steps)
schedules = {
    "Step Decay":         [step_decay(t) for t in steps],
    "Cosine Annealing":   [cosine_anneal(t) for t in steps],
    "Warmup + Cosine":    [warmup_cosine(t) for t in steps],
    "Exponential Decay":  [exponential_decay(t) for t in steps],
}

fig, ax = plt.subplots(figsize=(10, 5))
for name, lrs in schedules.items():
    ax.plot(steps, lrs, label=name, linewidth=2)
ax.set_xlabel("Training Step")
ax.set_ylabel("Learning Rate")
ax.set_title("Common Learning Rate Schedules")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lr_schedules.png", dpi=150)
plt.show()`}
        />

        <h3>Convergence Analysis</h3>
        <CodeBlock
          language="python"
          title="convergence_analysis.py"
          code={`import numpy as np

# ── How learning rate affects convergence ─────────────────────
# Simple quadratic: f(x) = 0.5 * x^2, gradient = x
# Optimal lr = 1.0 for this function (one-step convergence)

def quadratic_gd(lr, x0=10.0, n_steps=30):
    """Track GD on f(x) = 0.5 * x^2."""
    x = x0
    history = [x]
    for _ in range(n_steps):
        x = x - lr * x  # gradient of 0.5*x^2 is x
        history.append(x)
    return history

learning_rates = [0.1, 0.5, 0.9, 1.0, 1.5, 2.1]
print("Learning rate effect on convergence:")
print("-" * 60)
for lr in learning_rates:
    path = quadratic_gd(lr, x0=10.0, n_steps=30)
    final = path[-1]
    status = "diverged!" if abs(final) > 1e6 else f"{final:.8f}"
    oscillates = any(path[i] * path[i+1] < 0
                     for i in range(min(10, len(path)-1)))
    note = " (oscillating)" if oscillates else ""
    print(f"  lr={lr:.1f}: x_30 = {status}{note}")
# lr=0.1:  slow convergence
# lr=0.5:  moderate convergence
# lr=0.9:  fast but oscillating
# lr=1.0:  exact one-step convergence (optimal!)
# lr=1.5:  oscillating, slow convergence
# lr=2.1:  DIVERGES

# ── Condition number and convergence speed ────────────────────
# For f(x) = 0.5 * x^T A x, condition number kappa = max_eig/min_eig
# determines convergence rate: (kappa-1)/(kappa+1) per step

A_well = np.array([[2.0, 0], [0, 1.5]])     # kappa = 1.33
A_ill  = np.array([[100.0, 0], [0, 1.0]])   # kappa = 100

for name, A in [("Well-conditioned", A_well),
                ("Ill-conditioned", A_ill)]:
    eigs = np.linalg.eigvals(A)
    kappa = max(eigs) / min(eigs)
    rate = (kappa - 1) / (kappa + 1)
    steps_99 = (int(np.log(0.01) / np.log(rate))
                if rate < 1 else float("inf"))
    print(f"\\n{name}: kappa={kappa:.1f}, "
          f"convergence rate={rate:.4f}, "
          f"steps to 99% reduction: ~{steps_99}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Adam is the safe default</strong>: For most deep learning tasks, Adam with <InlineMath math="\eta = 3 \times 10^{-4}" /> works well out of the box. Use it unless you have a reason not to.</li>
          <li><strong>SGD + momentum for final performance</strong>: In many vision tasks, SGD with momentum (0.9) and a cosine schedule slightly outperforms Adam at convergence, though it requires more tuning.</li>
          <li><strong>AdamW (weight decay decoupled)</strong>: Standard Adam conflates L2 regularization with adaptive learning rates. AdamW fixes this and is the default for transformer training.</li>
          <li><strong>Learning rate is the most important hyperparameter</strong>: Try logarithmic sweeps: 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3. A factor of 3 is enough to see big differences.</li>
          <li><strong>Warmup is essential for transformers</strong>: Without it, the initial gradients are huge and training can diverge. Linear warmup for 1-10% of total steps is standard.</li>
          <li><strong>Gradient clipping prevents explosions</strong>: Clip gradients by global norm (typically 1.0) for RNNs and transformers. In PyTorch: <code>torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)</code>.</li>
          <li><strong>Batch size interacts with learning rate</strong>: The linear scaling rule says that when you double the batch size, double the learning rate too. This holds approximately for SGD.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Learning rate too high</strong>: Loss oscillates wildly or goes to NaN. If loss diverges, reduce lr by 10x immediately. This is the most common training failure.</li>
          <li><strong>Forgetting to zero gradients in PyTorch</strong>: PyTorch accumulates gradients by default. Always call <code>optimizer.zero_grad()</code> before <code>loss.backward()</code>. This is such a common bug that PyTorch added <code>set_to_none=True</code> as a faster alternative.</li>
          <li><strong>Assuming the loss landscape is convex</strong>: Neural network losses have saddle points, flat regions, and many local minima. SGD noise and momentum help navigate these, but there are no convergence guarantees to the global minimum.</li>
          <li><strong>Not using bias correction in Adam</strong>: Without bias correction, the first few updates use estimates of <InlineMath math="m" /> and <InlineMath math="v" /> that are biased toward zero. This can cause the initial learning rate to be much larger than intended.</li>
          <li><strong>Confusing L2 regularization with weight decay</strong>: In standard SGD, they are equivalent. In Adam, they are NOT. L2 regularization scales the penalty by the adaptive learning rate; weight decay does not. Always use AdamW for weight decay.</li>
          <li><strong>Using the same learning rate for all layers</strong>: In transfer learning, use smaller learning rates for pretrained layers (e.g., 1e-5) and larger rates for new heads (e.g., 1e-3). This is called discriminative fine-tuning.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Explain why Adam converges faster than vanilla SGD in practice. What are its potential downsides?</p>
        <p><strong>Solution:</strong></p>
        <ol>
          <li>
            <strong>Adaptive learning rates</strong>: Adam maintains per-parameter learning rates via the
            second moment estimate <InlineMath math="v_t" />. Parameters with consistently large gradients
            (e.g., frequent features) get smaller updates; parameters with rare gradients get larger updates.
            This is especially useful for sparse data (NLP embeddings) and varying curvature.
          </li>
          <li>
            <strong>Momentum</strong>: The first moment <InlineMath math="m_t" /> acts as momentum, smoothing
            out gradient noise and accelerating progress through flat regions and narrow valleys.
          </li>
          <li>
            <strong>Bias correction</strong>: Corrects the zero-initialization bias in the early steps,
            preventing the effective learning rate from being too large at the start.
          </li>
          <li>
            <strong>Downsides</strong>:
            <ul>
              <li>Adam can <em>generalize worse</em> than SGD+momentum on some vision tasks (Wilson et al., 2017). The adaptive rates can converge to sharp minima that don&apos;t generalize.</li>
              <li>Adam uses 3x the memory of SGD (stores <InlineMath math="m_t" /> and <InlineMath math="v_t" /> per parameter).</li>
              <li>The original Adam has a convergence bug: the effective learning rate can increase, causing divergence. AMSGrad and AdaBound fix this.</li>
            </ul>
          </li>
        </ol>
        <CodeBlock
          language="python"
          code={`import numpy as np

# Demonstrate why Adam handles varying curvature better
# f(x,y) = x^2 + 100*y^2 — very different curvature in x vs y

def f(x, y): return x**2 + 100 * y**2
def grad(x, y): return np.array([2*x, 200*y])

# SGD: same lr for both — must be small for y-direction
def sgd(lr=0.001, steps=200):
    p = np.array([5.0, 5.0])
    for _ in range(steps):
        p -= lr * grad(p[0], p[1])
    return f(p[0], p[1])

# Adam: adapts lr per parameter — handles curvature difference
def adam(lr=0.1, steps=200):
    p = np.array([5.0, 5.0])
    m, v = np.zeros(2), np.zeros(2)
    for t in range(1, steps + 1):
        g = grad(p[0], p[1])
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        m_hat = m / (1 - 0.9**t)
        v_hat = v / (1 - 0.999**t)
        p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return f(p[0], p[1])

print(f"SGD  final loss (200 steps): {sgd():.6f}")
print(f"Adam final loss (200 steps): {adam():.6f}")
# Adam reaches near-zero much faster on this ill-conditioned problem`}
        />
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Ruder (2016), &quot;An overview of gradient descent optimization algorithms&quot;</strong> &mdash; The definitive survey of SGD variants, clear and comprehensive</li>
          <li><strong>Kingma &amp; Ba (2015), &quot;Adam: A Method for Stochastic Optimization&quot;</strong> &mdash; The original Adam paper; essential reading for understanding adaptive methods</li>
          <li><strong>Boyd &amp; Vandenberghe, &quot;Convex Optimization&quot;</strong> &mdash; The gold-standard textbook for convexity, duality, and KKT conditions (free online)</li>
          <li><strong>Loshchilov &amp; Hutter (2019), &quot;Decoupled Weight Decay Regularization&quot;</strong> &mdash; The AdamW paper explaining why weight decay and L2 regularization differ in adaptive optimizers</li>
          <li><strong>Smith (2018), &quot;A disciplined approach to neural network hyper-parameters&quot;</strong> &mdash; Practical guide to learning rate and batch size selection (1cycle policy)</li>
          <li><strong>CS231n Optimization Notes (Stanford)</strong> &mdash; Excellent visual explanations of optimizer behaviors on loss surfaces</li>
        </ul>
      </TopicSection>
    </div>
  );
}
