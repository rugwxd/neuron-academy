"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function ActivationFunctions() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Without activation functions, a neural network is just a stack of matrix multiplications —
          which collapses into a single linear transformation no matter how many layers you add.
          Activation functions introduce <strong>nonlinearity</strong>, which is what gives neural
          networks their power to learn complex patterns.
        </p>
        <p>
          The history of activation functions tells the story of deep learning itself. <strong>Sigmoid</strong>
          was the original choice (it squashes values to [0, 1], mimicking biological neurons), but it
          causes <strong>vanishing gradients</strong> in deep networks — the gradient shrinks exponentially
          with depth, making early layers nearly impossible to train.
        </p>
        <p>
          <strong>ReLU</strong> (Rectified Linear Unit) solved this in 2012 and enabled the deep learning
          revolution. It&apos;s dead simple — <InlineMath math="\max(0, x)" /> — but its constant gradient
          of 1 for positive values prevents vanishing gradients. The downside: &quot;dead neurons&quot; that
          output zero forever if they get stuck in the negative region.
        </p>
        <p>
          Modern architectures use <strong>GELU</strong> (Gaussian Error Linear Unit) in Transformers and
          <strong>SiLU/Swish</strong> in vision models. These are smooth approximations of ReLU that allow
          small negative values to pass through, providing better gradient flow and slightly better
          performance. The choice of activation function matters less than it used to, but understanding
          <em>why</em> each exists is essential for debugging training failures.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Sigmoid</h3>
        <BlockMath math="\sigma(z) = \frac{1}{1 + e^{-z}}" />
        <BlockMath math="\sigma'(z) = \sigma(z)(1 - \sigma(z))" />
        <p>
          Range: <InlineMath math="(0, 1)" />. Maximum gradient: <InlineMath math="\sigma'(0) = 0.25" />.
          After <InlineMath math="n" /> layers, gradient shrinks by at most <InlineMath math="0.25^n" />.
          Use only for output layers in binary classification.
        </p>

        <h3>Tanh</h3>
        <BlockMath math="\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1" />
        <BlockMath math="\tanh'(z) = 1 - \tanh^2(z)" />
        <p>
          Range: <InlineMath math="(-1, 1)" />. Zero-centered (unlike sigmoid), so gradients don&apos;t
          have a systematic bias. Maximum gradient: <InlineMath math="\tanh'(0) = 1" />. Still suffers
          from vanishing gradients but less severely than sigmoid.
        </p>

        <h3>ReLU (Rectified Linear Unit)</h3>
        <BlockMath math="\text{ReLU}(z) = \max(0, z)" />
        <BlockMath math="\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}" />
        <p>
          No vanishing gradient for positive values (gradient is exactly 1). Computationally cheap.
          Problem: &quot;dying ReLU&quot; — neurons with permanently negative inputs have zero gradient
          and never recover.
        </p>

        <h3>Leaky ReLU</h3>
        <BlockMath math="\text{LeakyReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \leq 0 \end{cases}" />
        <p>
          where <InlineMath math="\alpha = 0.01" /> (typically). Fixes the dying ReLU problem by allowing
          a small gradient for negative values.
        </p>

        <h3>GELU (Gaussian Error Linear Unit)</h3>
        <BlockMath math="\text{GELU}(z) = z \cdot \Phi(z) = z \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{z}{\sqrt{2}}\right)\right]" />
        <p>Practical approximation:</p>
        <BlockMath math="\text{GELU}(z) \approx 0.5z\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(z + 0.044715z^3)\right]\right)" />
        <p>
          GELU is the default activation in GPT, BERT, and most Transformers. It&apos;s smooth, allows
          small negative values, and can be interpreted as a <strong>stochastic regularizer</strong> —
          it multiplies each input by a Bernoulli random variable whose probability depends on the input
          magnitude.
        </p>

        <h3>SiLU / Swish</h3>
        <BlockMath math="\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}" />
        <BlockMath math="\text{SiLU}'(z) = \sigma(z) + z \cdot \sigma(z)(1 - \sigma(z)) = \sigma(z)(1 + z(1 - \sigma(z)))" />
        <p>
          Found by neural architecture search (Google, 2017). Non-monotonic — has a small dip below zero.
          Default in EfficientNet, ConvNeXt, and many vision models. Very similar to GELU in practice.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>All Activations — Implementation and Visualization</h3>
        <CodeBlock
          language="python"
          title="activations_comparison.py"
          code={`import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

z = torch.linspace(-5, 5, 1000, requires_grad=False)

activations = {
    "Sigmoid": torch.sigmoid(z),
    "Tanh": torch.tanh(z),
    "ReLU": F.relu(z),
    "LeakyReLU": F.leaky_relu(z, negative_slope=0.01),
    "GELU": F.gelu(z),
    "SiLU (Swish)": F.silu(z),
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (name, values) in zip(axes.flat, activations.items()):
    ax.plot(z.numpy(), values.numpy(), linewidth=2)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)
    ax.set_title(name, fontsize=14)
    ax.set_xlim(-5, 5)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("activations.png", dpi=150)
plt.show()`}
        />

        <h3>Vanishing Gradient Demonstration</h3>
        <CodeBlock
          language="python"
          title="vanishing_gradient_demo.py"
          code={`import torch
import torch.nn as nn

def test_gradient_flow(activation_fn, n_layers=20, input_dim=256):
    """Measure gradient magnitude through a deep network."""
    layers = []
    for _ in range(n_layers):
        layers.append(nn.Linear(input_dim, input_dim))
        layers.append(activation_fn())

    model = nn.Sequential(*layers)

    # Xavier initialization
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    x = torch.randn(32, input_dim)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Check gradient magnitude at each layer
    grad_norms = []
    for name, param in model.named_parameters():
        if "weight" in name:
            grad_norms.append(param.grad.norm().item())

    return grad_norms

# Compare activation functions
for name, act_fn in [
    ("Sigmoid", nn.Sigmoid),
    ("Tanh",    nn.Tanh),
    ("ReLU",    nn.ReLU),
    ("GELU",    nn.GELU),
]:
    grads = test_gradient_flow(act_fn, n_layers=20)
    print(f"{name:10s} | First layer grad: {grads[0]:.2e} | "
          f"Last layer grad: {grads[-1]:.2e} | "
          f"Ratio: {grads[0]/grads[-1]:.2e}")

# Typical output:
# Sigmoid    | First layer grad: 1.52e-10 | Last layer grad: 4.21e-01 | Ratio: 3.61e-10  <- VANISHING!
# Tanh       | First layer grad: 2.37e-05 | Last layer grad: 3.89e-01 | Ratio: 6.09e-05  <- Still bad
# ReLU       | First layer grad: 1.82e-01 | Last layer grad: 3.15e-01 | Ratio: 5.78e-01  <- Healthy!
# GELU       | First layer grad: 1.65e-01 | Last layer grad: 2.98e-01 | Ratio: 5.54e-01  <- Healthy!`}
        />

        <h3>Custom Activation in PyTorch</h3>
        <CodeBlock
          language="python"
          title="custom_activation.py"
          code={`import torch
import torch.nn as nn

class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))
    Another smooth ReLU variant, used in YOLOv4."""
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

# Or use a simple lambda (autograd handles derivatives automatically)
class SquaredReLU(nn.Module):
    """Squared ReLU: max(0, x)^2 — used in Primer (2021)."""
    def forward(self, x):
        return torch.relu(x) ** 2

# Test
x = torch.randn(5, requires_grad=True)
y = Mish()(x)
y.sum().backward()
print(f"Mish output:    {y.data}")
print(f"Mish gradient:  {x.grad}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>Default choice for hidden layers: ReLU</strong>. It&apos;s fast, simple, and works
            well in most architectures (CNNs, basic MLPs). Use GELU for Transformers and SiLU for
            modern vision models.
          </li>
          <li>
            <strong>Output layer activation depends on the task</strong>:
            sigmoid for binary classification, softmax for multi-class, linear (no activation) for
            regression, tanh for outputs in [-1, 1].
          </li>
          <li>
            <strong>Initialization must match the activation</strong>: use He initialization
            (<InlineMath math="\text{Var}(w) = 2/n_{\text{in}}" />) for ReLU, Xavier initialization
            (<InlineMath math="\text{Var}(w) = 2/(n_{\text{in}} + n_{\text{out}})" />) for sigmoid/tanh.
            Wrong initialization + wrong activation = training failure.
          </li>
          <li>
            <strong>Batch normalization reduces sensitivity to activation choice</strong>: by normalizing
            layer inputs, BatchNorm keeps activations in the &quot;good&quot; range regardless of
            which activation you use. This is one reason it&apos;s so popular.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Using sigmoid/tanh in hidden layers of deep networks</strong>: gradients will
            vanish. Use ReLU, GELU, or SiLU. Sigmoid is only appropriate for the output layer of
            binary classification.
          </li>
          <li>
            <strong>Ignoring dying ReLU</strong>: if you see many neurons with zero output during
            training, they may be dead. Solutions: use Leaky ReLU, lower learning rate, or use
            batch normalization before the activation.
          </li>
          <li>
            <strong>Applying softmax inside the model and in the loss function</strong>: PyTorch&apos;s
            <code>CrossEntropyLoss</code> includes softmax internally. Applying softmax in your model
            AND in the loss = double softmax, which causes subtle training bugs.
          </li>
          <li>
            <strong>Thinking smooth activations are always better</strong>: GELU and SiLU are marginally
            better in Transformers but the gains are small (0.1-0.3%). ReLU is fine for most tasks and
            is significantly faster to compute.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> Why does ReLU work better than sigmoid in deep networks?
          What is the dying ReLU problem and how do you fix it? Compare GELU to ReLU.
        </p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Why ReLU beats sigmoid</strong>:
            <ul>
              <li>Sigmoid&apos;s maximum gradient is 0.25. After <InlineMath math="n" /> layers, the gradient
              is at most <InlineMath math="0.25^n" />, which vanishes exponentially.</li>
              <li>ReLU&apos;s gradient is exactly 1 for positive inputs — gradients flow unchanged through
              the network, regardless of depth.</li>
              <li>ReLU is also computationally cheaper (just a comparison vs. exponentiation).</li>
            </ul>
          </li>
          <li>
            <strong>Dying ReLU</strong>: if a neuron&apos;s input is always negative (e.g., due to a
            large negative bias after a bad weight update), ReLU outputs 0 with gradient 0 permanently.
            The neuron is &quot;dead&quot; — it can never recover.
            Fixes: Leaky ReLU (<InlineMath math="\alpha = 0.01" /> for negative inputs), PReLU (learnable <InlineMath math="\alpha" />),
            ELU, or careful initialization + learning rate tuning.
          </li>
          <li>
            <strong>GELU vs ReLU</strong>: GELU is smooth (differentiable everywhere, unlike ReLU at
            zero) and allows small negative values through. This provides better gradient flow near zero
            and acts as a soft gate based on input magnitude. GELU is standard in Transformers (BERT, GPT)
            because the smooth gradient helps with the attention mechanism&apos;s optimization landscape.
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Nair &amp; Hinton (2010) &quot;Rectified Linear Units Improve Restricted Boltzmann Machines&quot;</strong> — The ReLU paper.</li>
          <li><strong>Hendrycks &amp; Gimpel (2016) &quot;Gaussian Error Linear Units (GELUs)&quot;</strong> — The GELU paper.</li>
          <li><strong>Ramachandran et al. (2017) &quot;Searching for Activation Functions&quot;</strong> — How Google found Swish/SiLU via automated search.</li>
          <li><strong>He et al. (2015) &quot;Delving Deep into Rectifiers&quot;</strong> — He/Kaiming initialization for ReLU networks.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
