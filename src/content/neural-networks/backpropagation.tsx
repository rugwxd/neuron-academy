"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function Backpropagation() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Backpropagation is how neural networks <strong>learn</strong>. It answers the question: &quot;how
          much did each weight contribute to the error?&quot; — and then adjusts every weight proportionally.
        </p>
        <p>
          The idea is deceptively simple. In the <strong>forward pass</strong>, data flows through the
          network layer by layer, producing a prediction. You compare that prediction to the true answer
          using a loss function. In the <strong>backward pass</strong>, you trace the error backward through
          the network using the chain rule of calculus, computing the gradient of the loss with respect to
          every single weight. Then you nudge each weight in the direction that reduces the error.
        </p>
        <p>
          That&apos;s it. Backpropagation is just the <strong>chain rule applied systematically</strong>
          to a computational graph. What makes it efficient is that it reuses intermediate computations —
          instead of computing the gradient for each weight independently (which would be absurdly expensive),
          it propagates gradients backward through the graph, sharing computation at every node.
        </p>
        <p>
          Every deep learning framework (PyTorch, TensorFlow, JAX) implements backpropagation automatically
          via <strong>autograd</strong> — you build the forward computation and the framework traces it,
          building a computational graph that it then traverses in reverse to compute all gradients in one
          backward pass.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Computational Graph View</h3>
        <p>
          Consider a 2-layer network with input <InlineMath math="x" />, weights <InlineMath math="W_1, W_2" />,
          biases <InlineMath math="b_1, b_2" />, activation <InlineMath math="\sigma" />, and MSE loss:
        </p>
        <BlockMath math="z_1 = W_1 x + b_1" />
        <BlockMath math="a_1 = \sigma(z_1)" />
        <BlockMath math="z_2 = W_2 a_1 + b_2" />
        <BlockMath math="L = \frac{1}{2}(z_2 - y)^2" />

        <h3>Backward Pass (Chain Rule)</h3>
        <p>Start from the loss and work backward:</p>
        <BlockMath math="\frac{\partial L}{\partial z_2} = z_2 - y" />
        <BlockMath math="\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z_2} \cdot a_1^T" />
        <BlockMath math="\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial z_2}" />
        <BlockMath math="\frac{\partial L}{\partial a_1} = W_2^T \cdot \frac{\partial L}{\partial z_2}" />
        <BlockMath math="\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \odot \sigma'(z_1)" />
        <BlockMath math="\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_1} \cdot x^T" />
        <BlockMath math="\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1}" />
        <p>
          The symbol <InlineMath math="\odot" /> denotes element-wise multiplication. Notice how each
          gradient depends on the gradient from the layer above — this is the &quot;backward propagation&quot;
          of errors.
        </p>

        <h3>Concrete Example: Single Neuron</h3>
        <p>
          Let <InlineMath math="x = 2" />, <InlineMath math="w = 0.5" />, <InlineMath math="b = 0.1" />,
          <InlineMath math="y = 1" />, with sigmoid activation and MSE loss:
        </p>
        <BlockMath math="z = wx + b = 0.5 \cdot 2 + 0.1 = 1.1" />
        <BlockMath math="a = \sigma(z) = \frac{1}{1 + e^{-1.1}} \approx 0.7503" />
        <BlockMath math="L = \frac{1}{2}(a - y)^2 = \frac{1}{2}(0.7503 - 1)^2 \approx 0.0312" />
        <p>Backward:</p>
        <BlockMath math="\frac{\partial L}{\partial a} = a - y = 0.7503 - 1 = -0.2497" />
        <BlockMath math="\frac{\partial a}{\partial z} = \sigma(z)(1 - \sigma(z)) = 0.7503 \cdot 0.2497 \approx 0.1874" />
        <BlockMath math="\frac{\partial L}{\partial z} = -0.2497 \cdot 0.1874 \approx -0.0468" />
        <BlockMath math="\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot x = -0.0468 \cdot 2 = -0.0936" />
        <BlockMath math="\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} = -0.0468" />
        <p>
          The negative gradients tell us: <strong>increase</strong> both <InlineMath math="w" /> and <InlineMath math="b" /> to
          reduce the loss (the prediction was too low).
        </p>

        <h3>Vector/Matrix Form for a Batch</h3>
        <p>
          For a mini-batch of <InlineMath math="m" /> samples, the gradients become:
        </p>
        <BlockMath math="\frac{\partial L}{\partial W} = \frac{1}{m} \frac{\partial L}{\partial Z} \cdot A_{\text{prev}}^T" />
        <BlockMath math="\frac{\partial L}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \frac{\partial L}{\partial z_i}" />
      </TopicSection>

      <TopicSection type="code">
        <h3>Backpropagation from Scratch — Full 2-Layer Network</h3>
        <CodeBlock
          language="python"
          title="backprop_scratch.py"
          code={`import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# Network: 2 -> 4 -> 1 (XOR problem)
np.random.seed(42)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # (2, 4)
Y = np.array([[0, 1, 1, 0]])                        # (1, 4)

# Initialize weights (Xavier initialization)
W1 = np.random.randn(4, 2) * np.sqrt(2.0 / 2)  # (4, 2)
b1 = np.zeros((4, 1))                            # (4, 1)
W2 = np.random.randn(1, 4) * np.sqrt(2.0 / 4)  # (1, 4)
b2 = np.zeros((1, 1))                            # (1, 1)

lr = 1.0
for epoch in range(10000):
    # ---- FORWARD PASS ----
    Z1 = W1 @ X + b1           # (4, 4)
    A1 = sigmoid(Z1)           # (4, 4)
    Z2 = W2 @ A1 + b2          # (1, 4)
    A2 = sigmoid(Z2)           # (1, 4)

    # Loss
    m = X.shape[1]
    loss = mse_loss(A2, Y)

    # ---- BACKWARD PASS ----
    # dL/dA2
    dA2 = (2 / m) * (A2 - Y)              # (1, 4)

    # dL/dZ2 = dL/dA2 * sigmoid'(Z2)
    dZ2 = dA2 * sigmoid_deriv(Z2)          # (1, 4)

    # dL/dW2 = dZ2 @ A1.T
    dW2 = dZ2 @ A1.T                       # (1, 4)
    db2 = np.sum(dZ2, axis=1, keepdims=True)  # (1, 1)

    # dL/dA1 = W2.T @ dZ2
    dA1 = W2.T @ dZ2                       # (4, 4)

    # dL/dZ1 = dA1 * sigmoid'(Z1)
    dZ1 = dA1 * sigmoid_deriv(Z1)          # (4, 4)

    # dL/dW1 = dZ1 @ X.T
    dW1 = dZ1 @ X.T                        # (4, 2)
    db1 = np.sum(dZ1, axis=1, keepdims=True)  # (4, 1)

    # ---- UPDATE ----
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if (epoch + 1) % 2000 == 0:
        preds = (A2 > 0.5).astype(int)
        acc = np.mean(preds == Y)
        print(f"Epoch {epoch+1}: loss={loss:.6f}, acc={acc:.0%}")

# Final predictions
print("\\nPredictions:", np.round(A2, 3))`}
        />

        <h3>Verifying Gradients with PyTorch Autograd</h3>
        <CodeBlock
          language="python"
          title="verify_with_autograd.py"
          code={`import torch
import torch.nn as nn

# Same single-neuron example from the math section
x = torch.tensor([2.0])
w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)
y = torch.tensor([1.0])

# Forward pass
z = w * x + b                      # z = 1.1
a = torch.sigmoid(z)               # a ~ 0.7503
loss = 0.5 * (a - y) ** 2          # L ~ 0.0312

# Backward pass — PyTorch does this automatically
loss.backward()

print(f"z = {z.item():.4f}")
print(f"a = {a.item():.4f}")
print(f"loss = {loss.item():.4f}")
print(f"dL/dw = {w.grad.item():.4f}")   # Should be ~ -0.0936
print(f"dL/db = {b.grad.item():.4f}")   # Should be ~ -0.0468

# These match our hand-computed values!`}
        />

        <h3>How PyTorch Autograd Works Under the Hood</h3>
        <CodeBlock
          language="python"
          title="autograd_internals.py"
          code={`import torch

# Every tensor with requires_grad=True records operations
a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)

# PyTorch builds a computational graph as you compute
c = a * b        # MulBackward
d = c + a        # AddBackward
e = d ** 2        # PowBackward

# The graph is a DAG (directed acyclic graph)
print(f"e.grad_fn = {e.grad_fn}")           # PowBackward0
print(f"e.grad_fn.next_functions = {e.grad_fn.next_functions}")

# backward() traverses this graph in reverse topological order
e.backward()

# Analytical: e = (a*b + a)^2 = (2*3 + 2)^2 = 64
# de/da = 2*(a*b + a)*(b + 1) = 2*8*(3+1) = 64
# de/db = 2*(a*b + a)*a = 2*8*2 = 32
print(f"de/da = {a.grad.item()}")  # 64.0
print(f"de/db = {b.grad.item()}")  # 32.0`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>You never implement backprop manually in production</strong>: PyTorch, JAX, and
            TensorFlow handle it automatically. But understanding it is essential for debugging gradient
            issues (vanishing, exploding, NaN).
          </li>
          <li>
            <strong>Gradient checkpointing saves memory</strong>: for very deep networks, storing all
            intermediate activations for the backward pass uses enormous memory. Gradient checkpointing
            recomputes activations during the backward pass, trading compute for memory (~30% slower
            but ~60% less memory).
          </li>
          <li>
            <strong>Mixed precision training</strong>: use float16 for forward/backward computation but
            float32 for weight updates. This halves memory and doubles speed on modern GPUs with
            almost no accuracy loss.
          </li>
          <li>
            <strong>Gradient accumulation</strong>: if your batch doesn&apos;t fit in GPU memory, accumulate
            gradients over multiple mini-batches before calling <code>optimizer.step()</code>. This simulates
            a larger effective batch size.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Forgetting to zero gradients</strong>: PyTorch accumulates gradients by default.
            Always call <code>optimizer.zero_grad()</code> (or <code>model.zero_grad()</code>) before
            <code>loss.backward()</code>. Forgetting this causes gradients to grow across iterations.
          </li>
          <li>
            <strong>Calling backward() on a non-scalar</strong>: <code>loss.backward()</code> only works
            if <code>loss</code> is a scalar. If your loss is a vector, you need to pass
            a <code>gradient</code> argument or sum/mean the loss first.
          </li>
          <li>
            <strong>In-place operations break autograd</strong>: operations like <code>x += 1</code> or
            <code>x[:, 0] = 0</code> modify tensors in-place and can corrupt the computational graph.
            Use <code>x = x + 1</code> instead.
          </li>
          <li>
            <strong>Confusing .detach() with torch.no_grad()</strong>: <code>.detach()</code> removes a
            tensor from the computational graph (use for targets or when mixing networks).
            <code>torch.no_grad()</code> disables gradient computation entirely (use for inference/evaluation).
          </li>
          <li>
            <strong>Not understanding vanishing gradients</strong>: in deep networks with sigmoid activations,
            gradients shrink exponentially because <InlineMath math="\sigma'(z) \leq 0.25" />. After 10
            layers, the gradient is at most <InlineMath math="0.25^{10} \approx 10^{-6}" />. This is why
            ReLU and residual connections were invented.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> Walk me through backpropagation in a 2-layer neural network.
          What is the time and space complexity of the backward pass relative to the forward pass?
        </p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Forward pass</strong>: compute <InlineMath math="z_1 = W_1 x + b_1" />, <InlineMath math="a_1 = \sigma(z_1)" />,
            <InlineMath math="z_2 = W_2 a_1 + b_2" />, loss <InlineMath math="L" />. Store all intermediate values.
          </li>
          <li>
            <strong>Backward pass</strong>: compute <InlineMath math="\partial L / \partial z_2" />, then
            use it to compute <InlineMath math="\partial L / \partial W_2" /> (via outer product with <InlineMath math="a_1" />)
            and <InlineMath math="\partial L / \partial a_1" /> (via <InlineMath math="W_2^T" />).
            Continue to <InlineMath math="\partial L / \partial z_1" /> (element-wise multiply with <InlineMath math="\sigma'(z_1)" />)
            and <InlineMath math="\partial L / \partial W_1" />.
          </li>
          <li>
            <strong>Time complexity</strong>: the backward pass has approximately the <strong>same
            computational cost</strong> as the forward pass — roughly <InlineMath math="2\times" /> because
            each layer involves one matrix multiply in the forward pass and two in the backward pass
            (one for the parameter gradient, one for the input gradient). Overall: <InlineMath math="O(3 \times \text{forward})" />.
          </li>
          <li>
            <strong>Space complexity</strong>: you must store all intermediate activations
            (<InlineMath math="z_1, a_1, z_2" />) from the forward pass to use during the backward pass.
            This is <InlineMath math="O(\text{batch} \times \text{width} \times \text{depth})" /> and is
            typically the bottleneck for training large models.
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Rumelhart, Hinton &amp; Williams (1986) &quot;Learning representations by back-propagating errors&quot;</strong> — The paper that made backpropagation famous.</li>
          <li><strong>CS231n Backpropagation lecture notes</strong> — Stanford&apos;s exceptional treatment of computational graphs and backprop.</li>
          <li><strong>Karpathy &quot;Yes you should understand backprop&quot;</strong> — Blog post on why understanding backprop matters even with autograd.</li>
          <li><strong>Baydin et al. (2018) &quot;Automatic Differentiation in Machine Learning: a Survey&quot;</strong> — Comprehensive overview of forward-mode vs. reverse-mode AD.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
