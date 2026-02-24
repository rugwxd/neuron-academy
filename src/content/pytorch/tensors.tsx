"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function Tensors() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          A tensor is just a <strong>multi-dimensional array</strong> — a generalization of scalars (0D),
          vectors (1D), matrices (2D) to any number of dimensions. A batch of color images is a 4D tensor
          with shape <code>(batch, channels, height, width)</code>. A sequence of word embeddings is a 3D
          tensor with shape <code>(batch, seq_len, embedding_dim)</code>.
        </p>
        <p>
          What makes PyTorch tensors special isn&apos;t the multi-dimensional storage — NumPy has that.
          It&apos;s <strong>autograd</strong>: PyTorch can automatically compute derivatives of any
          computation involving tensors. When you set <code>requires_grad=True</code>, PyTorch silently
          records every operation you perform on that tensor, building a <strong>computational graph</strong>.
          When you call <code>.backward()</code>, it traverses this graph in reverse to compute the gradient
          of the output with respect to every tensor that required gradients.
        </p>
        <p>
          This is the backbone of deep learning: define a forward computation (the model), compute a loss,
          call <code>.backward()</code>, and autograd gives you all the gradients you need to update the
          parameters. No manual calculus required.
        </p>
        <p>
          Under the hood, tensors can live on CPU or GPU. Moving a tensor to GPU with <code>.to(&quot;cuda&quot;)</code>
          enables massively parallel computation — matrix multiplications that take seconds on CPU happen in
          milliseconds on GPU. This is why deep learning requires GPUs.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Automatic Differentiation</h3>
        <p>
          PyTorch implements <strong>reverse-mode automatic differentiation</strong>. For a function
          <InlineMath math="f: \mathbb{R}^n \to \mathbb{R}" />, it computes the full
          gradient <InlineMath math="\nabla f \in \mathbb{R}^n" /> in a single backward pass, regardless
          of <InlineMath math="n" />.
        </p>
        <p>
          Given a computational graph with intermediate variables <InlineMath math="v_1, v_2, \ldots, v_k" />:
        </p>
        <BlockMath math="\frac{\partial f}{\partial x_i} = \sum_{\text{paths } x_i \to f} \prod_{\text{edges}} \frac{\partial v_{\text{child}}}{\partial v_{\text{parent}}}" />

        <h3>Jacobian-Vector Product (JVP) vs. Vector-Jacobian Product (VJP)</h3>
        <p>
          For <InlineMath math="f: \mathbb{R}^n \to \mathbb{R}^m" /> with Jacobian <InlineMath math="J \in \mathbb{R}^{m \times n}" />:
        </p>
        <ul>
          <li><strong>Forward mode (JVP)</strong>: computes <InlineMath math="Jv" /> for a tangent vector <InlineMath math="v \in \mathbb{R}^n" />. Cost: <InlineMath math="O(1)" /> forward passes per input dimension.</li>
          <li><strong>Reverse mode (VJP)</strong>: computes <InlineMath math="v^T J" /> for an adjoint vector <InlineMath math="v \in \mathbb{R}^m" />. Cost: <InlineMath math="O(1)" /> backward passes per output dimension.</li>
        </ul>
        <p>
          Deep learning has many inputs (parameters) and one output (scalar loss), so reverse mode
          (backpropagation) is far more efficient: one backward pass computes <em>all</em> gradients.
        </p>

        <h3>Broadcasting Rules</h3>
        <p>
          When operating on tensors of different shapes, PyTorch broadcasts by aligning dimensions
          from the right and expanding dimensions of size 1:
        </p>
        <BlockMath math="(3, 4) + (4,) \to (3, 4) + (1, 4) \to (3, 4)" />
        <BlockMath math="(2, 1, 5) \times (3, 5) \to (2, 1, 5) \times (1, 3, 5) \to (2, 3, 5)" />
      </TopicSection>

      <TopicSection type="code">
        <h3>Tensor Basics</h3>
        <CodeBlock
          language="python"
          title="tensor_basics.py"
          code={`import torch

# ---- Creation ----
a = torch.tensor([1.0, 2.0, 3.0])              # From a list
b = torch.zeros(3, 4)                           # 3x4 zeros
c = torch.randn(2, 3, 4)                        # Random normal (2x3x4)
d = torch.arange(0, 10, 2)                      # [0, 2, 4, 6, 8]
e = torch.eye(3)                                # 3x3 identity matrix
f = torch.linspace(0, 1, steps=5)               # [0, 0.25, 0.5, 0.75, 1.0]

# ---- Shape operations ----
x = torch.randn(2, 3, 4)
print(x.shape)                                   # torch.Size([2, 3, 4])
print(x.view(6, 4).shape)                        # (6, 4) — reshape
print(x.permute(2, 0, 1).shape)                  # (4, 2, 3) — transpose
print(x.unsqueeze(0).shape)                      # (1, 2, 3, 4) — add dim
print(x.unsqueeze(0).squeeze(0).shape)           # (2, 3, 4) — remove dim

# ---- Indexing (same as NumPy) ----
print(x[0, :, :2].shape)                         # (3, 2)
print(x[:, 1:, ::2].shape)                       # (2, 2, 2)

# ---- Math operations ----
a = torch.randn(3, 4)
b = torch.randn(3, 4)
c = a + b                                        # Element-wise
d = a @ b.T                                      # Matrix multiply (3x3)
e = a * b                                        # Hadamard (element-wise) product
f = a.sum(dim=1)                                 # Sum along dim 1 (shape: (3,))
g = a.softmax(dim=1)                             # Softmax along dim 1

# ---- Device management ----
if torch.cuda.is_available():
    gpu_tensor = a.to("cuda")                    # Move to GPU
    result = gpu_tensor @ gpu_tensor.T            # Computation on GPU
    cpu_result = result.to("cpu")                 # Move back to CPU

# ---- NumPy interop ----
import numpy as np
np_arr = np.array([1.0, 2.0, 3.0])
tensor = torch.from_numpy(np_arr)                # Zero-copy (shares memory!)
back_to_np = tensor.numpy()                      # Zero-copy back`}
        />

        <h3>Autograd in Depth</h3>
        <CodeBlock
          language="python"
          title="autograd_deep_dive.py"
          code={`import torch

# ---- Basic autograd ----
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1      # y = (x+1)^2

y.backward()              # dy/dx = 2x + 2 = 8
print(f"dy/dx at x=3: {x.grad}")  # tensor(8.)

# ---- Autograd with vectors ----
x = torch.randn(3, requires_grad=True)
y = (x**2).sum()          # scalar output

y.backward()
print(f"dy/dx: {x.grad}")          # 2 * x
print(f"Expected: {2 * x.data}")   # Same!

# ---- Computational graph inspection ----
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = a * b                  # MulBackward0
d = c + a                  # AddBackward0
e = d.relu()               # ReluBackward0

print(f"e.grad_fn: {e.grad_fn}")
print(f"Graph: e <- {e.grad_fn.next_functions}")

e.backward()
print(f"de/da: {a.grad}")  # d(relu(ab + a))/da = b + 1 = 4 (since ab+a=8 > 0)
print(f"de/db: {b.grad}")  # d(relu(ab + a))/db = a = 2

# ---- torch.no_grad() for inference ----
model_param = torch.tensor(1.0, requires_grad=True)

# During inference, disable gradient tracking (saves memory and speed)
with torch.no_grad():
    output = model_param * 5
    print(f"requires_grad: {output.requires_grad}")  # False

# ---- .detach() to stop gradient flow ----
x = torch.tensor(2.0, requires_grad=True)
y = x**2
z = y.detach()  # z has same value as y but no gradient connection
w = z * 3

# w.backward() would NOT propagate gradients back to x

# ---- Second-order gradients ----
x = torch.tensor(2.0, requires_grad=True)
y = x**3                    # y = x^3

# First derivative
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx = {dy_dx}")   # 3 * x^2 = 12

# Second derivative
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(f"d2y/dx2 = {d2y_dx2}")  # 6 * x = 12`}
        />

        <h3>Building a Neural Network with Raw Tensors</h3>
        <CodeBlock
          language="python"
          title="nn_with_raw_tensors.py"
          code={`import torch
import torch.nn.functional as F

# 2-layer network with autograd (no nn.Module!)
torch.manual_seed(42)
n_in, n_hidden, n_out = 784, 128, 10

# Parameters — requires_grad=True so autograd tracks them
W1 = torch.randn(n_in, n_hidden, requires_grad=True) * 0.01
b1 = torch.zeros(n_hidden, requires_grad=True)
W2 = torch.randn(n_hidden, n_out, requires_grad=True) * 0.01
b2 = torch.zeros(n_out, requires_grad=True)

# Fake data (like MNIST)
X = torch.randn(64, 784)    # batch of 64 images
y = torch.randint(0, 10, (64,))

lr = 0.1
for step in range(100):
    # Forward pass — autograd records this
    h = (X @ W1 + b1).relu()       # Hidden layer
    logits = h @ W2 + b2           # Output layer
    loss = F.cross_entropy(logits, y)

    # Backward pass — autograd computes all gradients
    loss.backward()

    # Update (inside no_grad to prevent tracking)
    with torch.no_grad():
        W1 -= lr * W1.grad
        b1 -= lr * b1.grad
        W2 -= lr * W2.grad
        b2 -= lr * b2.grad

        # Zero gradients for next iteration
        W1.grad.zero_()
        b1.grad.zero_()
        W2.grad.zero_()
        b2.grad.zero_()

    if (step + 1) % 20 == 0:
        acc = (logits.argmax(dim=1) == y).float().mean()
        print(f"Step {step+1}: loss={loss.item():.4f}, acc={acc:.2%}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>Always check your tensor shapes</strong>: the majority of PyTorch bugs come from
            shape mismatches. Use <code>.shape</code> liberally and add assert statements:
            <code>assert x.shape == (batch, seq_len, d_model)</code>.
          </li>
          <li>
            <strong>Use .to(device) consistently</strong>: all tensors in a computation must be on the
            same device. A common pattern: <code>device = torch.device(&quot;cuda&quot; if torch.cuda.is_available() else &quot;cpu&quot;)</code>
            then <code>model.to(device)</code> and <code>x.to(device)</code>.
          </li>
          <li>
            <strong>Avoid unnecessary copies</strong>: <code>torch.from_numpy()</code> shares memory
            with the numpy array (zero-copy). <code>.view()</code> shares memory with the original
            tensor. <code>.contiguous().view()</code> or <code>.reshape()</code> may copy if needed.
          </li>
          <li>
            <strong>Use torch.no_grad() during inference</strong>: this disables gradient computation,
            saving memory (no computational graph) and time. Always wrap validation/test loops in it.
          </li>
          <li>
            <strong>Mixed precision with torch.autocast</strong>: wrapping the forward pass in
            <code>torch.autocast(&quot;cuda&quot;)</code> automatically uses float16 where safe and float32
            where needed. This roughly doubles training speed.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Forgetting to zero gradients</strong>: PyTorch accumulates gradients by default.
            If you forget <code>.zero_grad()</code>, gradients from previous steps add up, causing
            incorrect updates. This is the single most common PyTorch bug.
          </li>
          <li>
            <strong>Tensor on wrong device</strong>: you&apos;ll get errors like &quot;expected all tensors
            to be on the same device.&quot; Check with <code>tensor.device</code> and move with
            <code>.to(device)</code>.
          </li>
          <li>
            <strong>Modifying tensors in-place with autograd</strong>: operations like <code>x += 1</code> or
            <code>x[0] = 5</code> modify the tensor in-place, which can corrupt the computational graph.
            Use <code>x = x + 1</code> instead if autograd is involved.
          </li>
          <li>
            <strong>Confusing .view() with .reshape()</strong>: <code>.view()</code> requires the tensor
            to be contiguous in memory (it returns a view, not a copy). <code>.reshape()</code> works
            always but may copy. Use <code>.reshape()</code> when you don&apos;t care about copying,
            <code>.view()</code> when you want to guarantee zero-copy.
          </li>
          <li>
            <strong>Not detaching tensors stored for logging</strong>: if you append
            <code>loss</code> to a list for plotting, the entire computational graph stays in memory.
            Use <code>loss.item()</code> or <code>loss.detach()</code> first.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> Explain how PyTorch&apos;s autograd system works. What is a
          computational graph? Why does PyTorch use dynamic graphs while TensorFlow (v1) used static ones?
        </p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Computational graph</strong>: a directed acyclic graph (DAG) where nodes are
            operations (add, multiply, relu) and edges carry tensors. PyTorch records this graph as
            you execute Python code — each operation creates a <code>grad_fn</code> node.
          </li>
          <li>
            <strong>How backward works</strong>: calling <code>.backward()</code> on the loss triggers
            a reverse traversal of the graph. At each node, the local Jacobian is multiplied by the
            incoming gradient (vector-Jacobian product), and the result is propagated to parent nodes.
            This is reverse-mode automatic differentiation.
          </li>
          <li>
            <strong>Dynamic vs. static graphs</strong>:
            <ul>
              <li><em>Dynamic (PyTorch)</em>: the graph is built on-the-fly during each forward pass and
              destroyed after backward. This allows standard Python control flow (if/else, loops, recursion)
              inside the model. Great for debugging and research.</li>
              <li><em>Static (TF v1)</em>: the graph is defined once, then compiled and executed repeatedly.
              Allows more optimization but makes debugging painful and dynamic architectures awkward.</li>
              <li>TensorFlow 2 adopted eager execution (dynamic by default), and PyTorch added <code>torch.compile()</code>
              for static graph optimizations — the approaches converged.</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>PyTorch Autograd tutorial</strong> — Official guide to understanding the autograd engine.</li>
          <li><strong>Paszke et al. (2017) &quot;Automatic differentiation in PyTorch&quot;</strong> — The original PyTorch autograd paper.</li>
          <li><strong>Baydin et al. (2018) &quot;Automatic Differentiation in Machine Learning: a Survey&quot;</strong> — Forward-mode vs. reverse-mode AD explained thoroughly.</li>
          <li><strong>Karpathy &quot;micrograd&quot;</strong> — A tiny autograd engine in ~100 lines of Python. The best way to truly understand autograd.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
