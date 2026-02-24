"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";
import GradientDescent3D from "@/components/viz/GradientDescent3D";

export default function CalculusAndOptimization() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Calculus for ML boils down to one idea: <strong>how does the output change when I wiggle the input?</strong>
        </p>
        <p>
          The <strong>derivative</strong> tells you the slope — is the function going up or down, and how steeply?
          The <strong>gradient</strong> is just the derivative for functions with multiple inputs — it&apos;s a vector
          that points in the direction of steepest increase.
        </p>
        <p>
          Training a neural network is an optimization problem: you have a loss function that measures how wrong
          your model is, and you want to find the parameters that make it as small as possible. The gradient tells
          you which direction to step, and <strong>gradient descent</strong> takes small steps downhill until you
          reach a minimum.
        </p>
        <p>
          That&apos;s literally it. Everything else — chain rule, backpropagation, Adam optimizer — is just making
          this basic recipe faster and more reliable.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Partial Derivatives and the Gradient</h3>
        <p>
          For a function <InlineMath math="f(x_1, x_2, \ldots, x_n)" />, the gradient is the vector of all partial derivatives:
        </p>
        <BlockMath math="\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}" />
        <p>
          The gradient always points in the direction of <strong>steepest ascent</strong>. To minimize, go the opposite way.
        </p>

        <h3>The Chain Rule (Powers Backpropagation)</h3>
        <p>
          If <InlineMath math="y = f(g(x))" />, then:
        </p>
        <BlockMath math="\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}" />
        <p>
          For neural networks with layers <InlineMath math="z_1 \to z_2 \to \cdots \to L" />:
        </p>
        <BlockMath math="\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial z_n} \cdot \frac{\partial z_n}{\partial z_{n-1}} \cdots \frac{\partial z_2}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}" />

        <h3>Gradient Descent</h3>
        <p>Update rule:</p>
        <BlockMath math="\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)" />
        <p>
          where <InlineMath math="\eta" /> is the <strong>learning rate</strong>. Too large → overshoots. Too small → converges slowly.
        </p>

        <h3>The Jacobian and Hessian</h3>
        <p>
          The <strong>Jacobian</strong> <InlineMath math="J \in \mathbb{R}^{m \times n}" /> generalizes the gradient for vector-valued functions. The <strong>Hessian</strong> <InlineMath math="H \in \mathbb{R}^{n \times n}" /> is the matrix of second derivatives — its eigenvalues tell you about the curvature of the loss surface.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Gradient Descent from Scratch</h3>
        <CodeBlock
          language="python"
          title="gradient_descent.py"
          code={`import numpy as np

# Minimize f(x, y) = x² + 3y² + 0.5xy - 2x - 3y + 5
def f(x, y):
    return x**2 + 3*y**2 + 0.5*x*y - 2*x - 3*y + 5

def gradient(x, y):
    df_dx = 2*x + 0.5*y - 2
    df_dy = 6*y + 0.5*x - 3
    return np.array([df_dx, df_dy])

# Gradient descent
lr = 0.1
params = np.array([3.5, 3.5])  # starting point
history = [params.copy()]

for step in range(50):
    grad = gradient(params[0], params[1])
    params = params - lr * grad
    history.append(params.copy())
    if np.linalg.norm(grad) < 1e-6:
        print(f"Converged at step {step}")
        break

print(f"Minimum at: ({params[0]:.4f}, {params[1]:.4f})")
print(f"f(x*, y*) = {f(params[0], params[1]):.4f}")
# Minimum at: (0.8936, 0.4255)
# f(x*, y*) = 3.4787`}
        />

        <h3>Gradient Descent with PyTorch Autograd</h3>
        <CodeBlock
          language="python"
          title="autograd_example.py"
          code={`import torch

# PyTorch computes gradients automatically!
x = torch.tensor([3.5], requires_grad=True)
y = torch.tensor([3.5], requires_grad=True)
lr = 0.1

for step in range(50):
    # Forward pass
    loss = x**2 + 3*y**2 + 0.5*x*y - 2*x - 3*y + 5

    # Backward pass (computes gradients)
    loss.backward()

    # Update parameters (no_grad prevents tracking this operation)
    with torch.no_grad():
        x -= lr * x.grad
        y -= lr * y.grad
        x.grad.zero_()
        y.grad.zero_()

print(f"Minimum at: ({x.item():.4f}, {y.item():.4f})")
print(f"Loss: {loss.item():.4f}")`}
        />
      </TopicSection>

      <TopicSection type="see-it">
        <p className="mb-4">
          Watch gradient descent navigate a 2D loss surface. The contour plot shows level curves of the loss function —
          the ball starts at the top-right and rolls downhill toward the minimum (green circle). Try adjusting
          the learning rate to see what happens when it&apos;s too high or too low.
        </p>
        <GradientDescent3D />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>You rarely compute gradients by hand</strong> — PyTorch autograd and JAX handle this. But understanding the chain rule is essential for debugging gradient issues.</li>
          <li><strong>Learning rate is the most important hyperparameter</strong> — try logarithmic ranges: 1e-5, 1e-4, 1e-3, 1e-2, 1e-1.</li>
          <li><strong>Use Adam for most tasks</strong> — it adapts the learning rate per-parameter and handles saddle points better than vanilla SGD.</li>
          <li><strong>Gradient clipping</strong> prevents exploding gradients in RNNs and transformers.</li>
          <li><strong>Learning rate warmup + cosine decay</strong> is the default schedule for transformers.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Forgetting to zero gradients</strong> — PyTorch <em>accumulates</em> gradients by default. Call <code>optimizer.zero_grad()</code> before each backward pass.</li>
          <li><strong>Setting learning rate too high</strong> — loss oscillates or diverges. If loss goes to NaN, reduce lr by 10x.</li>
          <li><strong>Confusing the gradient with the derivative</strong> — gradient is a vector (for multi-variable functions), derivative is a scalar (for single-variable).</li>
          <li><strong>Assuming loss landscapes are convex</strong> — neural network losses have many local minima and saddle points. SGD noise actually helps escape saddle points.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Explain backpropagation in a 2-layer neural network. What is the chain rule doing?</p>
        <p><strong>Solution:</strong></p>
        <p>
          For a network with input <InlineMath math="x" />, hidden layer <InlineMath math="h = \sigma(W_1 x + b_1)" />,
          and output <InlineMath math="\hat{y} = W_2 h + b_2" />, with loss <InlineMath math="L = (\hat{y} - y)^2" />:
        </p>
        <ol>
          <li><strong>Forward pass</strong>: Compute <InlineMath math="h" />, then <InlineMath math="\hat{y}" />, then <InlineMath math="L" /></li>
          <li><strong>Backward pass</strong> (chain rule):
            <ul>
              <li><InlineMath math="\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial \hat{y}} \cdot h^T = 2(\hat{y} - y) \cdot h^T" /></li>
              <li><InlineMath math="\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \hat{y}} \cdot W_2^T \cdot \sigma'(W_1 x + b_1) \cdot x^T" /></li>
            </ul>
          </li>
        </ol>
        <p>
          The chain rule lets us compute the gradient of the loss with respect to <em>any</em> parameter
          by multiplying local derivatives along the computational graph path from loss to that parameter.
        </p>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>3Blue1Brown &quot;Essence of Calculus&quot;</strong> — Visual intuition for derivatives and integrals</li>
          <li><strong>CS231n Backpropagation Notes</strong> — Stanford&apos;s excellent write-up on computational graphs</li>
          <li><strong>Ruder (2016) &quot;An overview of gradient descent optimization algorithms&quot;</strong> — The definitive survey of SGD variants</li>
          <li><strong>PyTorch Autograd tutorial</strong> — Hands-on guide to automatic differentiation</li>
        </ul>
      </TopicSection>
    </div>
  );
}
