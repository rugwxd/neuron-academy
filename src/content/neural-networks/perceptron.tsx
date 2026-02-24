"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function Perceptron() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          The perceptron is where neural networks began. Invented by Frank Rosenblatt in 1958, it&apos;s
          the simplest possible neural network: a single neuron that takes inputs, multiplies each by
          a weight, sums them up, and passes the result through a step function. If the sum exceeds a
          threshold, it fires (outputs 1); otherwise it doesn&apos;t (outputs 0).
        </p>
        <p>
          Think of it as a <strong>linear classifier</strong> that draws a straight line (or hyperplane)
          to separate two classes. &quot;Is this email spam?&quot; — the perceptron checks a weighted
          combination of features, and if the result crosses the threshold, it says &quot;yes.&quot;
        </p>
        <p>
          The perceptron learning algorithm is beautifully simple: show it an example, check if it gets
          the answer right, and if not, nudge the weights in the right direction. Rosenblatt proved that
          if the data is <strong>linearly separable</strong>, the algorithm is guaranteed to converge.
        </p>
        <p>
          But here&apos;s the catch — and it nearly killed the entire field of neural networks. In 1969,
          Minsky and Papert published <em>Perceptrons</em>, proving that a single perceptron <strong>cannot
          learn XOR</strong> (exclusive or). XOR is not linearly separable — you can&apos;t draw one straight
          line to separate the classes. This triggered the first &quot;AI winter.&quot; The solution? Stack
          multiple perceptrons into layers — the <strong>multi-layer perceptron (MLP)</strong> — and use
          backpropagation to train them. That&apos;s the foundation of modern deep learning.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Single Perceptron</h3>
        <p>Given input vector <InlineMath math="\mathbf{x} = [x_1, x_2, \ldots, x_d]" /> and weights <InlineMath math="\mathbf{w}" />:</p>
        <BlockMath math="z = \mathbf{w}^T\mathbf{x} + b = \sum_{i=1}^{d} w_i x_i + b" />
        <BlockMath math="\hat{y} = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}" />

        <h3>Perceptron Learning Rule</h3>
        <p>For each misclassified sample <InlineMath math="(x_i, y_i)" />:</p>
        <BlockMath math="\mathbf{w} \leftarrow \mathbf{w} + \eta (y_i - \hat{y}_i) \mathbf{x}_i" />
        <BlockMath math="b \leftarrow b + \eta (y_i - \hat{y}_i)" />
        <p>
          If <InlineMath math="y_i = 1" /> but <InlineMath math="\hat{y}_i = 0" /> (false negative),
          we add <InlineMath math="\eta \mathbf{x}_i" /> to the weights — pushing the decision boundary
          to include this point. Vice versa for false positives.
        </p>

        <h3>Why XOR Fails</h3>
        <p>XOR truth table:</p>
        <BlockMath math="\begin{array}{cc|c} x_1 & x_2 & y \\ \hline 0 & 0 & 0 \\ 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{array}" />
        <p>
          A single perceptron defines a linear boundary <InlineMath math="w_1 x_1 + w_2 x_2 + b = 0" />.
          No single line can separate <InlineMath math="\{(0,1), (1,0)\}" /> from <InlineMath math="\{(0,0), (1,1)\}" />.
          This is provably impossible because XOR is not linearly separable.
        </p>

        <h3>MLP Solution to XOR</h3>
        <p>
          A 2-layer MLP with one hidden layer of 2 neurons can solve XOR. The hidden layer creates a
          <strong>nonlinear feature space</strong> where the problem becomes linearly separable:
        </p>
        <BlockMath math="h_1 = \sigma(x_1 + x_2 - 0.5) \quad \text{(OR gate)}" />
        <BlockMath math="h_2 = \sigma(-x_1 - x_2 + 1.5) \quad \text{(NAND gate)}" />
        <BlockMath math="y = \sigma(h_1 + h_2 - 1.5) \quad \text{(AND gate)}" />
        <p>
          The hidden layer transforms the 2D input into a new 2D space where a linear boundary works.
          This is the core insight: <strong>depth enables nonlinearity</strong>.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Perceptron from Scratch</h3>
        <CodeBlock
          language="python"
          title="perceptron_scratch.py"
          code={`import numpy as np

class Perceptron:
    def __init__(self, lr=0.1, n_epochs=100):
        self.lr = lr
        self.n_epochs = n_epochs

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.errors = []

        for epoch in range(self.n_epochs):
            n_errors = 0
            for xi, yi in zip(X, y):
                y_hat = self.predict_one(xi)
                error = yi - y_hat

                if error != 0:
                    self.w += self.lr * error * xi
                    self.b += self.lr * error
                    n_errors += 1

            self.errors.append(n_errors)
            if n_errors == 0:
                print(f"Converged at epoch {epoch}")
                break

        return self

    def predict_one(self, x):
        return 1 if (np.dot(self.w, x) + self.b) >= 0 else 0

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

# AND gate — linearly separable, perceptron can learn it
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

p = Perceptron(lr=0.1, n_epochs=20)
p.fit(X, y_and)
print("AND predictions:", p.predict(X))  # [0, 0, 0, 1]

# XOR — NOT linearly separable, perceptron FAILS
y_xor = np.array([0, 1, 1, 0])
p2 = Perceptron(lr=0.1, n_epochs=100)
p2.fit(X, y_xor)
print("XOR predictions:", p2.predict(X))  # Will NOT be [0, 1, 1, 0]`}
        />

        <h3>Solving XOR with a Multi-Layer Perceptron in PyTorch</h3>
        <CodeBlock
          language="python"
          title="xor_mlp_pytorch.py"
          code={`import torch
import torch.nn as nn

# XOR data
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 2-layer MLP: 2 -> 4 -> 1
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid(),
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# Train
for epoch in range(2000):
    y_hat = model(X)
    loss = criterion(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        preds = (y_hat > 0.5).float()
        acc = (preds == y).float().mean()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc.item():.0%}")

# Final predictions
with torch.no_grad():
    print("\\nXOR predictions:")
    for xi, yi in zip(X, model(X)):
        print(f"  {xi.tolist()} -> {yi.item():.4f}")

# Inspect hidden layer — the learned feature space
with torch.no_grad():
    hidden = torch.relu(model[0](X))
    print("\\nHidden representations:")
    for xi, hi in zip(X, hidden):
        print(f"  {xi.tolist()} -> {hi.tolist()}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>Nobody uses raw perceptrons anymore</strong>: they&apos;re historically important
            but strictly inferior to logistic regression (which is a perceptron with a sigmoid and
            proper loss function). Use them for building intuition, not production.
          </li>
          <li>
            <strong>The XOR lesson generalizes</strong>: any nonlinear decision boundary requires at
            least one hidden layer. Most real-world problems are nonlinear, which is why deep networks
            (many layers) are so powerful.
          </li>
          <li>
            <strong>Universal Approximation Theorem</strong>: a single hidden layer with enough neurons
            can approximate <em>any</em> continuous function. But in practice, depth (more layers with
            fewer neurons) is more parameter-efficient than width (one wide layer).
          </li>
          <li>
            <strong>Modern &quot;perceptrons&quot; use smooth activations</strong>: ReLU, GELU, or Swish
            instead of the step function. This makes the function differentiable, which is required
            for gradient-based training (backpropagation).
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Thinking the perceptron is useless</strong>: while you won&apos;t use it directly,
            every neuron in a modern neural network <em>is</em> a perceptron with a smooth activation
            function. Understanding it is understanding the atom of deep learning.
          </li>
          <li>
            <strong>Confusing the perceptron with logistic regression</strong>: a perceptron uses a
            step function and the perceptron learning rule. Logistic regression uses a sigmoid and
            gradient descent on cross-entropy loss. Logistic regression is strictly better.
          </li>
          <li>
            <strong>Assuming linear separability</strong>: most real datasets are not linearly separable.
            Always consider whether your problem needs nonlinear features or a multi-layer architecture.
          </li>
          <li>
            <strong>Forgetting the bias term</strong>: without a bias, the decision boundary must pass
            through the origin. Always include a bias (or equivalently, add a constant feature of 1).
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> Why can&apos;t a single perceptron learn XOR? How would you solve
          it with a neural network? What is the minimum architecture required?
        </p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Why it fails</strong>: a single perceptron computes <InlineMath math="w_1 x_1 + w_2 x_2 + b" /> and
            thresholds the result. This defines a single line (hyperplane) in the input space. XOR
            requires separating <InlineMath math="(0,0), (1,1)" /> from <InlineMath math="(0,1), (1,0)" />,
            which are interleaved — no single line can do this.
          </li>
          <li>
            <strong>Minimum architecture</strong>: a 2-input, 2-hidden-neuron, 1-output network.
            The two hidden neurons create a 2D feature space where XOR becomes linearly separable.
            This was demonstrated by solving XOR as AND(OR(x1, x2), NAND(x1, x2)).
          </li>
          <li>
            <strong>Training</strong>: use backpropagation with a differentiable activation function
            (sigmoid, ReLU) instead of the step function. The step function has zero gradient everywhere
            except at 0, making gradient-based learning impossible.
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Rosenblatt (1958) &quot;The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain&quot;</strong> — The original paper.</li>
          <li><strong>Minsky &amp; Papert (1969) <em>Perceptrons</em></strong> — The book that triggered the first AI winter by proving the XOR limitation.</li>
          <li><strong>Rumelhart, Hinton &amp; Williams (1986) &quot;Learning representations by back-propagating errors&quot;</strong> — The paper that showed how to train multi-layer networks, reviving neural networks.</li>
          <li><strong>3Blue1Brown &quot;But what is a neural network?&quot;</strong> — Excellent visual introduction starting from the perceptron.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
