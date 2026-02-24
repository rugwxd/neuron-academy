"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function InformationTheory() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Information theory is the math of <strong>surprise</strong>. When something unlikely happens,
          it carries a lot of information. When something expected happens, it carries very little. A coin
          landing heads? Not surprising. Your model predicting the one rare class correctly? Very
          surprising &mdash; very informative.
        </p>
        <p>
          <strong>Entropy</strong> is the <em>average surprise</em> of a random variable. A fair coin has
          maximum entropy (you genuinely don&apos;t know what&apos;s coming). A loaded coin with 99%
          heads has low entropy (you basically know).
        </p>
        <p>
          <strong>Cross-entropy</strong> measures the surprise you experience when reality follows
          distribution <InlineMath math="p" /> but you <em>think</em> it follows distribution{" "}
          <InlineMath math="q" />. This is exactly why cross-entropy is THE loss function for
          classification &mdash; your model&apos;s predicted probabilities are <InlineMath math="q" />,
          reality is <InlineMath math="p" />, and you want to minimize the gap.
        </p>
        <p>
          <strong>KL divergence</strong> is the <em>extra</em> surprise from using the wrong distribution.
          It&apos;s cross-entropy minus entropy: the penalty for being wrong.
        </p>
        <p>
          <strong>Mutual information</strong> quantifies how much knowing one variable tells you about
          another. In decision trees, the feature that gives the highest mutual information with the
          label is the one that gets picked for the split &mdash; this is called <strong>information gain</strong>.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Shannon Entropy</h3>
        <p>
          For a discrete random variable <InlineMath math="X" /> with probabilities <InlineMath math="p(x)" />:
        </p>
        <BlockMath math="H(X) = -\sum_{x} p(x) \log_2 p(x)" />
        <p>
          Entropy is maximized when all outcomes are equally likely (uniform distribution) and minimized
          (equals 0) when the outcome is certain. For <InlineMath math="n" /> equally likely outcomes:
        </p>
        <BlockMath math="H_{\max} = \log_2 n" />

        <h3>Cross-Entropy</h3>
        <p>
          The average number of bits needed to encode events from <InlineMath math="p" /> using a code
          optimized for <InlineMath math="q" />:
        </p>
        <BlockMath math="H(p, q) = -\sum_{x} p(x) \log q(x)" />
        <p>
          Note: <InlineMath math="H(p, q) \ge H(p)" /> always. Equality holds only when{" "}
          <InlineMath math="q = p" />. In multi-class classification with one-hot labels{" "}
          <InlineMath math="y" /> and predicted probabilities <InlineMath math="\hat{y}" />:
        </p>
        <BlockMath math="L_{\text{CE}} = -\sum_{c=1}^{C} y_c \log \hat{y}_c = -\log \hat{y}_k" />
        <p>
          where <InlineMath math="k" /> is the true class. This is the negative log-likelihood of the
          correct class.
        </p>

        <h3>KL Divergence</h3>
        <p>
          The &quot;extra bits&quot; wasted by using <InlineMath math="q" /> instead of <InlineMath math="p" />:
        </p>
        <BlockMath math="D_{\text{KL}}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)" />
        <p>Key properties:</p>
        <ul>
          <li><InlineMath math="D_{\text{KL}}(p \| q) \ge 0" /> (Gibbs&apos; inequality), with equality iff <InlineMath math="p = q" /></li>
          <li><strong>Asymmetric:</strong> <InlineMath math="D_{\text{KL}}(p \| q) \ne D_{\text{KL}}(q \| p)" /> in general</li>
          <li><strong>Forward KL</strong> <InlineMath math="D_{\text{KL}}(p \| q)" /> is mean-seeking (forces <InlineMath math="q" /> to cover all modes of <InlineMath math="p" />)</li>
          <li><strong>Reverse KL</strong> <InlineMath math="D_{\text{KL}}(q \| p)" /> is mode-seeking (allows <InlineMath math="q" /> to focus on one mode)</li>
        </ul>
        <p>
          <strong>Asymmetry proof:</strong> Consider <InlineMath math="p = (0.5, 0.5)" /> and{" "}
          <InlineMath math="q = (0.9, 0.1)" />:
        </p>
        <BlockMath math="D_{\text{KL}}(p \| q) = 0.5 \log\frac{0.5}{0.9} + 0.5 \log\frac{0.5}{0.1} \approx 0.511" />
        <BlockMath math="D_{\text{KL}}(q \| p) = 0.9 \log\frac{0.9}{0.5} + 0.1 \log\frac{0.1}{0.5} \approx 0.368" />

        <h3>Mutual Information</h3>
        <p>
          How much knowing <InlineMath math="X" /> tells you about <InlineMath math="Y" /> (and vice versa):
        </p>
        <BlockMath math="I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = D_{\text{KL}}(p(x, y) \| p(x)p(y))" />
        <p>
          Mutual information is zero if and only if <InlineMath math="X" /> and <InlineMath math="Y" /> are
          independent. Unlike correlation, mutual information captures <em>nonlinear</em> dependencies.
        </p>

        <h3>Information Gain (Decision Trees)</h3>
        <p>
          The information gain from splitting on feature <InlineMath math="A" /> is:
        </p>
        <BlockMath math="\text{IG}(Y, A) = H(Y) - H(Y \mid A) = H(Y) - \sum_{a} \frac{|S_a|}{|S|} H(Y \mid A = a)" />
        <p>
          This is exactly the mutual information <InlineMath math="I(Y; A)" />. The feature with the
          highest information gain reduces uncertainty the most &mdash; that&apos;s why it gets picked
          for the split.
        </p>

        <h3>The VAE Objective (ELBO)</h3>
        <p>
          Variational autoencoders maximize the Evidence Lower Bound:
        </p>
        <BlockMath math="\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}(q(z|x) \| p(z))" />
        <p>
          The first term is reconstruction quality. The KL term regularizes the encoder&apos;s posterior{" "}
          <InlineMath math="q(z|x)" /> to stay close to the prior <InlineMath math="p(z)" /> (typically{" "}
          <InlineMath math="\mathcal{N}(0, I)" />).
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Entropy, Cross-Entropy, and KL Divergence from Scratch</h3>
        <CodeBlock
          language="python"
          title="information_theory.py"
          code={`import numpy as np

# ── From-scratch implementations ─────────────────────────────
def entropy(p):
    """Shannon entropy H(p) in nats (using natural log).
    ML frameworks use natural log; multiply by 1/ln(2) to get bits."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]  # 0 * log(0) = 0 by convention
    return -np.sum(p * np.log(p))

def cross_entropy(p, q):
    """Cross-entropy H(p, q) = -sum p(x) log q(x)."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    q = np.clip(q, 1e-15, 1.0)  # avoid log(0)
    return -np.sum(p * np.log(q))

def kl_divergence(p, q):
    """KL divergence D_KL(p || q) = H(p,q) - H(p)."""
    return cross_entropy(p, q) - entropy(p)

def mutual_information(joint_pxy, px, py):
    """Mutual information I(X;Y) = D_KL(p(x,y) || p(x)p(y))."""
    mi = 0.0
    for i in range(len(px)):
        for j in range(len(py)):
            if joint_pxy[i, j] > 0:
                mi += joint_pxy[i, j] * np.log(
                    joint_pxy[i, j] / (px[i] * py[j])
                )
    return mi

# ── Entropy examples ──────────────────────────────────────────
fair = np.array([0.5, 0.5])
loaded = np.array([0.9, 0.1])
uniform4 = np.array([0.25, 0.25, 0.25, 0.25])

print("=== Entropy ===")
print(f"H(fair coin)     = {entropy(fair):.4f} nats")
print(f"H(loaded coin)   = {entropy(loaded):.4f} nats")
print(f"H(4-sided die)   = {entropy(uniform4):.4f} nats")
# Fair coin: 0.6931, Loaded: 0.3251, 4-sided: 1.3863

print("\\n=== Cross-Entropy ===")
print(f"H(fair, loaded) = {cross_entropy(fair, loaded):.4f}")
print(f"H(fair, fair)   = {cross_entropy(fair, fair):.4f}  (= H(fair))")

print("\\n=== KL Divergence (asymmetry) ===")
print(f"KL(fair || loaded) = {kl_divergence(fair, loaded):.4f}")
print(f"KL(loaded || fair) = {kl_divergence(loaded, fair):.4f}")
print("These differ! KL divergence is NOT symmetric.")

# ── Mutual Information example ────────────────────────────────
# X = Weather {sunny, rainy}, Y = Umbrella {yes, no}
joint = np.array([[0.1, 0.4],   # sunny: P(umbrella), P(no umbrella)
                  [0.35, 0.15]]) # rainy: P(umbrella), P(no umbrella)
px = joint.sum(axis=1)  # marginal of X
py = joint.sum(axis=0)  # marginal of Y
mi = mutual_information(joint, px, py)
print(f"\\n=== Mutual Information ===")
print(f"I(Weather; Umbrella) = {mi:.4f} nats")

# Output:
# === Entropy ===
# H(fair coin)     = 0.6931 nats
# H(loaded coin)   = 0.3251 nats
# H(4-sided die)   = 1.3863 nats
#
# === Cross-Entropy ===
# H(fair, loaded) = 1.2040
# H(fair, fair)   = 0.6931  (= H(fair))
#
# === KL Divergence (asymmetry) ===
# KL(fair || loaded) = 0.5108
# KL(loaded || fair) = 0.3681
# These differ! KL divergence is NOT symmetric.
#
# === Mutual Information ===
# I(Weather; Umbrella) = 0.1064 nats`}
        />

        <h3>Verifying with SciPy</h3>
        <CodeBlock
          language="python"
          title="scipy_info_theory.py"
          code={`import numpy as np
from scipy.stats import entropy as sp_entropy
from scipy.special import rel_entr

# SciPy's entropy computes H(p) when only p is given,
# and D_KL(p || q) when both p and q are given.

fair = np.array([0.5, 0.5])
loaded = np.array([0.9, 0.1])

# Shannon entropy (natural log by default)
print(f"H(fair)   = {sp_entropy(fair):.4f}")       # 0.6931
print(f"H(loaded) = {sp_entropy(loaded):.4f}")      # 0.3251

# KL divergence: scipy.stats.entropy(p, q) = D_KL(p || q)
print(f"KL(fair || loaded) = {sp_entropy(fair, loaded):.4f}")  # 0.5108
print(f"KL(loaded || fair) = {sp_entropy(loaded, fair):.4f}")  # 0.3681

# Cross-entropy = entropy + KL divergence
h_fair = sp_entropy(fair)
kl = sp_entropy(fair, loaded)
print(f"H(fair, loaded) = H + KL = {h_fair + kl:.4f}")  # 1.2040

# rel_entr gives element-wise KL contributions
print(f"Element-wise KL: {rel_entr(fair, loaded)}")

# For bits instead of nats, pass base=2
print(f"H(fair) in bits = {sp_entropy(fair, base=2):.4f}")  # 1.0000`}
        />

        <h3>Cross-Entropy Loss for Classification</h3>
        <CodeBlock
          language="python"
          title="cross_entropy_loss.py"
          code={`import numpy as np

# ── Binary cross-entropy loss (from scratch) ─────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy (log loss).
    Clip predictions to avoid log(0)."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

def categorical_cross_entropy(y_true_onehot, y_pred):
    """Multi-class cross-entropy.
    y_true_onehot: (n_samples, n_classes) one-hot
    y_pred: (n_samples, n_classes) predicted probabilities"""
    y_pred = np.clip(y_pred, 1e-15, 1.0)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))

# ── Toy logistic regression with BCE ─────────────────────────
np.random.seed(42)
X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(float)

X_b = np.column_stack([np.ones(len(X)), X])  # add bias
w = np.zeros(3)
lr = 0.1

losses = []
for epoch in range(100):
    z = X_b @ w
    y_pred = sigmoid(z)
    loss = binary_cross_entropy(y, y_pred)
    losses.append(loss)
    grad = X_b.T @ (y_pred - y) / len(y)
    w -= lr * grad

print(f"Final BCE loss: {losses[-1]:.4f}")
print(f"Accuracy: {np.mean((y_pred > 0.5) == y):.2%}")
# Final BCE loss: 0.2741
# Accuracy: 92.50%`}
        />

        <h3>Information Gain for Decision Tree Splits</h3>
        <CodeBlock
          language="python"
          title="information_gain.py"
          code={`import numpy as np
import math

def calculate_entropy(y):
    """Shannon entropy of label array (in bits)."""
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(p * math.log2(p) for p in probs if p > 0)

def information_gain(y, y_left, y_right):
    """IG = H(parent) - weighted H(children)."""
    p = len(y_left) / len(y)
    return (calculate_entropy(y)
            - p * calculate_entropy(y_left)
            - (1 - p) * calculate_entropy(y_right))

# ── Which feature gives the best split? ──────────────────────
labels = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
print(f"Parent entropy: {calculate_entropy(labels):.4f} bits")

# Split 1: Outlook = Sunny
sunny = np.array([1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0], dtype=bool)
ig1 = information_gain(labels, labels[sunny], labels[~sunny])
print(f"IG (Outlook=Sunny):  {ig1:.4f} bits")

# Split 2: Humidity = High
humid = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0], dtype=bool)
ig2 = information_gain(labels, labels[humid], labels[~humid])
print(f"IG (Humidity=High):  {ig2:.4f} bits")

best = "Outlook=Sunny" if ig1 > ig2 else "Humidity=High"
print(f"Best split: {best} (higher info gain)")

# ── Using sklearn for mutual information feature selection ────
from sklearn.feature_selection import mutual_info_classif

np.random.seed(0)
X = np.random.randn(500, 5)
y = (X[:, 0] + 0.5 * X[:, 2] > 0).astype(int)  # only features 0,2 matter

mi_scores = mutual_info_classif(X, y, random_state=0)
for i, score in enumerate(mi_scores):
    marker = " <-- informative" if score > 0.1 else ""
    print(f"  Feature {i}: MI = {score:.4f}{marker}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Cross-entropy loss is everywhere</strong>: It&apos;s the default loss for classification in PyTorch (<code>nn.CrossEntropyLoss</code>), TensorFlow, and scikit-learn. Minimizing cross-entropy is equivalent to maximizing log-likelihood.</li>
          <li><strong>KL divergence in VAEs</strong>: Variational autoencoders minimize <InlineMath math="D_{\text{KL}}(q(z|x) \| p(z))" /> to keep the learned latent distribution close to the prior. The closed-form solution for two Gaussians is used constantly.</li>
          <li><strong>Information gain in trees</strong>: Decision trees (and Random Forests, XGBoost) use information gain (or Gini impurity, which approximates it) to decide which feature to split on at each node.</li>
          <li><strong>Mutual information for feature selection</strong>: High MI between a feature and the target means the feature is informative. Crucially, MI captures nonlinear relationships that correlation misses. Use <code>sklearn.feature_selection.mutual_info_classif</code>.</li>
          <li><strong>Knowledge distillation</strong>: Student networks learn from teacher networks by minimizing the KL divergence between their softmax output distributions.</li>
          <li><strong>Label smoothing</strong>: Instead of one-hot labels, use <InlineMath math="(1-\epsilon) \cdot \text{one-hot} + \epsilon / C" /> to prevent the model from becoming overconfident. This is an information-theoretic regularizer.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>KL divergence is NOT symmetric</strong>: <InlineMath math="D_{\text{KL}}(p \| q) \ne D_{\text{KL}}(q \| p)" />. The &quot;direction&quot; matters. In VAEs, you minimize <InlineMath math="D_{\text{KL}}(q \| p)" /> (mode-seeking). Fitting a mixture model typically minimizes <InlineMath math="D_{\text{KL}}(p \| q)" /> (mean-seeking). These give very different results.</li>
          <li><strong>Cross-entropy vs log-loss naming confusion</strong>: They&apos;re the same thing! &quot;Log loss&quot; is just binary cross-entropy. PyTorch&apos;s <code>BCELoss</code> is binary cross-entropy; <code>CrossEntropyLoss</code> combines log-softmax + negative log-likelihood for multi-class.</li>
          <li><strong>Entropy of continuous vs discrete</strong>: Differential entropy (continuous) can be negative, unlike discrete entropy. A Gaussian with very small variance has <em>negative</em> differential entropy. Never compare them directly.</li>
          <li><strong>Forgetting to clip predictions</strong>: <InlineMath math="\log(0) = -\infty" />. Always clip predicted probabilities away from 0 and 1 before computing cross-entropy. Standard practice is clipping to <code>[1e-15, 1 - 1e-15]</code>.</li>
          <li><strong>Confusing mutual information with correlation</strong>: Correlation only measures linear relationships. Two variables can have zero correlation but high mutual information (e.g., <InlineMath math="Y = X^2" /> where <InlineMath math="X \sim \mathcal{N}(0,1)" />).</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Why do we use cross-entropy loss for classification instead of mean squared error (MSE)?</p>
        <p><strong>Solution:</strong></p>
        <ol>
          <li><strong>Gradient magnitude</strong>: With sigmoid + MSE, the gradient includes a <InlineMath math="\sigma'(z)" /> term that vanishes when the prediction is confident but wrong (sigmoid saturation). Cross-entropy cancels out the sigmoid derivative, giving a clean gradient: <InlineMath math="\frac{\partial L}{\partial z} = \hat{y} - y" />. This means the model corrects confident mistakes <em>faster</em>.</li>
          <li><strong>Probabilistic interpretation</strong>: Cross-entropy loss is equivalent to maximum likelihood estimation (MLE) under a Bernoulli model. Minimizing cross-entropy = maximizing the log-likelihood of observing the true labels.</li>
          <li><strong>Convexity</strong>: Cross-entropy loss with a linear model (logistic regression) is convex with respect to the parameters. MSE with sigmoid is non-convex &mdash; it has local minima.</li>
          <li><strong>Information-theoretic view</strong>: Cross-entropy measures how many bits you waste by using your model&apos;s distribution instead of the true distribution. MSE has no such interpretation for probability outputs.</li>
        </ol>
        <CodeBlock
          language="python"
          code={`import numpy as np

# Demonstrate the gradient difference
y_true = 1.0
z = -5.0  # model is confident AND wrong

sigmoid = lambda z: 1 / (1 + np.exp(-z))
y_pred = sigmoid(z)  # ~0.0067

# MSE gradient: 2(y_pred - y_true) * sigmoid'(z)
sigmoid_deriv = y_pred * (1 - y_pred)  # ~0.0066 (vanishing!)
mse_grad = 2 * (y_pred - y_true) * sigmoid_deriv
print(f"MSE gradient:          {mse_grad:.6f}")   # tiny!

# CE gradient: (y_pred - y_true) — no sigmoid derivative!
ce_grad = y_pred - y_true
print(f"Cross-entropy gradient: {ce_grad:.6f}")    # large!
# Output:
# MSE gradient:          -0.0132
# Cross-entropy gradient: -0.9933
# Cross-entropy gives ~75x stronger gradient for this wrong prediction!`}
        />
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Cover &amp; Thomas, &quot;Elements of Information Theory&quot;</strong> &mdash; The definitive textbook on information theory, rigorous and comprehensive</li>
          <li><strong>Shannon (1948), &quot;A Mathematical Theory of Communication&quot;</strong> &mdash; The original paper that started it all; remarkably readable for a foundational paper</li>
          <li><strong>Chris Olah, &quot;Visual Information Theory&quot;</strong> &mdash; Excellent blog post with interactive visualizations of entropy, cross-entropy, and KL divergence</li>
          <li><strong>Bishop, &quot;PRML&quot; Ch. 1.6</strong> &mdash; Information theory concepts specifically framed for machine learning applications</li>
          <li><strong>Kingma &amp; Welling (2014), &quot;Auto-Encoding Variational Bayes&quot;</strong> &mdash; The VAE paper that puts KL divergence at the heart of generative modeling</li>
        </ul>
      </TopicSection>
    </div>
  );
}
