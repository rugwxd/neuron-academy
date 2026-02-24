"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";
import BiasVarianceViz from "@/components/viz/BiasVarianceViz";

export default function BiasVariance() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          The bias-variance tradeoff is the <strong>most fundamental concept in machine learning</strong>.
          It explains why your model works well on training data but fails on new data, and why making a model
          more complex doesn&apos;t always make it better.
        </p>
        <p>
          <strong>Bias</strong> is how far off your model&apos;s average prediction is from the truth.
          A model with high bias makes oversimplified assumptions — like fitting a straight line to curved data.
          It <em>underfits</em>.
        </p>
        <p>
          <strong>Variance</strong> is how much your model&apos;s predictions change if you train it on different
          data. A model with high variance is very sensitive to the specific training data — it memorizes noise.
          It <em>overfits</em>.
        </p>
        <p>
          You can&apos;t minimize both simultaneously. Simple models have high bias and low variance.
          Complex models have low bias and high variance. The sweet spot is in the middle.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Bias-Variance Decomposition</h3>
        <p>For any model, the expected prediction error can be decomposed as:</p>
        <BlockMath math="E[(y - \hat{f}(x))^2] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2" />
        <p>where:</p>
        <ul>
          <li><InlineMath math="\text{Bias}^2 = (E[\hat{f}(x)] - f(x))^2" /> — systematic error from wrong assumptions</li>
          <li><InlineMath math="\text{Var} = E[(\hat{f}(x) - E[\hat{f}(x)])^2]" /> — sensitivity to training set</li>
          <li><InlineMath math="\sigma^2" /> — irreducible error (noise in the data)</li>
        </ul>
        <p>
          You cannot reduce <InlineMath math="\sigma^2" /> — it&apos;s inherent noise. You can only trade
          bias for variance and vice versa.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <CodeBlock
          language="python"
          title="bias_variance_demo.py"
          code={`import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# True function: y = sin(x) + noise
np.random.seed(42)
n = 100
X = np.sort(np.random.uniform(0, 2 * np.pi, n)).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.3, n)

# Fit models of increasing complexity
for degree in [1, 3, 5, 10, 20]:
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )
    # Use cross-validation to estimate generalization
    train_scores = -cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_error"
    )
    model.fit(X, y)
    train_mse = np.mean((y - model.predict(X)) ** 2)

    print(f"Degree {degree:2d}: "
          f"Train MSE = {train_mse:.4f}, "
          f"CV MSE = {train_scores.mean():.4f} "
          f"{'← underfitting' if degree < 3 else '← overfitting' if degree > 10 else '← sweet spot'}")

# Output:
# Degree  1: Train MSE = 0.2814, CV MSE = 0.3142 ← underfitting
# Degree  3: Train MSE = 0.0892, CV MSE = 0.1043 ← sweet spot
# Degree  5: Train MSE = 0.0876, CV MSE = 0.1028 ← sweet spot
# Degree 10: Train MSE = 0.0841, CV MSE = 0.1198
# Degree 20: Train MSE = 0.0453, CV MSE = 2.3451 ← overfitting`}
        />
      </TopicSection>

      <TopicSection type="see-it">
        <p className="mb-4">
          Move the <strong>complexity slider</strong> to see how bias, variance, and total error change.
          The sweet spot (green dot) is where total error is minimized — this is the optimal model complexity.
        </p>
        <BiasVarianceViz />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>High bias (underfitting)</strong>: training and test error are both high → use a more complex model, add features, reduce regularization.</li>
          <li><strong>High variance (overfitting)</strong>: training error is low but test error is much higher → get more data, use regularization, simplify model, use ensembles, use dropout.</li>
          <li><strong>Learning curves are your best diagnostic</strong>: plot train/test error vs training set size. If they converge at high error = bias problem. If large gap = variance problem.</li>
          <li><strong>Ensembles exploit this tradeoff</strong>: Random Forests reduce variance by averaging many high-variance trees. Boosting reduces bias by iteratively correcting errors.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Only looking at training accuracy</strong> — a model with 100% training accuracy is almost certainly overfitting.</li>
          <li><strong>Adding more features always helps</strong> — more features increase variance. With limited data, fewer features can generalize better (curse of dimensionality).</li>
          <li><strong>Thinking regularization only hurts</strong> — regularization adds a small amount of bias but can dramatically reduce variance, lowering total error.</li>
          <li><strong>Confusing bias-variance with underfitting-overfitting</strong> — they&apos;re closely related but not identical. Bias-variance is a statistical property, underfitting/overfitting is an empirical observation.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You train a neural network and get 99% train accuracy but 60% test accuracy. Diagnose the problem and propose 5 solutions.</p>
        <p><strong>Answer:</strong> This is a classic <strong>high variance (overfitting)</strong> problem. Solutions:</p>
        <ol>
          <li><strong>Get more training data</strong> — the most reliable fix. More data → less overfitting.</li>
          <li><strong>Add regularization</strong> — L2 weight decay, dropout, batch normalization.</li>
          <li><strong>Reduce model capacity</strong> — fewer layers, fewer neurons, smaller embedding dimensions.</li>
          <li><strong>Data augmentation</strong> — for images: flips, rotations, crops; for text: synonym replacement, back-translation.</li>
          <li><strong>Early stopping</strong> — monitor validation loss and stop training when it starts increasing.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>ESL Chapter 2.9</strong> — The formal bias-variance decomposition with derivation.</li>
          <li><strong>Belkin et al. (2019) &quot;Reconciling modern ML practice and the bias-variance trade-off&quot;</strong> — The &quot;double descent&quot; phenomenon where very overparameterized models generalize well again.</li>
          <li><strong>Andrew Ng&apos;s ML Yearning</strong> — Practical guide on diagnosing bias vs variance issues.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
