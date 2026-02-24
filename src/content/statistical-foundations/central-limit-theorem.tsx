"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";
import CLTViz from "@/components/viz/CLTViz";

export default function CentralLimitTheorem() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          The Central Limit Theorem (CLT) is probably the <strong>most important theorem in all of statistics</strong>.
          It says something remarkable:
        </p>
        <p>
          <strong>If you take many samples from ANY distribution and compute the mean of each sample,
          the distribution of those means will be approximately normal (bell-shaped) — regardless of what
          the original distribution looks like.</strong>
        </p>
        <p>
          This is wild. You could be sampling from a uniform distribution (flat), an exponential distribution
          (skewed), or even a bimodal distribution (two peaks) — as long as your sample size is large enough,
          the sample means will form a bell curve.
        </p>
        <p>
          This is why the normal distribution is everywhere in statistics. It&apos;s not because data is inherently
          normal — it&apos;s because <em>averages</em> are normal, and most of what we measure and estimate are
          some form of average.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Formal Statement</h3>
        <p>
          Let <InlineMath math="X_1, X_2, \ldots, X_n" /> be i.i.d. random variables with
          mean <InlineMath math="\mu" /> and finite variance <InlineMath math="\sigma^2" />. Then as <InlineMath math="n \to \infty" />:
        </p>
        <BlockMath math="\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)" />
        <p>Equivalently, the sample mean is approximately normally distributed:</p>
        <BlockMath math="\bar{X}_n \approx \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)" />

        <h3>Key implications</h3>
        <ul>
          <li>The mean of the sampling distribution equals the population mean: <InlineMath math="E[\bar{X}] = \mu" /></li>
          <li>The standard deviation of the sampling distribution (standard error) decreases with <InlineMath math="\sqrt{n}" />: <InlineMath math="SE = \frac{\sigma}{\sqrt{n}}" /></li>
          <li>The approximation gets better as <InlineMath math="n" /> increases. Rule of thumb: <InlineMath math="n \geq 30" /> is usually sufficient for moderately skewed distributions.</li>
        </ul>
      </TopicSection>

      <TopicSection type="code">
        <CodeBlock
          language="python"
          title="clt_demonstration.py"
          code={`import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def demonstrate_clt(distribution, n_samples=10000, sample_size=30):
    """Show CLT in action for any distribution."""
    # Draw many samples and compute means
    means = []
    for _ in range(n_samples):
        sample = distribution(sample_size)
        means.append(np.mean(sample))
    means = np.array(means)

    print(f"Mean of means: {means.mean():.4f}")
    print(f"Std of means:  {means.std():.4f}")
    print(f"Theoretical SE: {distribution(10000).std() / np.sqrt(sample_size):.4f}")
    return means

# 1. Uniform distribution (flat — definitely not normal!)
print("=== Uniform Distribution ===")
uniform_means = demonstrate_clt(lambda n: np.random.uniform(0, 10, n))

# 2. Exponential distribution (heavily right-skewed)
print("\\n=== Exponential Distribution ===")
exp_means = demonstrate_clt(lambda n: np.random.exponential(2, n))

# 3. Bimodal distribution (two peaks — very non-normal)
print("\\n=== Bimodal Distribution ===")
def bimodal(n):
    mask = np.random.random(n) < 0.5
    return np.where(mask, np.random.normal(3, 0.8, n), np.random.normal(7, 0.8, n))
bimodal_means = demonstrate_clt(bimodal)

# All three produce bell-shaped distributions of means!`}
        />

        <CodeBlock
          language="python"
          title="clt_sample_size_effect.py"
          code={`# Show how sample size affects the normal approximation
for n in [2, 5, 10, 30, 100]:
    means = [np.mean(np.random.exponential(2, n)) for _ in range(10000)]
    from scipy.stats import shapiro
    _, p_value = shapiro(np.random.choice(means, 500))  # Shapiro-Wilk normality test
    print(f"n = {n:3d}: std of means = {np.std(means):.4f}, "
          f"normality p-value = {p_value:.4f} "
          f"{'✓ normal' if p_value > 0.05 else '✗ not normal'}")

# n =   2: normality p-value ≈ 0.00  ✗ not normal
# n =   5: normality p-value ≈ 0.01  ✗ not normal
# n =  10: normality p-value ≈ 0.10  ✓ normal
# n =  30: normality p-value ≈ 0.45  ✓ normal
# n = 100: normality p-value ≈ 0.82  ✓ normal`}
        />
      </TopicSection>

      <TopicSection type="see-it">
        <p className="mb-4">
          Choose a source distribution, set the sample size, and draw samples. Watch as the distribution
          of sample means (histogram) converges to a normal distribution (pink curve) — even when the
          source distribution is far from normal. Try different sample sizes to see the CLT in action.
        </p>
        <CLTViz />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Why confidence intervals work</strong>: The CLT guarantees that <InlineMath math="\bar{X}" /> is approximately normal, so we can construct intervals like <InlineMath math="\bar{X} \pm z_{\alpha/2} \cdot SE" />.</li>
          <li><strong>Why t-tests work</strong>: Hypothesis tests about means rely on the sampling distribution being normal.</li>
          <li><strong>A/B testing</strong>: When computing the difference in means between groups, the CLT ensures that difference is approximately normal (even for binary metrics like conversion rate).</li>
          <li><strong>When CLT fails</strong>: Heavy-tailed distributions (Cauchy, Pareto with <InlineMath math="\alpha \leq 2" />) have infinite variance — CLT doesn&apos;t apply. Financial returns can have heavy tails.</li>
          <li><strong>n ≥ 30 is a myth</strong>: For symmetric distributions, n = 10 might suffice. For heavily skewed distributions, you might need n = 100+.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Thinking the data becomes normal</strong>: The CLT says the <em>sample mean</em> is normal, not the data itself. The original data keeps its original distribution.</li>
          <li><strong>Applying CLT to small samples</strong>: For small n, use the t-distribution instead (which has heavier tails and accounts for estimating variance from the sample).</li>
          <li><strong>Ignoring the independence requirement</strong>: CLT requires independent observations. Autocorrelated data (time series) violates this — the effective sample size is smaller.</li>
          <li><strong>Forgetting the finite variance requirement</strong>: Distributions with infinite variance (e.g., Cauchy) don&apos;t satisfy CLT. The sample mean doesn&apos;t converge.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> In an A/B test, the control group has conversion rate 5.2% (n=10,000) and treatment has 5.8% (n=10,000). Is this significant? Use the CLT.</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>By CLT, each conversion rate is approximately normal with <InlineMath math="SE = \sqrt{\frac{p(1-p)}{n}}" /></li>
          <li>The difference in proportions: <InlineMath math="\hat{p}_T - \hat{p}_C = 0.058 - 0.052 = 0.006" /></li>
          <li>Under H₀ (no difference), pooled <InlineMath math="p = 0.055" /></li>
          <li>Standard error: <InlineMath math="SE = \sqrt{0.055 \times 0.945 \times (\frac{1}{10000} + \frac{1}{10000})} = 0.00323" /></li>
          <li>Z-statistic: <InlineMath math="z = \frac{0.006}{0.00323} = 1.86" /></li>
          <li>p-value (two-sided) = 0.063 → <strong>Not significant at 5% level</strong> (but close — likely need more samples).</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Seeing Theory (Brown University)</strong> — Beautiful interactive visualization of probability concepts including CLT.</li>
          <li><strong>Rice (2007) &quot;Mathematical Statistics and Data Analysis&quot;</strong> — Rigorous treatment with proofs.</li>
          <li><strong>Berry-Esseen theorem</strong> — Quantifies how quickly the normal approximation improves with n.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
