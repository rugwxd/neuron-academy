"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function HypothesisTesting() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Hypothesis testing is a formal framework for deciding whether the data you observed
          is consistent with a specific claim about the world. You start with a <strong>null
          hypothesis</strong> (<InlineMath math="H_0" />) — usually &quot;nothing interesting is
          happening&quot; — and ask: <em>if the null were true, how surprising would my data be?</em>
        </p>
        <p>
          The answer to that question is the <strong>p-value</strong>: the probability of
          observing data as extreme as (or more extreme than) what you actually saw, assuming
          the null hypothesis is true. A small p-value means your data would be very unlikely
          under the null, which gives you evidence to <strong>reject</strong> it.
        </p>
        <p>
          But there are two ways to be wrong. A <strong>Type I error</strong> (false positive) is
          rejecting the null when it&apos;s actually true — you claim there&apos;s an effect when
          there isn&apos;t one. A <strong>Type II error</strong> (false negative) is failing to reject
          the null when it&apos;s actually false — you miss a real effect. The probability of a
          Type I error is controlled by your significance level <InlineMath math="\alpha" /> (typically 0.05).
          The probability of a Type II error is <InlineMath math="\beta" />, and <strong>power</strong> = <InlineMath math="1 - \beta" /> is
          the probability of correctly detecting a true effect.
        </p>
        <p>
          Statistical significance is not the same as practical significance. A massive dataset
          can make a trivially small effect statistically significant. Always pair your p-value
          with an <strong>effect size</strong> and a <strong>confidence interval</strong>.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>The Testing Framework</h3>
        <ol>
          <li>State hypotheses: <InlineMath math="H_0" /> (null) vs. <InlineMath math="H_1" /> (alternative)</li>
          <li>Choose significance level <InlineMath math="\alpha" /> (typically 0.05)</li>
          <li>Compute a test statistic from the data</li>
          <li>Find the p-value: <InlineMath math="P(\text{test stat} \geq \text{observed} \mid H_0)" /></li>
          <li>If <InlineMath math="p \leq \alpha" />, reject <InlineMath math="H_0" /></li>
        </ol>

        <h3>The z-Test (Known Variance)</h3>
        <p>Testing <InlineMath math="H_0: \mu = \mu_0" /> with known <InlineMath math="\sigma" />:</p>
        <BlockMath math="z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}}" />
        <p>Under <InlineMath math="H_0" />, <InlineMath math="z \sim \mathcal{N}(0, 1)" />.</p>

        <h3>Type I and Type II Errors</h3>
        <BlockMath math="\alpha = P(\text{reject } H_0 \mid H_0 \text{ is true}) \quad \text{(false positive rate)}" />
        <BlockMath math="\beta = P(\text{fail to reject } H_0 \mid H_1 \text{ is true}) \quad \text{(false negative rate)}" />
        <BlockMath math="\text{Power} = 1 - \beta = P(\text{reject } H_0 \mid H_1 \text{ is true})" />

        <h3>Power Analysis</h3>
        <p>
          For a two-sample z-test comparing means with equal group sizes, the required sample
          size per group is:
        </p>
        <BlockMath math="n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2\sigma^2}{\delta^2}" />
        <p>
          where <InlineMath math="\delta = \mu_1 - \mu_0" /> is the minimum detectable effect.
          Power increases with larger <InlineMath math="n" />, larger effect size <InlineMath math="\delta" />,
          larger <InlineMath math="\alpha" />, and smaller variance <InlineMath math="\sigma^2" />.
        </p>

        <h3>Effect Size (Cohen&apos;s d)</h3>
        <BlockMath math="d = \frac{\bar{X}_1 - \bar{X}_2}{s_{\text{pooled}}}" />
        <p>
          Convention: <InlineMath math="d = 0.2" /> (small), <InlineMath math="d = 0.5" /> (medium), <InlineMath math="d = 0.8" /> (large).
        </p>
      </TopicSection>

      <TopicSection type="code">
        <CodeBlock
          language="python"
          title="hypothesis_testing_framework.py"
          code={`import numpy as np
from scipy import stats

np.random.seed(42)

# ============================================================
# Example 1: One-sample z-test (known population variance)
# A factory claims widgets weigh 50g on average. We sample 36.
# ============================================================
sample = np.random.normal(50.8, 3.0, size=36)  # true mean slightly above 50
mu_0 = 50.0
sigma = 3.0  # known population std

z_stat = (sample.mean() - mu_0) / (sigma / np.sqrt(len(sample)))
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # two-sided

print("=== One-sample z-test ===")
print(f"Sample mean: {sample.mean():.3f}")
print(f"z-statistic: {z_stat:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Reject H0 at alpha=0.05? {'Yes' if p_value < 0.05 else 'No'}")

# ============================================================
# Example 2: Two-sample test (A/B scenario)
# Control: mean=100, Treatment: mean=103
# ============================================================
control = np.random.normal(100, 15, size=200)
treatment = np.random.normal(103, 15, size=200)

# Welch's t-test (does not assume equal variances)
t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=False)
cohens_d = (treatment.mean() - control.mean()) / np.sqrt(
    (treatment.std(ddof=1)**2 + control.std(ddof=1)**2) / 2
)

print("\\n=== Two-sample Welch's t-test ===")
print(f"Control mean: {control.mean():.2f}, Treatment mean: {treatment.mean():.2f}")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.4f}")
print(f"Cohen's d: {cohens_d:.3f}")
print(f"Reject H0? {'Yes' if p_val < 0.05 else 'No'}")`}
        />

        <CodeBlock
          language="python"
          title="power_analysis.py"
          code={`from scipy import stats
import numpy as np

# ==============================================
# Power analysis: how many samples do we need?
# ==============================================

def required_sample_size(effect_size, alpha=0.05, power=0.80):
    """Sample size per group for a two-sample t-test."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n))

# Cohen's d = 0.2 (small), 0.5 (medium), 0.8 (large)
for d in [0.2, 0.5, 0.8]:
    n = required_sample_size(d)
    print(f"Cohen's d = {d}: need n = {n} per group")

# Output:
# Cohen's d = 0.2: need n = 394 per group
# Cohen's d = 0.5: need n = 64 per group
# Cohen's d = 0.8: need n = 25 per group

# Using statsmodels (more precise, accounts for t-distribution)
from statsmodels.stats.power import TTestIndPower
analysis = TTestIndPower()

for d in [0.2, 0.5, 0.8]:
    n = analysis.solve_power(effect_size=d, alpha=0.05, power=0.80)
    print(f"Cohen's d = {d}: need n = {int(np.ceil(n))} per group (exact)")

# Power curve: how does power change with sample size?
import matplotlib.pyplot as plt
ns = np.arange(10, 500)
for d in [0.2, 0.5, 0.8]:
    powers = [analysis.power(effect_size=d, nobs1=n, alpha=0.05) for n in ns]
    plt.plot(ns, powers, label=f"d = {d}")
plt.axhline(0.8, color='red', linestyle='--', alpha=0.5, label='80% power')
plt.xlabel('Sample size per group')
plt.ylabel('Power')
plt.legend()
plt.title('Power Curves for Two-Sample t-Test')
plt.show()`}
        />

        <CodeBlock
          language="python"
          title="simulate_type_errors.py"
          code={`import numpy as np
from scipy import stats

np.random.seed(42)

# Simulate Type I error rate (false positives)
# H0 is TRUE (both groups have same mean)
n_simulations = 10000
alpha = 0.05
type_1_errors = 0

for _ in range(n_simulations):
    a = np.random.normal(100, 15, size=50)
    b = np.random.normal(100, 15, size=50)  # same distribution!
    _, p = stats.ttest_ind(a, b)
    if p < alpha:
        type_1_errors += 1

print(f"Type I error rate: {type_1_errors/n_simulations:.3f} (expected: {alpha})")

# Simulate power (true positive rate)
# H1 is TRUE (treatment has higher mean)
true_effect = 3.0  # treatment mean is 103 vs control 100
powers_by_n = {}

for n in [20, 50, 100, 200]:
    detections = 0
    for _ in range(n_simulations):
        a = np.random.normal(100, 15, size=n)
        b = np.random.normal(100 + true_effect, 15, size=n)
        _, p = stats.ttest_ind(a, b)
        if p < alpha:
            detections += 1
    powers_by_n[n] = detections / n_simulations
    print(f"n = {n:3d}: power = {powers_by_n[n]:.3f}")

# n =  20: power ≈ 0.12
# n =  50: power ≈ 0.26
# n = 100: power ≈ 0.47
# n = 200: power ≈ 0.75`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Always do power analysis before collecting data</strong>: Determine the minimum detectable effect (MDE) you care about, then calculate the sample size needed to detect it with 80% power. Running an underpowered experiment wastes resources.</li>
          <li><strong>Report effect sizes alongside p-values</strong>: A p-value of 0.001 with Cohen&apos;s d of 0.02 means a statistically significant but practically meaningless effect. Stakeholders care about <em>how big</em>, not just <em>whether</em>.</li>
          <li><strong>One-sided vs. two-sided tests</strong>: Use one-sided only when you have a strong prior that the effect can only go one direction <em>and</em> you would take no action if it went the other way. In practice, two-sided is almost always safer.</li>
          <li><strong>Non-parametric alternatives</strong>: When your data is heavily skewed or has outliers, consider the Mann-Whitney U test (compares medians/ranks) instead of the t-test.</li>
          <li><strong>Permutation tests</strong>: When you&apos;re unsure about distributional assumptions, permutation tests give exact p-values by computing the test statistic over all possible group assignments.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Interpreting p-values as the probability H0 is true</strong>: <InlineMath math="P(\text{data} \mid H_0) \neq P(H_0 \mid \text{data})" />. The p-value is the former; the latter requires Bayes&apos; theorem and a prior on <InlineMath math="H_0" />.</li>
          <li><strong>Stopping data collection when p &lt; 0.05</strong>: This is &quot;optional stopping&quot; and inflates Type I error. If you check after every 10 observations, your real alpha can exceed 0.20. Use sequential testing methods instead.</li>
          <li><strong>Failing to correct for multiple comparisons</strong>: If you test 20 hypotheses at <InlineMath math="\alpha = 0.05" />, you expect 1 false positive even when all nulls are true. Apply corrections like Bonferroni or Benjamini-Hochberg.</li>
          <li><strong>Confusing &quot;not significant&quot; with &quot;no effect&quot;</strong>: Failing to reject <InlineMath math="H_0" /> does not prove <InlineMath math="H_0" /> is true. You may simply be underpowered. Report the confidence interval to show what effect sizes are consistent with your data.</li>
          <li><strong>P-hacking</strong>: Trying different subgroups, transformations, or outcome variables until you find <InlineMath math="p < 0.05" />. Pre-register your analysis plan to guard against this.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Your team runs an A/B test with 10,000 users per group. The treatment group shows a 1.2% lift in conversion (from 5.0% to 5.06%). The p-value is 0.03. Your PM wants to ship the feature. What do you say?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Statistical significance</strong>: Yes, <InlineMath math="p = 0.03 < 0.05" />, so we reject the null of no difference.</li>
          <li><strong>Effect size</strong>: The absolute lift is 0.06 percentage points (from 5.0% to 5.06%). The relative lift of 1.2% is very small.</li>
          <li><strong>Practical significance</strong>: We need to evaluate the business impact. A 0.06pp lift on 1M monthly users means ~600 extra conversions. Is that worth the engineering cost of maintaining this feature?</li>
          <li><strong>Confidence interval</strong>: Compute the 95% CI for the difference. If it&apos;s something like (0.005pp, 0.12pp), even the upper bound is tiny.</li>
          <li><strong>Multiple testing</strong>: Was this the only metric tested? If you looked at 10 metrics, the Bonferroni-corrected threshold is 0.005, and this result would not survive correction.</li>
          <li><strong>Recommendation</strong>: Statistical significance alone doesn&apos;t justify shipping. Consider the magnitude of the effect, cost of the feature, and whether it affects other metrics negatively.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Cohen (1988) &quot;Statistical Power Analysis for the Behavioral Sciences&quot;</strong> — The seminal work on power analysis and effect sizes.</li>
          <li><strong>Wasserstein &amp; Lazar (2016) &quot;The ASA Statement on p-Values&quot;</strong> — The American Statistical Association&apos;s official guidance on what p-values do and don&apos;t mean.</li>
          <li><strong>Lakens (2013) &quot;Calculating and reporting effect sizes&quot;</strong> — Practical guide to choosing and computing effect sizes.</li>
          <li><strong>statsmodels power module documentation</strong> — Comprehensive tooling for power analysis in Python.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
