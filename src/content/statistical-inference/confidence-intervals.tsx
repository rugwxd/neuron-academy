"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function ConfidenceIntervals() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          A confidence interval gives you a <strong>range of plausible values</strong> for a population
          parameter based on your sample data. Instead of saying &quot;the average customer spends $47,&quot;
          you say &quot;we&apos;re 95% confident the average customer spends between $42 and $52.&quot;
        </p>
        <p>
          The key insight is that a single point estimate (like a sample mean) is almost certainly
          not exactly the true value. Confidence intervals acknowledge this uncertainty. A wider
          interval means more uncertainty; a narrower one means your estimate is more precise.
        </p>
        <p>
          The &quot;95% confidence&quot; part is often misunderstood. It does <strong>not</strong> mean there&apos;s a
          95% probability the true parameter is in this specific interval. The true parameter is
          fixed — it&apos;s either in the interval or it isn&apos;t. What 95% means is: if you repeated
          the experiment many times and built a CI each time, about 95% of those intervals would
          contain the true parameter. It&apos;s a statement about the <em>procedure</em>, not about any
          single interval.
        </p>
        <p>
          Bootstrap confidence intervals offer a powerful alternative when you can&apos;t rely on
          distributional assumptions. Instead of assuming normality, you resample your data
          thousands of times and directly observe how your estimate varies — the spread of
          those resampled estimates <em>is</em> your confidence interval.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Classical (Parametric) Confidence Interval for a Mean</h3>
        <p>
          Given a sample of size <InlineMath math="n" /> with sample mean <InlineMath math="\bar{X}" /> and
          sample standard deviation <InlineMath math="s" />, the <InlineMath math="(1 - \alpha)" /> confidence
          interval for the population mean <InlineMath math="\mu" /> is:
        </p>
        <BlockMath math="\bar{X} \pm t_{\alpha/2, \, n-1} \cdot \frac{s}{\sqrt{n}}" />
        <p>
          where <InlineMath math="t_{\alpha/2, \, n-1}" /> is the critical value from the t-distribution
          with <InlineMath math="n-1" /> degrees of freedom. For large <InlineMath math="n" />, this
          approaches the z-critical value <InlineMath math="z_{\alpha/2}" /> (e.g., 1.96 for 95% CI).
        </p>

        <h3>Confidence Interval for a Proportion</h3>
        <p>
          For a sample proportion <InlineMath math="\hat{p}" /> with sample size <InlineMath math="n" />:
        </p>
        <BlockMath math="\hat{p} \pm z_{\alpha/2} \cdot \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}" />
        <p>
          This relies on the CLT approximation and works well when <InlineMath math="n\hat{p} \geq 10" /> and <InlineMath math="n(1-\hat{p}) \geq 10" />.
          For small samples or extreme proportions, use the Wilson score interval instead.
        </p>

        <h3>Margin of Error and Sample Size</h3>
        <p>
          The margin of error <InlineMath math="E" /> is the half-width of the CI. To achieve a desired
          margin of error for a mean:
        </p>
        <BlockMath math="n = \left(\frac{z_{\alpha/2} \cdot \sigma}{E}\right)^2" />
        <p>
          Halving the margin of error requires <strong>quadrupling</strong> the sample size — precision
          comes at a steep cost.
        </p>

        <h3>Bootstrap Confidence Intervals</h3>
        <p>
          Let <InlineMath math="\hat{\theta}" /> be the statistic of interest computed on the original sample.
          Draw <InlineMath math="B" /> bootstrap resamples (sampling with replacement), computing <InlineMath math="\hat{\theta}^*_1, \hat{\theta}^*_2, \ldots, \hat{\theta}^*_B" />.
        </p>
        <p><strong>Percentile method:</strong></p>
        <BlockMath math="CI_{1-\alpha} = \left[\hat{\theta}^*_{(\alpha/2)}, \; \hat{\theta}^*_{(1-\alpha/2)}\right]" />
        <p><strong>BCa (Bias-Corrected and Accelerated) method:</strong> adjusts for bias and skewness in the bootstrap distribution, yielding more accurate coverage:</p>
        <BlockMath math="z_0 = \Phi^{-1}\left(\frac{\#\{\hat{\theta}^*_b < \hat{\theta}\}}{B}\right)" />
        <BlockMath math="\alpha_1 = \Phi\left(z_0 + \frac{z_0 + z_{\alpha/2}}{1 - a(z_0 + z_{\alpha/2})}\right)" />
        <p>
          where <InlineMath math="a" /> is the acceleration constant estimated via jackknife.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <CodeBlock
          language="python"
          title="classical_confidence_intervals.py"
          code={`import numpy as np
from scipy import stats

np.random.seed(42)

# --- CI for a mean (unknown population variance → t-distribution) ---
data = np.array([23.1, 25.4, 22.8, 26.1, 24.3, 27.0, 23.9, 25.6, 24.7, 26.5])
n = len(data)
x_bar = data.mean()
s = data.std(ddof=1)            # sample std (Bessel's correction)
se = s / np.sqrt(n)             # standard error
alpha = 0.05

t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
ci_lower = x_bar - t_crit * se
ci_upper = x_bar + t_crit * se
print(f"Sample mean: {x_bar:.2f}")
print(f"95% CI for mean: ({ci_lower:.2f}, {ci_upper:.2f})")

# scipy shortcut
ci = stats.t.interval(0.95, df=n - 1, loc=x_bar, scale=se)
print(f"scipy CI: ({ci[0]:.2f}, {ci[1]:.2f})")

# --- CI for a proportion ---
successes, trials = 84, 200
p_hat = successes / trials
se_p = np.sqrt(p_hat * (1 - p_hat) / trials)
z_crit = stats.norm.ppf(1 - alpha / 2)
ci_prop = (p_hat - z_crit * se_p, p_hat + z_crit * se_p)
print(f"\\nProportion: {p_hat:.3f}")
print(f"95% CI for proportion: ({ci_prop[0]:.3f}, {ci_prop[1]:.3f})")

# --- Sample size calculation ---
desired_margin = 0.5   # want mean within ±0.5
sigma_est = 3.0        # estimated population std
n_needed = (z_crit * sigma_est / desired_margin) ** 2
print(f"\\nSample size needed for margin ±{desired_margin}: {int(np.ceil(n_needed))}")`}
        />

        <CodeBlock
          language="python"
          title="bootstrap_confidence_intervals.py"
          code={`import numpy as np
from scipy import stats

np.random.seed(42)

# Generate sample data (median income — skewed distribution)
income = np.random.lognormal(mean=10.8, sigma=0.6, size=200)
print(f"Sample median income: \${np.median(income):,.0f}")

# --- Percentile Bootstrap CI for the median ---
B = 10000
boot_medians = np.array([
    np.median(np.random.choice(income, size=len(income), replace=True))
    for _ in range(B)
])

ci_percentile = np.percentile(boot_medians, [2.5, 97.5])
print(f"Percentile bootstrap 95% CI: (\${ci_percentile[0]:,.0f}, \${ci_percentile[1]:,.0f})")

# --- BCa Bootstrap (using scipy) ---
res = stats.bootstrap(
    (income,),
    statistic=np.median,
    n_resamples=10000,
    confidence_level=0.95,
    method='BCa',
    random_state=42
)
print(f"BCa bootstrap 95% CI: (\${res.confidence_interval.low:,.0f}, \${res.confidence_interval.high:,.0f})")

# --- Visualize coverage: run 100 experiments ---
true_mean = 5.0
covered = 0
for _ in range(100):
    sample = np.random.normal(true_mean, 2, size=30)
    ci = stats.t.interval(0.95, df=29, loc=sample.mean(),
                          scale=sample.std(ddof=1) / np.sqrt(30))
    if ci[0] <= true_mean <= ci[1]:
        covered += 1

print(f"\\nCoverage check: {covered}/100 CIs contain the true mean")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Report CIs, not just point estimates</strong>: A model accuracy of 87% means little without knowing whether the CI is (85%, 89%) or (72%, 98%). Always report uncertainty.</li>
          <li><strong>Use bootstrap for non-standard statistics</strong>: Medians, ratios, Gini coefficients, AUC — anything where you don&apos;t have a clean formula for the standard error. Bootstrap handles it effortlessly.</li>
          <li><strong>Wilson score for proportions</strong>: The Wald interval (the normal approximation) can produce nonsensical results for proportions near 0 or 1. Use <InlineMath math="\texttt{statsmodels.stats.proportion.proportion\_confint(method='wilson')}" /> instead.</li>
          <li><strong>Watch for dependence</strong>: Standard CIs assume i.i.d. data. For time series, clustered data, or repeated measures, use cluster-robust standard errors or hierarchical bootstrap.</li>
          <li><strong>CI width as a planning tool</strong>: Before running an experiment, compute the expected CI width for your planned sample size. If it&apos;s too wide to be useful, collect more data.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Saying &quot;95% probability the true value is in this interval&quot;</strong>: The true value is fixed. The correct interpretation is about the long-run frequency of the procedure. If you want probability statements about parameters, use Bayesian credible intervals.</li>
          <li><strong>Ignoring the confidence level when comparing CIs</strong>: A 99% CI is wider than a 95% CI for the same data. You&apos;re trading precision for coverage.</li>
          <li><strong>Using z-critical values with small samples</strong>: For <InlineMath math="n < 30" />, use the t-distribution. The z-approximation underestimates the CI width and produces undercoverage.</li>
          <li><strong>Treating non-overlapping CIs as the definitive test for significance</strong>: Two CIs can overlap and the difference can still be statistically significant. The correct approach is to construct a CI for the <em>difference</em>.</li>
          <li><strong>Too few bootstrap resamples</strong>: Use at least 10,000 for percentile CIs and 15,000+ for BCa. Fewer resamples give unstable interval endpoints.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You train a classifier and get accuracy of 0.88 on a test set of 500 examples. Construct a 95% confidence interval. A colleague claims their model gets 0.91 on the same test set — is it significantly better?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>Accuracy is a proportion: <InlineMath math="\hat{p} = 0.88" />, <InlineMath math="n = 500" /></li>
          <li>Standard error: <InlineMath math="SE = \sqrt{\frac{0.88 \times 0.12}{500}} = 0.01453" /></li>
          <li>95% CI: <InlineMath math="0.88 \pm 1.96 \times 0.01453 = (0.8515, 0.9085)" /></li>
          <li>For the colleague&apos;s model: <InlineMath math="SE = \sqrt{\frac{0.91 \times 0.09}{500}} = 0.01279" />, CI = (0.885, 0.935)</li>
          <li>The CIs overlap. To test whether the difference is significant, we need a CI for the <em>difference</em>. Since both models are evaluated on the <strong>same</strong> test set, the predictions are correlated — we must use McNemar&apos;s test or a paired bootstrap, not an independent two-proportion z-test.</li>
          <li><strong>Key insight:</strong> Overlapping CIs do not mean &quot;no significant difference,&quot; and same-test-set comparisons require paired tests.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Efron &amp; Tibshirani (1994) &quot;An Introduction to the Bootstrap&quot;</strong> — The definitive reference on bootstrap methods, including BCa and bootstrap hypothesis tests.</li>
          <li><strong>Agresti &amp; Coull (1998) &quot;Approximate is Better than Exact for Interval Estimation of Binomial Proportions&quot;</strong> — Why the Wilson interval outperforms the Wald interval.</li>
          <li><strong>Larry Wasserman &quot;All of Statistics&quot; Ch. 6</strong> — Clear treatment of confidence intervals from a modern perspective.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
