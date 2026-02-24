"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function DistributionsDeepDive() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          A <strong>probability distribution</strong> describes how likely different outcomes are for a
          random variable. Think of it as a recipe that tells you: if you draw a value at random, what
          are the chances it falls in a particular range? Distributions are the language of uncertainty —
          every statistical model, hypothesis test, and confidence interval is built on them.
        </p>
        <p>
          <strong>Discrete distributions</strong> (Binomial, Poisson) assign probabilities to specific
          countable values: the number of heads in 10 flips, the number of customer arrivals per hour.
          <strong>Continuous distributions</strong> (Normal, t, Chi-square) assign probabilities to
          ranges of real-valued outcomes: the height of a randomly chosen person, the time until an
          event occurs.
        </p>
        <p>
          The <strong>PDF</strong> (probability density function) for continuous distributions and
          <strong>PMF</strong> (probability mass function) for discrete distributions tell you the
          relative likelihood at each point. The <strong>CDF</strong> (cumulative distribution function)
          tells you the probability of being at or below a given value:
          <InlineMath math="F(x) = P(X \leq x)" />. The CDF always goes from 0 to 1 and is
          non-decreasing.
        </p>
        <p>
          Understanding when each distribution applies is a core data science skill. The normal
          distribution is the default for measurement error and averages. The t-distribution handles
          small samples. The chi-square distribution governs variance estimation and categorical tests.
          Binomial and Poisson handle counts.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Normal (Gaussian) Distribution</h3>
        <p>
          The most important continuous distribution. Parameters: mean <InlineMath math="\mu" /> and
          variance <InlineMath math="\sigma^2" />.
        </p>
        <BlockMath math="f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right), \quad x \in (-\infty, \infty)" />
        <p>Properties:</p>
        <ul>
          <li><InlineMath math="E[X] = \mu" />, <InlineMath math="\text{Var}(X) = \sigma^2" /></li>
          <li>Symmetric about <InlineMath math="\mu" />, with skewness = 0 and excess kurtosis = 0</li>
          <li>68-95-99.7 rule: approximately 68% of data falls within <InlineMath math="\pm 1\sigma" />, 95% within <InlineMath math="\pm 2\sigma" />, 99.7% within <InlineMath math="\pm 3\sigma" /></li>
          <li>Sum of independent normals is normal: <InlineMath math="X + Y \sim \mathcal{N}(\mu_X + \mu_Y, \sigma_X^2 + \sigma_Y^2)" /></li>
        </ul>

        <h3>Student&apos;s t-Distribution</h3>
        <p>
          Arises when estimating the mean of a normally distributed population with unknown variance
          from a small sample. Parameter: degrees of freedom <InlineMath math="\nu" />.
        </p>
        <BlockMath math="f(t) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\;\Gamma\!\left(\frac{\nu}{2}\right)} \left(1 + \frac{t^2}{\nu}\right)^{-(\nu+1)/2}" />
        <p>Properties:</p>
        <ul>
          <li>Symmetric and bell-shaped, but with <strong>heavier tails</strong> than the normal</li>
          <li><InlineMath math="E[X] = 0" /> (for <InlineMath math="\nu > 1" />), <InlineMath math="\text{Var}(X) = \frac{\nu}{\nu - 2}" /> (for <InlineMath math="\nu > 2" />)</li>
          <li>As <InlineMath math="\nu \to \infty" />, the t-distribution converges to the standard normal</li>
          <li>For <InlineMath math="\nu \leq 30" />, use t instead of z for hypothesis tests and confidence intervals</li>
        </ul>

        <h3>Chi-Square Distribution</h3>
        <p>
          The sum of <InlineMath math="k" /> squared standard normal variables:
          if <InlineMath math="Z_i \sim \mathcal{N}(0,1)" />, then <InlineMath math="\sum Z_i^2 \sim \chi^2_k" />.
        </p>
        <BlockMath math="f(x) = \frac{1}{2^{k/2}\,\Gamma(k/2)}\, x^{k/2 - 1}\, e^{-x/2}, \quad x > 0" />
        <p>Properties:</p>
        <ul>
          <li><InlineMath math="E[X] = k" />, <InlineMath math="\text{Var}(X) = 2k" /></li>
          <li>Always non-negative and right-skewed (less so as <InlineMath math="k" /> increases)</li>
          <li>Used in: goodness-of-fit tests, tests of independence, confidence intervals for variance</li>
          <li>Key identity: <InlineMath math="\frac{(n-1)s^2}{\sigma^2} \sim \chi^2_{n-1}" /> when sampling from a normal population</li>
        </ul>

        <h3>Binomial Distribution</h3>
        <p>
          The number of successes in <InlineMath math="n" /> independent Bernoulli trials, each with
          success probability <InlineMath math="p" />.
        </p>
        <BlockMath math="P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n" />
        <p>Properties:</p>
        <ul>
          <li><InlineMath math="E[X] = np" />, <InlineMath math="\text{Var}(X) = np(1-p)" /></li>
          <li>For large <InlineMath math="n" /> with moderate <InlineMath math="p" />, approximated by <InlineMath math="\mathcal{N}(np, np(1-p))" /> (by the CLT)</li>
          <li>For large <InlineMath math="n" /> with small <InlineMath math="p" />, approximated by <InlineMath math="\text{Poisson}(np)" /></li>
        </ul>

        <h3>Poisson Distribution</h3>
        <p>
          Models the number of events in a fixed interval when events occur independently at a constant
          average rate <InlineMath math="\lambda" />.
        </p>
        <BlockMath math="P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots" />
        <p>Properties:</p>
        <ul>
          <li><InlineMath math="E[X] = \lambda" />, <InlineMath math="\text{Var}(X) = \lambda" /> (mean equals variance — a diagnostic check)</li>
          <li>Sum of independent Poissons is Poisson: <InlineMath math="\text{Poi}(\lambda_1) + \text{Poi}(\lambda_2) \sim \text{Poi}(\lambda_1 + \lambda_2)" /></li>
          <li>For large <InlineMath math="\lambda" />, approximated by <InlineMath math="\mathcal{N}(\lambda, \lambda)" /></li>
        </ul>
      </TopicSection>

      <TopicSection type="code">
        <CodeBlock
          language="python"
          title="continuous_distributions.py"
          code={`import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(-5, 5, 500)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# ── Row 1: PDFs ──
# Normal distribution: varying mean and std
ax = axes[0, 0]
for mu, sigma, label in [(0, 1, "N(0,1)"), (0, 2, "N(0,4)"), (2, 1, "N(2,1)")]:
    ax.plot(x, stats.norm.pdf(x, mu, sigma), linewidth=2, label=label)
ax.set_title("Normal PDF")
ax.legend()

# t-distribution: varying degrees of freedom
ax = axes[0, 1]
ax.plot(x, stats.norm.pdf(x), linewidth=2, linestyle="--", label="Normal", color="black")
for df in [1, 3, 10, 30]:
    ax.plot(x, stats.t.pdf(x, df), linewidth=2, label=f"t(df={df})")
ax.set_title("t-Distribution PDF")
ax.legend(fontsize=8)

# Chi-square distribution: varying df
ax = axes[0, 2]
x_chi = np.linspace(0.01, 20, 500)
for k in [1, 2, 3, 5, 10]:
    ax.plot(x_chi, stats.chi2.pdf(x_chi, k), linewidth=2, label=f"k={k}")
ax.set_title("Chi-Square PDF")
ax.legend()

# ── Row 2: CDFs ──
ax = axes[1, 0]
for mu, sigma, label in [(0, 1, "N(0,1)"), (0, 2, "N(0,4)"), (2, 1, "N(2,1)")]:
    ax.plot(x, stats.norm.cdf(x, mu, sigma), linewidth=2, label=label)
ax.set_title("Normal CDF")
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
ax.legend()

ax = axes[1, 1]
ax.plot(x, stats.norm.cdf(x), linewidth=2, linestyle="--", label="Normal", color="black")
for df in [1, 3, 10, 30]:
    ax.plot(x, stats.t.cdf(x, df), linewidth=2, label=f"t(df={df})")
ax.set_title("t-Distribution CDF")
ax.legend(fontsize=8)

ax = axes[1, 2]
for k in [1, 2, 3, 5, 10]:
    ax.plot(x_chi, stats.chi2.cdf(x_chi, k), linewidth=2, label=f"k={k}")
ax.set_title("Chi-Square CDF")
ax.legend()

fig.suptitle("Continuous Distributions: PDF (top) and CDF (bottom)", fontsize=14)
fig.tight_layout()
plt.show()`}
        />

        <CodeBlock
          language="python"
          title="discrete_distributions.py"
          code={`import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# ── Binomial PMF ──
ax = axes[0, 0]
n = 20
for p in [0.2, 0.5, 0.8]:
    k = np.arange(0, n + 1)
    pmf = stats.binom.pmf(k, n, p)
    ax.bar(k + (p - 0.5) * 0.25, pmf, width=0.25, alpha=0.7, label=f"p={p}")
ax.set_title(f"Binomial PMF (n={n})")
ax.set_xlabel("k (number of successes)")
ax.legend()

# ── Binomial CDF ──
ax = axes[0, 1]
for p in [0.2, 0.5, 0.8]:
    k = np.arange(0, n + 1)
    cdf = stats.binom.cdf(k, n, p)
    ax.step(k, cdf, linewidth=2, where="mid", label=f"p={p}")
ax.set_title(f"Binomial CDF (n={n})")
ax.legend()

# ── Poisson PMF ──
ax = axes[1, 0]
k = np.arange(0, 25)
for lam in [1, 4, 10]:
    pmf = stats.poisson.pmf(k, lam)
    ax.bar(k + (lam - 4) * 0.15, pmf, width=0.4, alpha=0.7, label=f"lambda={lam}")
ax.set_title("Poisson PMF")
ax.set_xlabel("k (number of events)")
ax.legend()

# ── Poisson CDF ──
ax = axes[1, 1]
for lam in [1, 4, 10]:
    cdf = stats.poisson.cdf(k, lam)
    ax.step(k, cdf, linewidth=2, where="mid", label=f"lambda={lam}")
ax.set_title("Poisson CDF")
ax.legend()

fig.suptitle("Discrete Distributions: PMF (left) and CDF (right)", fontsize=14)
fig.tight_layout()
plt.show()`}
        />

        <CodeBlock
          language="python"
          title="distribution_fitting.py"
          code={`import numpy as np
from scipy import stats

np.random.seed(42)

# ── Fit a distribution to observed data ──
# Simulate "observed" data from an unknown distribution
data = np.random.gamma(shape=3, scale=2, size=1000)

# Try fitting several distributions and compare
candidates = {
    "Normal":     stats.norm,
    "Lognormal":  stats.lognorm,
    "Gamma":      stats.gamma,
    "Exponential": stats.expon,
}

print("Distribution Fitting (AIC-like comparison via log-likelihood)")
print("-" * 60)

results = []
for name, dist in candidates.items():
    # Fit parameters via MLE
    params = dist.fit(data)

    # Compute log-likelihood
    log_lik = np.sum(dist.logpdf(data, *params))
    n_params = len(params)
    aic = 2 * n_params - 2 * log_lik

    # Kolmogorov-Smirnov goodness-of-fit test
    ks_stat, ks_pval = stats.kstest(data, dist.cdf, args=params)

    results.append((name, aic, ks_stat, ks_pval, params))
    print(f"{name:12s}: AIC={aic:8.1f}  KS-stat={ks_stat:.4f}  p={ks_pval:.4f}")

# Best fit: lowest AIC and highest KS p-value
best = min(results, key=lambda x: x[1])
print(f"\\nBest fit: {best[0]} (AIC={best[1]:.1f})")

# ── Q-Q plot to visually assess fit ──
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 6))
stats.probplot(data, dist=stats.gamma, sparams=stats.gamma.fit(data)[:1], plot=ax)
ax.set_title("Q-Q Plot: Data vs Fitted Gamma")
plt.show()`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>Normal distribution is the default for error modeling</strong>: Measurement errors,
            residuals from regression, and sensor noise are often modeled as normal. When you assume
            &quot;errors are normally distributed,&quot; you&apos;re invoking this.
          </li>
          <li>
            <strong>Use t-distribution for small samples</strong>: When <InlineMath math="n < 30" /> and
            the population standard deviation is unknown, the t-distribution accounts for the additional
            uncertainty from estimating <InlineMath math="\sigma" />. As <InlineMath math="n" /> grows,
            the difference vanishes.
          </li>
          <li>
            <strong>Poisson for count data with a known rate</strong>: Website visits per minute, insurance
            claims per year, typos per page. Key diagnostic: if the variance roughly equals the mean,
            Poisson is a good fit. If variance far exceeds the mean, use Negative Binomial instead
            (overdispersion).
          </li>
          <li>
            <strong>Chi-square appears in many tests</strong>: Chi-square test of independence (contingency
            tables), goodness-of-fit test, and confidence intervals for variance all rely on the chi-square
            distribution.
          </li>
          <li>
            <strong>Use Q-Q plots to assess distributional fit</strong>: Plot sample quantiles against
            theoretical quantiles. If the points lie on the 45-degree line, the distribution fits well.
            Deviations in the tails reveal heavy tails or skewness.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Assuming all data is normally distributed</strong>: Many real-world quantities
            (income, wait times, counts, proportions) are <em>not</em> normal. Always check with a
            histogram or Q-Q plot before applying methods that assume normality.
          </li>
          <li>
            <strong>Using the normal when you should use the t</strong>: With small samples, a z-test
            gives confidence intervals that are too narrow. The t-distribution&apos;s wider tails provide
            honest uncertainty estimates.
          </li>
          <li>
            <strong>Confusing PDF values with probabilities</strong>: For continuous distributions,
            <InlineMath math="f(x)" /> can exceed 1 — it&apos;s a <em>density</em>, not a probability.
            Only <InlineMath math="P(a \leq X \leq b) = \int_a^b f(x)\,dx" /> is a probability.
          </li>
          <li>
            <strong>Using Poisson when events aren&apos;t independent</strong>: If one event makes the next
            more likely (e.g., earthquake aftershocks), the Poisson assumption breaks. Use a Hawkes
            process or other self-exciting model.
          </li>
          <li>
            <strong>Forgetting the support of a distribution</strong>: Chi-square and Poisson only take
            non-negative values. Plugging negative numbers into a gamma PDF returns nonsense. Always
            check what range of values a distribution can take.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> A call center receives an average of 4.5 calls per minute. (a)
          What is the probability of receiving exactly 7 calls in a given minute? (b) What is the
          probability of receiving 0 calls? (c) What distribution would you use for the time between
          calls?
        </p>
        <p><strong>Answer:</strong></p>
        <p>
          The number of calls per minute follows a <strong>Poisson distribution</strong> with
          <InlineMath math="\lambda = 4.5" />, assuming calls arrive independently at a constant rate.
        </p>
        <p>(a) Probability of exactly 7 calls:</p>
        <BlockMath math="P(X = 7) = \frac{4.5^7 \cdot e^{-4.5}}{7!} = \frac{37366.95 \times 0.01111}{5040} \approx 0.0824" />
        <p>About 8.2% chance.</p>
        <p>(b) Probability of 0 calls:</p>
        <BlockMath math="P(X = 0) = \frac{4.5^0 \cdot e^{-4.5}}{0!} = e^{-4.5} \approx 0.0111" />
        <p>About 1.1% chance — rare but not impossible.</p>
        <p>
          (c) The time between events in a Poisson process follows an <strong>Exponential distribution</strong>
          with rate parameter <InlineMath math="\lambda = 4.5" /> per minute, giving a mean inter-arrival
          time of <InlineMath math="1/\lambda = 1/4.5 \approx 0.222" /> minutes (about 13.3 seconds). This
          is a fundamental connection: Poisson counts and Exponential inter-arrival times are two sides of
          the same process.
        </p>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Casella &amp; Berger, &quot;Statistical Inference&quot;</strong> — The standard graduate-level text covering all major distributions with proofs and derivations.</li>
          <li><strong>Seeing Theory (Brown University)</strong> — Interactive visualizations of distributions at seeing-theory.brown.edu.</li>
          <li><strong>scipy.stats documentation</strong> — Over 100 distributions implemented with consistent API (pdf, cdf, ppf, rvs, fit).</li>
          <li><strong>Michael Betancourt, &quot;Probability Theory for Scientists and Engineers&quot;</strong> — Free notes connecting distributions to real-world modeling scenarios.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
