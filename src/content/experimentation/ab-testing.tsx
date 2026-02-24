"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function ABTesting() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          A/B testing is the gold standard for answering causal questions in tech: <strong>does
          this change actually improve outcomes, or did we just get lucky?</strong> You randomly
          split users into a control group (the current experience) and a treatment group
          (the new experience), measure an outcome, and use statistics to determine whether
          any observed difference is real.
        </p>
        <p>
          The magic of randomization is that it makes the two groups comparable on <em>every</em>
          dimension — observed and unobserved. Without randomization, any difference could
          be driven by a lurking confound. With it, the only systematic difference between
          groups is the treatment itself. This is what gives A/B tests their causal power.
        </p>
        <p>
          But running a good A/B test is harder than it looks. You need to determine the right
          <strong>sample size</strong> upfront through power analysis — too few users and you
          won&apos;t detect real effects; too many and you waste traffic. You need to decide on
          your <strong>primary metric</strong> before the test starts (not after peeking at results).
          And you need to handle the temptation to <strong>peek</strong> at results mid-experiment,
          which inflates your false positive rate unless you use sequential testing methods.
        </p>
        <p>
          Sequential testing and Bayesian A/B testing are modern alternatives that let you
          monitor results continuously and stop early when there&apos;s a clear winner, without
          inflating error rates. These methods are increasingly the industry standard at
          companies running hundreds of experiments simultaneously.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Sample Size Calculation (Two-Proportion z-Test)</h3>
        <p>
          To detect a minimum detectable effect (MDE) of <InlineMath math="\delta" /> from a
          baseline conversion rate <InlineMath math="p_c" /> with significance level <InlineMath math="\alpha" /> and
          power <InlineMath math="1 - \beta" />:
        </p>
        <BlockMath math="n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot (p_c(1-p_c) + p_t(1-p_t))}{\delta^2}" />
        <p>
          where <InlineMath math="p_t = p_c + \delta" /> is the expected treatment rate and <InlineMath math="n" /> is
          per group. A simpler approximation using pooled variance:
        </p>
        <BlockMath math="n \approx \frac{2(z_{\alpha/2} + z_\beta)^2 \cdot \bar{p}(1-\bar{p})}{\delta^2}" />
        <p>where <InlineMath math="\bar{p} = (p_c + p_t)/2" />.</p>

        <h3>For Continuous Metrics (Revenue, Time, etc.)</h3>
        <BlockMath math="n = \frac{2(z_{\alpha/2} + z_\beta)^2 \sigma^2}{\delta^2}" />
        <p>where <InlineMath math="\sigma^2" /> is the variance of the metric and <InlineMath math="\delta" /> is the absolute MDE.</p>

        <h3>Variance Reduction (CUPED)</h3>
        <p>
          CUPED (Controlled-experiment Using Pre-Experiment Data) reduces variance by adjusting
          the metric using a pre-experiment covariate <InlineMath math="X" />:
        </p>
        <BlockMath math="\hat{Y}_{\text{adj}} = Y - \theta(X - \bar{X})" />
        <p>where <InlineMath math="\theta = \text{Cov}(Y, X) / \text{Var}(X)" />. The variance reduction is:</p>
        <BlockMath math="\text{Var}(\hat{Y}_{\text{adj}}) = \text{Var}(Y)(1 - \rho^2_{XY})" />
        <p>
          If <InlineMath math="\rho = 0.5" />, you reduce variance by 25%, equivalent to getting 33% more users for free.
          If <InlineMath math="\rho = 0.8" />, variance drops by 64%.
        </p>

        <h3>Sequential Testing (Group Sequential Design)</h3>
        <p>
          Instead of a fixed-sample test, check results at <InlineMath math="K" /> pre-specified &quot;looks.&quot;
          The <strong>O&apos;Brien-Fleming</strong> spending function uses adjusted significance thresholds
          at each interim analysis:
        </p>
        <BlockMath math="z_k^* = z_{\alpha/2} \cdot \sqrt{\frac{K}{k}}" />
        <p>
          At the first look (<InlineMath math="k=1" />), the threshold is very stringent. By the final
          look (<InlineMath math="k=K" />), it&apos;s close to the unadjusted threshold. The overall Type I
          error rate is maintained at <InlineMath math="\alpha" />.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <CodeBlock
          language="python"
          title="ab_test_sample_size.py"
          code={`import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

# ==================================================
# Sample size calculation for conversion rate test
# ==================================================
baseline_rate = 0.10        # 10% conversion
mde_absolute = 0.01         # want to detect 1pp lift (10% → 11%)
alpha = 0.05
power = 0.80

# Method 1: Manual calculation
p_c = baseline_rate
p_t = baseline_rate + mde_absolute
p_bar = (p_c + p_t) / 2
z_alpha = stats.norm.ppf(1 - alpha / 2)
z_beta = stats.norm.ppf(power)

n_manual = (z_alpha + z_beta)**2 * (p_c*(1-p_c) + p_t*(1-p_t)) / mde_absolute**2
print(f"Manual calculation: {int(np.ceil(n_manual))} per group")

# Method 2: statsmodels (uses Cohen's h for proportions)
effect_size = proportion_effectsize(p_c, p_t)
analysis = NormalIndPower()
n_statsmodels = analysis.solve_power(
    effect_size=abs(effect_size),
    alpha=alpha,
    power=power,
    ratio=1.0,  # equal group sizes
    alternative='two-sided'
)
print(f"statsmodels: {int(np.ceil(n_statsmodels))} per group")

# How long will this test run?
daily_traffic = 50000
total_needed = int(np.ceil(n_manual)) * 2
days = total_needed / daily_traffic
print(f"\\nTotal users needed: {total_needed:,}")
print(f"At {daily_traffic:,} users/day: {days:.1f} days")

# Sensitivity analysis: how does MDE affect sample size?
print("\\n--- Sensitivity Analysis ---")
for mde in [0.005, 0.01, 0.015, 0.02, 0.03]:
    es = proportion_effectsize(baseline_rate, baseline_rate + mde)
    n = analysis.solve_power(effect_size=abs(es), alpha=0.05, power=0.80)
    print(f"MDE = {mde:.3f} ({mde/baseline_rate*100:.0f}% relative): "
          f"n = {int(np.ceil(n)):>8,} per group")`}
        />

        <CodeBlock
          language="python"
          title="ab_test_analysis.py"
          code={`import numpy as np
from scipy import stats

np.random.seed(42)

# ==================================================
# Analyze an A/B test: conversion rate
# ==================================================
# Simulate data
n_control, n_treatment = 15000, 15000
p_control_true, p_treatment_true = 0.100, 0.112  # 12% relative lift

control_conversions = np.random.binomial(n_control, p_control_true)
treatment_conversions = np.random.binomial(n_treatment, p_treatment_true)

p_c = control_conversions / n_control
p_t = treatment_conversions / n_treatment

print(f"Control:   {control_conversions}/{n_control} = {p_c:.4f}")
print(f"Treatment: {treatment_conversions}/{n_treatment} = {p_t:.4f}")
print(f"Absolute lift: {p_t - p_c:.4f}")
print(f"Relative lift: {(p_t - p_c) / p_c * 100:.2f}%")

# Two-proportion z-test
p_pooled = (control_conversions + treatment_conversions) / (n_control + n_treatment)
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
z_stat = (p_t - p_c) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"\\nz-statistic: {z_stat:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Significant at 5%: {'Yes' if p_value < 0.05 else 'No'}")

# Confidence interval for the difference
se_diff = np.sqrt(p_c*(1-p_c)/n_control + p_t*(1-p_t)/n_treatment)
ci_lower = (p_t - p_c) - 1.96 * se_diff
ci_upper = (p_t - p_c) + 1.96 * se_diff
print(f"95% CI for difference: ({ci_lower:.4f}, {ci_upper:.4f})")

# ==================================================
# Analyze revenue (continuous metric) with Welch's t
# ==================================================
control_rev = np.random.lognormal(2.5, 1.2, size=n_control)
treatment_rev = np.random.lognormal(2.55, 1.2, size=n_treatment)

t_stat, p_val_rev = stats.ttest_ind(treatment_rev, control_rev, equal_var=False)
print(f"\\n--- Revenue Test ---")
print(f"Control mean:   \${control_rev.mean():.2f}")
print(f"Treatment mean: \${treatment_rev.mean():.2f}")
print(f"p-value: {p_val_rev:.4f}")`}
        />

        <CodeBlock
          language="python"
          title="sequential_testing.py"
          code={`import numpy as np
from scipy import stats

np.random.seed(42)

# ==================================================
# Sequential testing with O'Brien-Fleming boundaries
# ==================================================

def obrien_fleming_boundary(alpha, n_looks, look_number):
    """O'Brien-Fleming spending function boundary."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    return z_alpha * np.sqrt(n_looks / look_number)

# Simulate a test with 5 interim analyses
n_looks = 5
alpha = 0.05
total_n_per_group = 20000
n_per_look = total_n_per_group // n_looks

true_p_c, true_p_t = 0.10, 0.115

print("=== Sequential Test with O'Brien-Fleming ===")
print(f"{'Look':>4} {'n/group':>8} {'p_c':>8} {'p_t':>8} {'z':>8} "
      f"{'boundary':>10} {'Decision':>12}")
print("-" * 70)

cum_conv_c, cum_conv_t = 0, 0
cum_n_c, cum_n_t = 0, 0
stopped = False

for look in range(1, n_looks + 1):
    # Accumulate data
    new_conv_c = np.random.binomial(n_per_look, true_p_c)
    new_conv_t = np.random.binomial(n_per_look, true_p_t)
    cum_conv_c += new_conv_c
    cum_conv_t += new_conv_t
    cum_n_c += n_per_look
    cum_n_t += n_per_look

    p_c = cum_conv_c / cum_n_c
    p_t = cum_conv_t / cum_n_t
    p_pooled = (cum_conv_c + cum_conv_t) / (cum_n_c + cum_n_t)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/cum_n_c + 1/cum_n_t))
    z = (p_t - p_c) / se if se > 0 else 0

    boundary = obrien_fleming_boundary(alpha, n_looks, look)
    decision = "STOP: Reject" if abs(z) > boundary else "Continue"

    print(f"{look:4d} {cum_n_c:8d} {p_c:8.4f} {p_t:8.4f} {z:8.3f} "
          f"{boundary:10.3f} {decision:>12}")

    if abs(z) > boundary and not stopped:
        print(f"\\n>>> Stopped early at look {look} with "
              f"{cum_n_c + cum_n_t} total users (saved "
              f"{(total_n_per_group*2 - cum_n_c - cum_n_t)} users)")
        stopped = True
        break

if not stopped:
    print("\\n>>> Test ran to completion (all 5 looks)")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Pre-register everything</strong>: Before starting the test, document your primary metric, sample size, test duration, and analysis method. This prevents p-hacking and post-hoc rationalization. Many companies use experiment platforms that enforce this.</li>
          <li><strong>Use CUPED for variance reduction</strong>: Adjusting metrics by pre-experiment covariates (e.g., last week&apos;s behavior) is the single biggest power improvement you can make without increasing sample size. Netflix, Microsoft, and Booking.com all use CUPED.</li>
          <li><strong>Guard against network effects</strong>: If users interact (social networks, marketplaces), treatment effects can &quot;leak&quot; between groups. Use cluster randomization (randomize by region or social cluster) or switchback designs (randomize by time).</li>
          <li><strong>Watch for novelty/primacy effects</strong>: A new feature may see inflated metrics initially (novelty) or deflated metrics (change aversion). Run tests long enough to capture the steady-state effect — typically at least 2 full weeks to cover weekly cycles.</li>
          <li><strong>Have guardrail metrics</strong>: Beyond your primary metric, define guardrail metrics (latency, crash rate, revenue) that must not degrade. Set these up with one-sided tests and don&apos;t require correction for multiple testing — they&apos;re safety checks, not discovery.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Peeking at results and stopping when p &lt; 0.05</strong>: If you check daily and stop the first time <InlineMath math="p < 0.05" />, your true Type I error rate can be 20-30%, not 5%. Use sequential testing (spending functions) or Bayesian methods if you need to monitor continuously.</li>
          <li><strong>Underpowered tests</strong>: Running a test with 500 users per group when you need 50,000 virtually guarantees you won&apos;t detect a real effect. Worse, any &quot;significant&quot; result from an underpowered test is more likely to be a false positive (winner&apos;s curse).</li>
          <li><strong>Wrong unit of randomization</strong>: If you randomize at the page-view level but analyze at the user level, users who visit multiple times pollute both groups. Always randomize at the user level (using a hash of user ID).</li>
          <li><strong>Ignoring the SRM check</strong>: A Sample Ratio Mismatch (SRM) test checks that the control/treatment split matches the expected ratio (e.g., 50/50). If it doesn&apos;t, something is broken — bots, redirect failures, or logging issues. Run an SRM chi-squared test <em>before</em> looking at the primary metric.</li>
          <li><strong>Post-hoc subgroup analysis without correction</strong>: &quot;The feature didn&apos;t work overall, but it worked great for iOS users in France!&quot; This is multiple testing in disguise. Pre-register subgroup analyses or treat them as hypotheses for the next test.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Your company runs an A/B test on a new checkout flow. After 2 weeks, the treatment has a 2.1% lift in conversion (p = 0.04). But your colleague points out that average revenue per user dropped by 1.5% (p = 0.08). How do you make a decision?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Check the primary metric</strong>: If conversion was pre-registered as the primary metric, the 2.1% lift at p = 0.04 is significant. But we need context.</li>
          <li><strong>Evaluate the revenue signal</strong>: A 1.5% revenue drop at p = 0.08 is not significant at the 5% level, but it&apos;s concerning. Compute the 95% CI for the revenue change. If the lower bound of the CI implies a large potential loss, this is a real risk.</li>
          <li><strong>Look at revenue per conversion</strong>: More conversions but lower revenue could mean the new flow is converting low-value users or encouraging smaller purchases. Segment by order value.</li>
          <li><strong>Compute the net impact</strong>: Total revenue = conversion rate x average order value x traffic. Even with lower AOV, higher conversion might yield more total revenue. Compute the CI for total revenue, not just each component.</li>
          <li><strong>Run longer or follow up</strong>: If the decision is ambiguous, extend the test to get tighter CIs on revenue. Or ship the change and run a follow-up test with revenue as the primary metric.</li>
          <li><strong>Key principle</strong>: Never make launch decisions on a single metric. Use a decision framework that weighs the primary metric, guardrail metrics, and business context together.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Kohavi, Tang &amp; Xu (2020) &quot;Trustworthy Online Controlled Experiments&quot;</strong> — The definitive textbook on A/B testing in industry, from the leaders of Microsoft&apos;s experimentation platform.</li>
          <li><strong>Deng et al. (2013) &quot;Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data&quot;</strong> — The original CUPED paper from Microsoft.</li>
          <li><strong>Johari et al. (2017) &quot;Peeking at A/B Tests&quot;</strong> — How always-valid p-values solve the peeking problem. The foundation for sequential testing at Optimizely.</li>
          <li><strong>Larsen et al. (2024) &quot;Statistical Challenges in Online Controlled Experiments&quot;</strong> — A comprehensive survey of modern challenges: interference, long-term effects, and metric sensitivity.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
