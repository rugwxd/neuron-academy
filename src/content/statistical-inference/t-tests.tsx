"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function TTests() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          The t-test is the workhorse of statistical inference. It answers a simple question:
          <strong> is the difference I see in means real, or could it be due to random chance?</strong>
          It&apos;s called a &quot;t-test&quot; because it uses the <strong>t-distribution</strong> — a
          bell curve with heavier tails than the normal distribution, which accounts for the
          extra uncertainty of estimating variance from a sample rather than knowing it exactly.
        </p>
        <p>
          There are three main flavors. The <strong>one-sample t-test</strong> asks whether a sample
          mean differs from a hypothesized value (e.g., &quot;Is the average delivery time different
          from 30 minutes?&quot;). The <strong>two-sample t-test</strong> (also called the independent
          samples t-test) asks whether two groups have different means (e.g., &quot;Do users on the
          new design spend more time on the page?&quot;). The <strong>paired t-test</strong> handles
          paired observations — the same subjects measured twice, like before-and-after a treatment.
        </p>
        <p>
          The paired t-test is special because it eliminates between-subject variability. Instead
          of comparing two noisy groups, you look at within-subject differences, which are
          typically much less variable. This makes paired designs far more powerful than independent
          designs when pairing is possible.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>One-Sample t-Test</h3>
        <p>
          Testing <InlineMath math="H_0: \mu = \mu_0" /> against <InlineMath math="H_1: \mu \neq \mu_0" />:
        </p>
        <BlockMath math="t = \frac{\bar{X} - \mu_0}{s / \sqrt{n}}" />
        <p>
          where <InlineMath math="s" /> is the sample standard deviation. Under <InlineMath math="H_0" />,
          <InlineMath math="t \sim t_{n-1}" /> (Student&apos;s t-distribution with <InlineMath math="n-1" /> degrees of freedom).
        </p>

        <h3>Two-Sample t-Test (Independent)</h3>
        <p><strong>Equal variance assumed (Student&apos;s t-test):</strong></p>
        <BlockMath math="t = \frac{\bar{X}_1 - \bar{X}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}" />
        <p>where the pooled standard deviation is:</p>
        <BlockMath math="s_p = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}" />
        <p>Degrees of freedom: <InlineMath math="df = n_1 + n_2 - 2" /></p>

        <p><strong>Unequal variance (Welch&apos;s t-test — preferred in practice):</strong></p>
        <BlockMath math="t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}" />
        <p>Degrees of freedom via the Welch-Satterthwaite approximation:</p>
        <BlockMath math="df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1 - 1} + \frac{(s_2^2/n_2)^2}{n_2 - 1}}" />

        <h3>Paired t-Test</h3>
        <p>
          Let <InlineMath math="D_i = X_{i,\text{after}} - X_{i,\text{before}}" /> be the paired differences.
          The paired t-test is simply a one-sample t-test on the differences:
        </p>
        <BlockMath math="t = \frac{\bar{D}}{s_D / \sqrt{n}}" />
        <p>
          where <InlineMath math="\bar{D}" /> and <InlineMath math="s_D" /> are the mean and standard
          deviation of the differences, with <InlineMath math="df = n - 1" />.
        </p>

        <h3>Assumptions</h3>
        <ul>
          <li>Data is continuous (or approximately so)</li>
          <li>Observations are independent (within groups; paired observations are dependent by design)</li>
          <li>Data is approximately normally distributed (the t-test is robust to non-normality for <InlineMath math="n \geq 30" /> by the CLT)</li>
          <li>For Student&apos;s t-test: equal variances across groups (use Welch&apos;s t-test if unsure)</li>
        </ul>
      </TopicSection>

      <TopicSection type="code">
        <CodeBlock
          language="python"
          title="t_tests_complete.py"
          code={`import numpy as np
from scipy import stats

np.random.seed(42)

# ==================================================
# 1. ONE-SAMPLE T-TEST
# Question: Is the average page load time different from 3 seconds?
# ==================================================
load_times = np.array([3.2, 2.8, 3.5, 3.1, 2.9, 3.4, 3.0, 3.6, 2.7, 3.3,
                        3.1, 3.5, 2.8, 3.2, 3.4, 3.0, 2.9, 3.3, 3.1, 3.2])
mu_0 = 3.0

t_stat, p_value = stats.ttest_1samp(load_times, mu_0)
print("=== One-Sample t-Test ===")
print(f"Sample mean: {load_times.mean():.3f}s")
print(f"H0: mu = {mu_0}s")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'}")

# Manual calculation
n = len(load_times)
t_manual = (load_times.mean() - mu_0) / (load_times.std(ddof=1) / np.sqrt(n))
p_manual = 2 * stats.t.sf(abs(t_manual), df=n-1)
print(f"Manual check: t={t_manual:.4f}, p={p_manual:.4f}")

# ==================================================
# 2. TWO-SAMPLE T-TEST (Independent)
# Question: Does the new checkout flow increase order value?
# ==================================================
control_revenue = np.random.normal(85, 20, size=150)
treatment_revenue = np.random.normal(90, 22, size=150)

# Welch's t-test (default: equal_var=False is safer)
t_stat, p_value = stats.ttest_ind(treatment_revenue, control_revenue,
                                   equal_var=False)

# Effect size (Cohen's d)
pooled_std = np.sqrt((control_revenue.std(ddof=1)**2 +
                      treatment_revenue.std(ddof=1)**2) / 2)
cohens_d = (treatment_revenue.mean() - control_revenue.mean()) / pooled_std

print("\\n=== Two-Sample t-Test (Welch's) ===")
print(f"Control mean:   \${control_revenue.mean():.2f}")
print(f"Treatment mean: \${treatment_revenue.mean():.2f}")
print(f"Difference:     \${treatment_revenue.mean() - control_revenue.mean():.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Cohen's d: {cohens_d:.3f}")

# ==================================================
# 3. PAIRED T-TEST
# Question: Does a training program improve test scores?
# ==================================================
n_students = 25
before = np.random.normal(70, 10, size=n_students)
# After: correlated with before (same students), slight improvement
improvement = np.random.normal(5, 4, size=n_students)
after = before + improvement

t_stat, p_value = stats.ttest_rel(after, before)
diffs = after - before

print("\\n=== Paired t-Test ===")
print(f"Mean before: {before.mean():.2f}")
print(f"Mean after:  {after.mean():.2f}")
print(f"Mean difference: {diffs.mean():.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")

# Show why paired > independent here
t_ind, p_ind = stats.ttest_ind(after, before)
print(f"\\nIf we WRONGLY used independent t-test:")
print(f"  t = {t_ind:.4f}, p = {p_ind:.4f}")
print(f"Paired test is more powerful because it removes between-subject variance.")`}
        />

        <CodeBlock
          language="python"
          title="t_test_assumptions_check.py"
          code={`import numpy as np
from scipy import stats

np.random.seed(42)
data_a = np.random.normal(50, 10, size=40)
data_b = np.random.lognormal(3.9, 0.2, size=40)  # skewed

# 1. Check normality with Shapiro-Wilk test
for name, data in [("Group A (normal)", data_a), ("Group B (skewed)", data_b)]:
    stat, p = stats.shapiro(data)
    print(f"{name}: Shapiro-Wilk p = {p:.4f} "
          f"({'Normal' if p > 0.05 else 'Non-normal'})")

# 2. Check equal variances with Levene's test
stat, p = stats.levene(data_a, data_b)
print(f"\\nLevene's test: p = {p:.4f} "
      f"({'Equal variances' if p > 0.05 else 'Unequal variances'})")

# 3. If normality is violated, use non-parametric alternative
# Mann-Whitney U test (compares ranks, not means)
u_stat, p_mw = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
print(f"\\nMann-Whitney U test: U = {u_stat:.1f}, p = {p_mw:.4f}")

# Wilcoxon signed-rank test (non-parametric paired test)
paired_a = np.random.normal(50, 10, size=30)
paired_b = paired_a + np.random.normal(2, 5, size=30)
w_stat, p_w = stats.wilcoxon(paired_b - paired_a)
print(f"Wilcoxon signed-rank test: W = {w_stat:.1f}, p = {p_w:.4f}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Always use Welch&apos;s t-test over Student&apos;s t-test</strong>: Welch&apos;s test does not assume equal variances and performs just as well when variances are actually equal. There&apos;s no downside. <InlineMath math="\texttt{scipy.stats.ttest\_ind(equal\_var=False)}" /> is what you want.</li>
          <li><strong>Prefer paired designs when possible</strong>: Paired tests are more powerful because they eliminate between-subject variability. In A/B testing, within-subject designs (showing both variants to the same users) or CUPED-adjusted metrics achieve the same effect.</li>
          <li><strong>The t-test is robust</strong>: For reasonably-sized samples (<InlineMath math="n \geq 30" /> per group), the t-test handles moderate non-normality well thanks to the CLT. Don&apos;t default to non-parametric tests without reason — they have less power.</li>
          <li><strong>Unequal sample sizes</strong>: The t-test works fine with unequal group sizes, but power is maximized when groups are equal. If you&apos;re designing an experiment, aim for balanced groups.</li>
          <li><strong>One-sided tests in practice</strong>: If you truly only care about whether treatment is <em>better</em> (not just different), a one-sided test gives you more power. But be honest — if a large negative effect would also be important to know, use two-sided.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Using an independent t-test on paired data</strong>: This throws away the pairing information and reduces power dramatically. If the same subjects appear in both conditions, use a paired test.</li>
          <li><strong>Not checking for outliers</strong>: The t-test is sensitive to extreme values because it&apos;s based on means and variances. A single outlier can flip your result. Check with box plots and consider trimmed means or the Mann-Whitney test.</li>
          <li><strong>Applying the t-test to non-independent observations</strong>: If users can appear in both groups (contamination), or if there&apos;s network interference between groups, the independence assumption is violated. The p-value becomes meaningless.</li>
          <li><strong>Multiple t-tests instead of ANOVA</strong>: Comparing 4 groups pairwise requires 6 t-tests, inflating the family-wise error rate. Use ANOVA first, then post-hoc tests (Tukey HSD) if ANOVA is significant.</li>
          <li><strong>Equal variance assumption with Student&apos;s t-test</strong>: If you run a Levene&apos;s test first and then decide which t-test to use, you&apos;ve already altered your Type I error rate. Just always use Welch&apos;s.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You&apos;re testing whether a new ML model has lower prediction error than the baseline. You run both models on the same 50 test cases. The baseline has mean absolute error 12.3 (std 4.1) and the new model has MAE 11.1 (std 3.8). Are the results significant? Which t-test do you use and why?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Paired t-test</strong> is correct because both models are evaluated on the <em>same</em> 50 test cases. The errors are correlated — a case that&apos;s hard for one model is likely hard for the other.</li>
          <li>Compute paired differences: <InlineMath math="D_i = \text{MAE}_{\text{baseline},i} - \text{MAE}_{\text{new},i}" /></li>
          <li>The mean difference is <InlineMath math="\bar{D} = 12.3 - 11.1 = 1.2" /></li>
          <li>We need the standard deviation of the <em>differences</em>, not the individual groups. Since the errors are positively correlated, <InlineMath math="s_D" /> will be smaller than you&apos;d expect from the individual stds. Suppose <InlineMath math="s_D = 2.5" />.</li>
          <li><InlineMath math="t = \frac{1.2}{2.5 / \sqrt{50}} = \frac{1.2}{0.354} = 3.39" /></li>
          <li>With <InlineMath math="df = 49" />, the p-value <InlineMath math="\approx 0.001" /> — significant improvement.</li>
          <li><strong>Key insight:</strong> An independent t-test would use the wrong standard error and likely fail to detect the difference. The paired test is much more powerful here.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Student (1908) &quot;The Probable Error of a Mean&quot;</strong> — William Sealy Gosset&apos;s original paper, published under the pseudonym &quot;Student&quot; because Guinness Brewery didn&apos;t allow employees to publish.</li>
          <li><strong>Ruxton (2006) &quot;The unequal variance t-test is an underused alternative to Student&apos;s t-test&quot;</strong> — Makes the case for always using Welch&apos;s test.</li>
          <li><strong>Dietterich (1998) &quot;Approximate Statistical Tests for Comparing Supervised Classification Algorithms&quot;</strong> — Essential reading for comparing ML models with proper statistical tests.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
