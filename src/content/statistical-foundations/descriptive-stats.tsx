"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function DescriptiveStatistics() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Descriptive statistics are the <strong>numbers that summarize a dataset</strong>. Before you
          build models, run experiments, or make predictions, you need to understand what you&apos;re
          working with. Descriptive stats answer three fundamental questions: Where is the data
          centered? How spread out is it? What shape does the distribution have?
        </p>
        <p>
          <strong>Measures of central tendency</strong> (mean, median, mode) tell you where the
          &quot;typical&quot; value lives. The <em>mean</em> is the balance point — it minimizes the sum
          of squared deviations. The <em>median</em> is the middle value when sorted — it minimizes the
          sum of absolute deviations. The <em>mode</em> is the most frequent value. For symmetric
          distributions, these three are equal. For skewed distributions, they diverge, and choosing the
          wrong one can be misleading.
        </p>
        <p>
          <strong>Measures of spread</strong> (variance, standard deviation, IQR, range) tell you how
          much values differ from the center. A dataset of all identical values has zero spread. Real
          data always has variability, and quantifying it is essential for understanding uncertainty.
        </p>
        <p>
          <strong>Measures of shape</strong> (skewness, kurtosis) describe the asymmetry and tail
          behavior of the distribution. These determine whether standard methods (which often assume
          normality) will work, or whether you need robust alternatives.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Measures of Central Tendency</h3>
        <p>For a sample <InlineMath math="x_1, x_2, \ldots, x_n" />:</p>
        <BlockMath math="\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i \quad \text{(sample mean)}" />
        <p>
          The <strong>median</strong> is the value <InlineMath math="m" /> such that at least half the
          data lies at or below <InlineMath math="m" /> and at least half lies at or above. For sorted
          data <InlineMath math="x_{(1)} \leq x_{(2)} \leq \cdots \leq x_{(n)}" />:
        </p>
        <BlockMath math="\text{median} = \begin{cases} x_{((n+1)/2)} & \text{if } n \text{ is odd} \\ \frac{1}{2}\left(x_{(n/2)} + x_{(n/2+1)}\right) & \text{if } n \text{ is even} \end{cases}" />
        <p>
          The <strong>weighted mean</strong> generalizes the mean when observations have different
          importances: <InlineMath math="\bar{x}_w = \frac{\sum w_i x_i}{\sum w_i}" />.
        </p>

        <h3>Measures of Spread</h3>
        <BlockMath math="s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2 \quad \text{(sample variance)}" />
        <p>
          We divide by <InlineMath math="n-1" /> (Bessel&apos;s correction) rather than <InlineMath math="n" /> because
          the sample mean <InlineMath math="\bar{x}" /> is estimated from the same data, consuming one degree of
          freedom. This makes <InlineMath math="s^2" /> an <strong>unbiased estimator</strong> of
          the population variance <InlineMath math="\sigma^2" />:
        </p>
        <BlockMath math="E[s^2] = \sigma^2" />
        <p>Standard deviation is the square root: <InlineMath math="s = \sqrt{s^2}" />. It has the same units as the data, making it more interpretable than variance.</p>

        <p>The <strong>Interquartile Range (IQR)</strong> is a robust measure of spread:</p>
        <BlockMath math="\text{IQR} = Q_3 - Q_1" />
        <p>
          where <InlineMath math="Q_1" /> and <InlineMath math="Q_3" /> are the 25th and 75th
          percentiles. The IQR is unaffected by outliers, unlike the standard deviation.
        </p>

        <h3>Measures of Shape</h3>
        <p><strong>Skewness</strong> measures asymmetry. The sample skewness is:</p>
        <BlockMath math="g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^3" />
        <ul>
          <li><InlineMath math="g_1 > 0" />: right-skewed (long right tail — e.g., income, home prices)</li>
          <li><InlineMath math="g_1 < 0" />: left-skewed (long left tail — e.g., exam scores near 100%)</li>
          <li><InlineMath math="g_1 = 0" />: symmetric (e.g., normal distribution)</li>
        </ul>

        <p><strong>Kurtosis</strong> measures tail heaviness. The sample excess kurtosis is:</p>
        <BlockMath math="g_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}" />
        <ul>
          <li><InlineMath math="g_2 > 0" /> (leptokurtic): heavier tails than normal — more extreme outliers</li>
          <li><InlineMath math="g_2 < 0" /> (platykurtic): lighter tails than normal — fewer extremes</li>
          <li><InlineMath math="g_2 = 0" /> (mesokurtic): tails like a normal distribution</li>
        </ul>

        <h3>Coefficient of Variation</h3>
        <p>
          When comparing spread across variables with different units or scales, use the coefficient
          of variation:
        </p>
        <BlockMath math="CV = \frac{s}{\bar{x}} \times 100\%" />
        <p>
          This is a dimensionless ratio. A height variable with <InlineMath math="CV = 5\%" /> has less
          relative variability than a weight variable with <InlineMath math="CV = 15\%" />, even if the
          raw standard deviations suggest otherwise.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <CodeBlock
          language="python"
          title="descriptive_stats_core.py"
          code={`import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

# ── Generate example data ──
income = np.random.lognormal(mean=10.5, sigma=0.8, size=5000)  # right-skewed
scores = 100 - np.random.exponential(scale=10, size=5000)       # left-skewed
scores = np.clip(scores, 0, 100)
heights = np.random.normal(loc=170, scale=8, size=5000)          # symmetric

for name, data in [("Income", income), ("Scores", scores), ("Heights", heights)]:
    print(f"\\n=== {name} ===")
    print(f"  Mean:     {np.mean(data):>12.2f}")
    print(f"  Median:   {np.median(data):>12.2f}")
    print(f"  Std Dev:  {np.std(data, ddof=1):>12.2f}")  # ddof=1 for sample
    print(f"  Variance: {np.var(data, ddof=1):>12.2f}")
    print(f"  IQR:      {np.percentile(data, 75) - np.percentile(data, 25):>12.2f}")
    print(f"  Skewness: {stats.skew(data):>12.2f}")
    print(f"  Kurtosis: {stats.kurtosis(data):>12.2f}")  # excess kurtosis
    print(f"  CV:       {np.std(data, ddof=1) / np.mean(data) * 100:>11.1f}%")
    print(f"  Range:    [{np.min(data):.2f}, {np.max(data):.2f}]")

# Income:  mean >> median (right-skewed, skewness > 0)
# Scores:  mean << median (left-skewed, skewness < 0)
# Heights: mean ≈ median  (symmetric, skewness ≈ 0)`}
        />

        <CodeBlock
          language="python"
          title="robust_statistics.py"
          code={`import numpy as np
from scipy import stats

# ── Mean vs Median: robustness to outliers ──
data = [10, 12, 11, 13, 12, 11, 14, 12, 13, 10000]  # one outlier

print(f"Mean:      {np.mean(data):.1f}")     # 1010.8 — destroyed by outlier
print(f"Median:    {np.median(data):.1f}")   # 12.0   — unaffected
print(f"Trim mean: {stats.trim_mean(data, 0.1):.1f}")  # 12.0 — trim 10% each tail

# ── Std Dev vs MAD: robust spread ──
# Median Absolute Deviation: MAD = median(|x_i - median(x)|)
mad = stats.median_abs_deviation(data)
print(f"Std Dev: {np.std(data, ddof=1):.1f}")  # ~3156 — dominated by outlier
print(f"MAD:     {mad:.1f}")                    # ~1.0 — robust

# ── When to use what ──
# Use mean + std when:  data is roughly symmetric, no extreme outliers
# Use median + IQR when: data is skewed or has outliers
# Use trimmed mean when: you want a compromise between robustness and efficiency

# ── Pandas makes this easy ──
import pandas as pd
df = pd.DataFrame({"value": data})
print(df["value"].describe())  # gives count, mean, std, min, 25%, 50%, 75%, max`}
        />

        <CodeBlock
          language="python"
          title="grouped_descriptive_stats.py"
          code={`import pandas as pd
import numpy as np

# ── Real-world pattern: grouped descriptive stats ──
np.random.seed(42)
df = pd.DataFrame({
    "department": np.random.choice(["Engineering", "Sales", "Marketing"], 300),
    "salary": np.random.lognormal(11.0, 0.4, 300),
    "tenure_years": np.random.exponential(4, 300),
})

# Per-group summary
summary = df.groupby("department").agg(
    count=("salary", "size"),
    mean_salary=("salary", "mean"),
    median_salary=("salary", "median"),
    std_salary=("salary", "std"),
    p25_salary=("salary", lambda x: x.quantile(0.25)),
    p75_salary=("salary", lambda x: x.quantile(0.75)),
    mean_tenure=("tenure_years", "mean"),
).round(0)

print(summary)

# ── Check if mean >> median (indicates right-skew) ──
for dept in df["department"].unique():
    subset = df[df["department"] == dept]["salary"]
    ratio = subset.mean() / subset.median()
    print(f"{dept}: mean/median ratio = {ratio:.3f} "
          f"({'skewed' if ratio > 1.1 else 'roughly symmetric'})")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>Always report median with skewed data</strong>: Median household income is about $75K,
            while mean household income is about $105K in the US. The mean is pulled up by billionaires.
            Reporting only the mean misrepresents the typical experience.
          </li>
          <li>
            <strong>Use IQR for outlier detection</strong>: A common rule flags values below
            <InlineMath math="Q_1 - 1.5 \times \text{IQR}" /> or above
            <InlineMath math="Q_3 + 1.5 \times \text{IQR}" /> as outliers. This is what box plots use.
          </li>
          <li>
            <strong>Check for <InlineMath math="n-1" /> vs <InlineMath math="n" /></strong>: NumPy&apos;s
            <code>np.std()</code> uses <InlineMath math="n" /> by default (population formula). For sample
            statistics, always pass <code>ddof=1</code>. Pandas uses <code>ddof=1</code> by default.
          </li>
          <li>
            <strong>Kurtosis reveals tail risk</strong>: In finance, high kurtosis means extreme events
            (crashes, spikes) happen more often than a normal model predicts. This is why VaR models
            based on normality underestimate risk.
          </li>
          <li>
            <strong>Use the coefficient of variation to compare variability across scales</strong>: You
            cannot compare the standard deviation of heights (in cm) directly to the standard deviation
            of weights (in kg). CV normalizes by the mean.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Reporting mean for skewed data</strong>: &quot;Average salary at our company is $200K&quot;
            when the CEO makes $5M and most employees make $80K. Always pair the mean with the median
            to show skewness, or report the median alone.
          </li>
          <li>
            <strong>Forgetting Bessel&apos;s correction</strong>: Using <InlineMath math="n" /> instead of
            <InlineMath math="n - 1" /> in the variance formula systematically underestimates the
            population variance. For large <InlineMath math="n" /> the difference is negligible, but for
            small samples it matters significantly.
          </li>
          <li>
            <strong>Confusing standard deviation with standard error</strong>: Standard deviation
            (<InlineMath math="s" />) measures the spread of <em>individual observations</em>. Standard
            error (<InlineMath math="s / \sqrt{n}" />) measures the precision of the <em>sample mean</em>.
            They answer different questions.
          </li>
          <li>
            <strong>Treating kurtosis as &quot;peakedness&quot;</strong>: This is a widespread misconception.
            Kurtosis is about <em>tail weight</em>, not the height of the peak. A distribution can be
            flat-topped and still have high kurtosis if its tails are heavy.
          </li>
          <li>
            <strong>Comparing standard deviations across different scales</strong>: A standard deviation
            of 10 for temperature (range 0-40) is very different from 10 for income (range 0-1M).
            Use the coefficient of variation for cross-variable comparisons.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> A dataset has mean 100, median 60, and standard deviation 150.
          What can you infer about the distribution? What descriptive statistics would you report, and
          what transformations would you consider?
        </p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Shape inference</strong>: Mean (100) is much larger than median (60), indicating
            <strong>strong right skew</strong>. A few very large values are pulling the mean up. The
            large standard deviation (150) relative to the mean suggests high variability and possibly
            extreme outliers.
          </li>
          <li>
            <strong>What to report</strong>: I would report median (60) as the measure of center since
            it better represents the typical value. For spread, I would report the IQR rather than the
            standard deviation because the latter is inflated by the same outliers that inflate the mean.
            I would also report the 5th and 95th percentiles to characterize the range.
          </li>
          <li>
            <strong>Transformations</strong>: For modeling, I would consider a <strong>log transform</strong>
            (<InlineMath math="\log(x + 1)" />), which compresses the right tail and often makes
            right-skewed data approximately normal. Alternatively, a <strong>Box-Cox transform</strong>
            finds the optimal power transformation: <InlineMath math="y^{(\lambda)} = \frac{y^\lambda - 1}{\lambda}" />.
            If the data includes zeros, Yeo-Johnson is preferred over Box-Cox.
          </li>
          <li>
            <strong>Further investigation</strong>: I would plot a histogram to check if the distribution
            is unimodal or multimodal (e.g., maybe there are two distinct populations). I would also examine
            the maximum value — with <InlineMath math="\text{mean} = 100" /> and <InlineMath math="s = 150" />,
            there may be values exceeding 500+ that deserve scrutiny.
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Freedman, Pisani &amp; Purves, &quot;Statistics&quot; (4th ed.)</strong> — The best intuitive introduction to descriptive statistics with minimal formulas.</li>
          <li><strong>Robust Statistics (Huber &amp; Ronchetti)</strong> — Theory behind why median and MAD are more robust than mean and standard deviation.</li>
          <li><strong>Westfall (2014), &quot;Kurtosis as Peakedness, 1905-2014. R.I.P.&quot;</strong> — Definitive paper debunking the peakedness interpretation of kurtosis.</li>
          <li><strong>scipy.stats module</strong> — Documentation covers all descriptive statistics functions with examples.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
