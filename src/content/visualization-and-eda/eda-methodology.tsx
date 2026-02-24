"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function EDAMethodology() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Exploratory Data Analysis (EDA) is the <strong>structured investigation of a dataset before
          you model it</strong>. The goal is to build intuition about what the data contains, what might
          be wrong with it, and what relationships exist. EDA is where you discover that half your
          &quot;age&quot; column is negative, that revenue and signups are nearly collinear, or that your
          target variable is heavily imbalanced.
        </p>
        <p>
          A disciplined EDA follows a repeatable order: <strong>univariate</strong> (examine each variable
          alone), then <strong>bivariate</strong> (relationships between pairs), then <strong>multivariate</strong>
          (interactions and higher-order structure). At each stage you&apos;re asking: What is the shape?
          Where are the outliers? What&apos;s missing? What needs transformation?
        </p>
        <p>
          EDA is not optional. Skipping it and jumping straight to modeling is the single most common
          reason ML projects fail. A model trained on garbage data produces garbage predictions, no
          matter how sophisticated the architecture.
        </p>
        <p>
          John Tukey, who coined the term in 1977, described EDA as &quot;detective work.&quot; You are
          looking for clues, not confirming hypotheses. Stay curious and let the data surprise you.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Quantifying Distribution Shape</h3>
        <p>
          For a random variable <InlineMath math="X" /> with mean <InlineMath math="\mu" /> and
          standard deviation <InlineMath math="\sigma" />, the standardized moments characterize
          distribution shape:
        </p>
        <BlockMath math="\text{Skewness} = E\!\left[\left(\frac{X - \mu}{\sigma}\right)^3\right] = \frac{\mu_3}{\sigma^3}" />
        <BlockMath math="\text{Kurtosis} = E\!\left[\left(\frac{X - \mu}{\sigma}\right)^4\right] = \frac{\mu_4}{\sigma^4}" />
        <p>
          Skewness measures asymmetry (positive = right tail, negative = left tail). Excess kurtosis
          <InlineMath math="= \text{Kurtosis} - 3" /> measures tail heaviness relative to a normal
          distribution. These guide transformation choices: high skewness suggests a log or Box-Cox
          transform.
        </p>

        <h3>Correlation Measures</h3>
        <p>Pearson&apos;s correlation captures linear dependence:</p>
        <BlockMath math="r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2 \sum_{i=1}^n (y_i - \bar{y})^2}}" />
        <p>
          For monotonic but nonlinear relationships, Spearman&apos;s rank correlation <InlineMath math="\rho_s" /> is
          more robust — it applies Pearson&apos;s formula to the <em>ranks</em> of the data rather than the
          raw values. For categorical associations, use Cram&eacute;r&apos;s V:
        </p>
        <BlockMath math="V = \sqrt{\frac{\chi^2 / n}{\min(r-1, c-1)}}" />
        <p>
          where <InlineMath math="\chi^2" /> is the chi-squared statistic from the contingency table
          with <InlineMath math="r" /> rows and <InlineMath math="c" /> columns.
        </p>

        <h3>Missingness Patterns</h3>
        <p>Data can be missing in three ways:</p>
        <ul>
          <li><strong>MCAR</strong> (Missing Completely At Random): Missingness is independent of all values. Safe to drop rows.</li>
          <li><strong>MAR</strong> (Missing At Random): Missingness depends on <em>observed</em> values. Can impute using other columns.</li>
          <li><strong>MNAR</strong> (Missing Not At Random): Missingness depends on the <em>missing value itself</em>. Requires domain-specific handling.</li>
        </ul>
      </TopicSection>

      <TopicSection type="code">
        <CodeBlock
          language="python"
          title="eda_step1_first_look.py"
          code={`import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

# ── Step 1: First Look — shape, types, memory ──
print(f"Shape: {df.shape}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(df.dtypes.value_counts())
print(df.head(3))

# ── Step 2: Missing data audit ──
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_report = pd.DataFrame({"count": missing, "pct": missing_pct})
print(missing_report[missing_report["count"] > 0].sort_values("pct", ascending=False))

# ── Step 3: Duplicates ──
n_dups = df.duplicated().sum()
print(f"Duplicate rows: {n_dups} ({n_dups / len(df) * 100:.1f}%)")

# ── Step 4: Numeric summary ──
print(df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T)

# ── Step 5: Categorical summary ──
for col in df.select_dtypes(include="object").columns:
    n_unique = df[col].nunique()
    top_val = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
    print(f"{col}: {n_unique} unique, most common = {top_val}")`}
        />

        <CodeBlock
          language="python"
          title="eda_step2_univariate.py"
          code={`import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

numeric_cols = df.select_dtypes(include=np.number).columns

# ── Histograms + KDE for every numeric column ──
n_cols = len(numeric_cols)
fig, axes = plt.subplots(
    (n_cols + 2) // 3, 3, figsize=(14, 4 * ((n_cols + 2) // 3))
)
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    data = df[col].dropna()

    ax.hist(data, bins=50, density=True, alpha=0.5, color="steelblue")
    if len(data) > 10:
        kde_x = np.linspace(data.min(), data.max(), 200)
        kde = stats.gaussian_kde(data)
        ax.plot(kde_x, kde(kde_x), color="coral", linewidth=2)

    skew = data.skew()
    kurt = data.kurtosis()  # excess kurtosis
    ax.set_title(f"{col}\\nskew={skew:.2f}, kurt={kurt:.2f}", fontsize=10)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.tight_layout()
plt.show()

# ── Flag columns that need transformation ──
for col in numeric_cols:
    skew = df[col].skew()
    if abs(skew) > 1:
        print(f"HIGH SKEW: {col} (skew={skew:.2f}) — consider log transform")`}
        />

        <CodeBlock
          language="python"
          title="eda_step3_bivariate.py"
          code={`# ── Correlation heatmap (numeric pairs) ──
corr = df[numeric_cols].corr(method="spearman")  # robust to outliers
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax, square=True)
ax.set_title("Spearman Correlation Matrix")
plt.tight_layout()
plt.show()

# ── Flag highly correlated pairs (potential multicollinearity) ──
threshold = 0.85
high_corr = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > threshold:
            high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
            print(f"HIGH CORR: {corr.columns[i]} vs {corr.columns[j]} = {corr.iloc[i, j]:.3f}")

# ── Target variable analysis ──
target = "price"  # adjust to your target
for col in numeric_cols:
    if col == target:
        continue
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(df[col], df[target], alpha=0.2, s=10)
    axes[0].set_xlabel(col)
    axes[0].set_ylabel(target)
    # Add lowess smoother for nonlinear trends
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothed = lowess(df[target].values, df[col].values, frac=0.3)
    axes[0].plot(smoothed[:, 0], smoothed[:, 1], color="red", linewidth=2)
    axes[0].set_title(f"{col} vs {target}")

    # Residuals from linear fit
    slope, intercept = np.polyfit(df[col].dropna(), df[target].dropna(), 1)
    residuals = df[target] - (slope * df[col] + intercept)
    axes[1].scatter(df[col], residuals, alpha=0.2, s=10)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title(f"Residuals (linear fit)")
    fig.tight_layout()
    plt.show()`}
        />

        <CodeBlock
          language="python"
          title="eda_step4_outliers.py"
          code={`# ── Outlier detection: IQR method ──
def flag_iqr_outliers(series, k=1.5):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - k * IQR, Q3 + k * IQR
    outliers = (series < lower) | (series > upper)
    return outliers, lower, upper

for col in numeric_cols:
    outliers, lo, hi = flag_iqr_outliers(df[col].dropna())
    n_out = outliers.sum()
    if n_out > 0:
        print(f"{col}: {n_out} outliers ({n_out/len(df)*100:.1f}%) "
              f"outside [{lo:.2f}, {hi:.2f}]")

# ── Boxplots for visual outlier detection ──
fig, ax = plt.subplots(figsize=(12, 5))
df[numeric_cols].boxplot(ax=ax, vert=False)
ax.set_title("Boxplot: Outlier Overview")
plt.tight_layout()
plt.show()`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>Follow a checklist every time</strong>: Shape and types, missing data, duplicates,
            univariate distributions, bivariate correlations, target analysis, outliers. A repeatable
            process prevents you from missing critical issues.
          </li>
          <li>
            <strong>Automate your EDA template</strong>: Libraries like <code>ydata-profiling</code>
            (formerly pandas-profiling) generate comprehensive HTML reports. Use them as a starting
            point, but always do manual deep dives on the most important features.
          </li>
          <li>
            <strong>Log-transform right-skewed features before modeling</strong>: Income, revenue, counts,
            and areas are almost always right-skewed. <InlineMath math="\log(x + 1)" /> often makes
            them approximately normal and improves linear model performance.
          </li>
          <li>
            <strong>Use Spearman over Pearson when in doubt</strong>: Pearson&apos;s <InlineMath math="r" /> only
            captures linear relationships and is sensitive to outliers. Spearman&apos;s <InlineMath math="\rho_s" />
            captures any monotonic relationship.
          </li>
          <li>
            <strong>Document findings as you go</strong>: Write a brief summary after each EDA phase. These notes
            directly inform feature engineering and model selection decisions.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Only looking at <code>.describe()</code></strong>: Summary statistics can hide bimodal
            distributions, clusters, and nonlinear patterns. Always plot histograms and scatter plots.
            Anscombe&apos;s quartet is the classic example — four datasets with identical means, variances,
            and correlations but completely different shapes.
          </li>
          <li>
            <strong>Dropping missing data without investigating why</strong>: If missingness is not
            random (MNAR), dropping rows introduces bias. For example, high-income respondents may
            skip the income question — dropping them skews your data toward lower incomes.
          </li>
          <li>
            <strong>Using the test set during EDA</strong>: Your EDA should only use the training set.
            If you examine the test set and make decisions based on it (feature selection,
            transformations), you&apos;ve leaked information and your evaluation will be optimistic.
          </li>
          <li>
            <strong>Ignoring feature interactions</strong>: Two features may each have weak correlation
            with the target but together be highly predictive. Plot interaction scatter plots and
            consider computing ratio or product features.
          </li>
          <li>
            <strong>Treating correlation as causation</strong>: EDA reveals associations, not causes.
            Ice cream sales and drowning rates are correlated — both are caused by hot weather.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> You receive a dataset with 50 features and 100,000 rows. You have
          one hour before a meeting where you need to present initial findings. Walk through your EDA
          process.
        </p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>First 5 minutes — shape and types</strong>: <code>df.shape</code>,
            <code>df.dtypes</code>, <code>df.head()</code>. Identify numeric vs. categorical vs.
            datetime. Check for ID columns that shouldn&apos;t be used as features.
          </li>
          <li>
            <strong>Next 5 minutes — missing data and duplicates</strong>: Compute missing percentages per
            column. Flag columns with more than 30% missing (may need to drop). Check for exact duplicate
            rows and understand why they exist.
          </li>
          <li>
            <strong>Next 15 minutes — univariate analysis</strong>: Run <code>df.describe()</code> with
            extended percentiles. Generate histograms for all numeric features. Look for: extreme skewness,
            impossible values (negative ages), low-variance features, suspicious spikes at specific values
            (e.g., 999 as a missing code).
          </li>
          <li>
            <strong>Next 15 minutes — target analysis</strong>: Plot the target distribution. Compute
            correlations with all features. Create scatter plots or box plots of the top 10 correlated
            features against the target.
          </li>
          <li>
            <strong>Next 10 minutes — bivariate and multicollinearity</strong>: Correlation heatmap. Flag
            pairs with <InlineMath math="|\rho| > 0.85" />. For the top features, check for nonlinear
            patterns using LOWESS smoothers.
          </li>
          <li>
            <strong>Final 10 minutes — summarize findings</strong>: List data quality issues,
            recommended transformations, features to drop, and promising features to explore further.
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>John Tukey, &quot;Exploratory Data Analysis&quot; (1977)</strong> — The original book that established EDA as a discipline.</li>
          <li><strong>Hadley Wickham &amp; Garrett Grolemund, &quot;R for Data Science&quot;</strong> — Chapters 3-7 on EDA are language-agnostic in their principles.</li>
          <li><strong>ydata-profiling</strong> — Automated EDA reports: <code>pip install ydata-profiling</code>.</li>
          <li><strong>Anscombe&apos;s Quartet and the Datasaurus Dozen</strong> — Classic demonstrations of why summary statistics alone are insufficient.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
