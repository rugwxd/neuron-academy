"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function TimeSeriesComponents() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Every time series is a combination of three ingredients: <strong>trend</strong> (long-term direction &mdash; is it going up, down, or flat?), <strong>seasonality</strong> (repeating patterns &mdash; weekly spikes, holiday surges, quarterly cycles), and <strong>residuals</strong> (the leftover noise that neither trend nor seasonality explains).
        </p>
        <p>
          <strong>Decomposition</strong> is how we separate these components to understand what actually drives the data. Think of it like separating a song into vocals, instruments, and background noise &mdash; each part tells you something different.
        </p>
        <p>
          Before you can forecast a time series, you need to understand its structure. Is there a trend? Is it seasonal? Is the seasonal pattern getting bigger over time (multiplicative) or staying the same size (additive)? Decomposition answers these questions.
        </p>
        <p>
          A related concept is <strong>stationarity</strong> &mdash; a time series is stationary if its statistical properties (mean, variance) don&apos;t change over time. Most forecasting models require stationarity, so you need to test for it and transform your data if it&apos;s not stationary.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Additive Decomposition</h3>
        <p>When the seasonal fluctuations are roughly constant over time:</p>
        <BlockMath math="y(t) = T(t) + S(t) + R(t)" />
        <p>where <InlineMath math="T(t)" /> is the trend component, <InlineMath math="S(t)" /> is the seasonal component, and <InlineMath math="R(t)" /> is the residual (remainder).</p>

        <h3>Multiplicative Decomposition</h3>
        <p>When seasonal fluctuations grow proportionally with the level of the series:</p>
        <BlockMath math="y(t) = T(t) \times S(t) \times R(t)" />
        <p>Equivalently, take logs to convert to additive: <InlineMath math="\log y(t) = \log T(t) + \log S(t) + \log R(t)" />.</p>

        <h3>Moving Average for Trend Extraction</h3>
        <p>A centered moving average of order <InlineMath math="m" /> smooths out the seasonal pattern:</p>
        <BlockMath math="\hat{T}(t) = \frac{1}{m} \sum_{j=-k}^{k} y(t+j), \quad m = 2k+1" />
        <p>For even-order seasonality (e.g., monthly data with period 12), use a <InlineMath math="2 \times 12" /> moving average.</p>

        <h3>STL Decomposition</h3>
        <p>STL (Seasonal and Trend decomposition using LOESS) uses locally weighted regression to iteratively extract the seasonal and trend components. It is more robust to outliers and allows the seasonal component to change over time.</p>

        <h3>Stationarity</h3>
        <p>A time series <InlineMath math="y_t" /> is <strong>weakly stationary</strong> if:</p>
        <BlockMath math="E[y_t] = \mu \quad \text{(constant mean)}" />
        <BlockMath math="\text{Var}(y_t) = \sigma^2 \quad \text{(constant variance)}" />
        <BlockMath math="\text{Cov}(y_t, y_{t+h}) = \gamma(h) \quad \text{(covariance depends only on lag } h \text{)}" />

        <h3>Augmented Dickey-Fuller (ADF) Test</h3>
        <p>Tests the null hypothesis that a unit root is present (series is non-stationary):</p>
        <BlockMath math="\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t" />
        <p>If the test statistic is more negative than the critical value, we reject the null and conclude the series is stationary. A small p-value (typically <InlineMath math="p < 0.05" />) means stationarity.</p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Time Series Decomposition from Scratch</h3>
        <CodeBlock
          language="python"
          title="decomposition_scratch.py"
          code={`import numpy as np
import matplotlib.pyplot as plt

def decompose_additive(y, period):
    """Decompose time series into trend, seasonal, residual (additive)."""
    n = len(y)

    # Step 1: Estimate trend with centered moving average
    trend = np.full(n, np.nan)
    half = period // 2
    for t in range(half, n - half):
        if period % 2 == 0:
            # 2xm MA for even period
            trend[t] = (0.5 * y[t - half] + np.sum(y[t - half + 1:t + half]) + 0.5 * y[t + half]) / period
        else:
            trend[t] = np.mean(y[t - half:t + half + 1])

    # Step 2: Detrend to get seasonal + residual
    detrended = y - trend

    # Step 3: Average each seasonal position to get seasonal component
    seasonal = np.zeros(n)
    for i in range(period):
        indices = range(i, n, period)
        vals = [detrended[j] for j in indices if not np.isnan(detrended[j])]
        season_avg = np.mean(vals) if vals else 0
        for j in indices:
            seasonal[j] = season_avg

    # Center the seasonal component (should sum to zero over one period)
    seasonal -= np.mean(seasonal[:period])

    # Step 4: Residual
    residual = y - trend - seasonal

    return trend, seasonal, residual

# Generate synthetic data: trend + seasonality + noise
np.random.seed(42)
n = 365
t = np.arange(n)
trend_true = 0.05 * t + 10
seasonal_true = 5 * np.sin(2 * np.pi * t / 30)  # monthly-ish cycle
noise = np.random.randn(n) * 1.5
y = trend_true + seasonal_true + noise

trend, seasonal, residual = decompose_additive(y, period=30)

fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
axes[0].plot(t, y, "k", linewidth=0.8)
axes[0].set_title("Original")
axes[1].plot(t, trend, "b", linewidth=1.2)
axes[1].set_title("Trend (Moving Average)")
axes[2].plot(t, seasonal, "g", linewidth=1.0)
axes[2].set_title("Seasonal")
axes[3].plot(t, residual, "r", linewidth=0.6, alpha=0.7)
axes[3].set_title("Residual")
plt.tight_layout()
plt.show()`}
        />

        <h3>statsmodels seasonal_decompose and STL</h3>
        <CodeBlock
          language="python"
          title="decomposition_statsmodels.py"
          code={`import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL

# Create monthly data with trend and seasonality
np.random.seed(42)
dates = pd.date_range("2018-01-01", periods=72, freq="MS")
trend = np.linspace(100, 200, 72)
seasonal = 15 * np.sin(2 * np.pi * np.arange(72) / 12)
noise = np.random.randn(72) * 5
y = pd.Series(trend + seasonal + noise, index=dates)

# --- Classical decomposition ---
result_add = seasonal_decompose(y, model="additive", period=12)
result_add.plot()
plt.suptitle("Classical Additive Decomposition")
plt.tight_layout()
plt.show()

# For multiplicative (when amplitude grows with level):
# result_mul = seasonal_decompose(y, model="multiplicative", period=12)

# --- STL decomposition (more robust) ---
stl = STL(y, period=12, robust=True)
result_stl = stl.fit()
result_stl.plot()
plt.suptitle("STL Decomposition (LOESS-based)")
plt.tight_layout()
plt.show()

# Access individual components
print("Trend (first 5):", result_stl.trend.head().values)
print("Seasonal (first 5):", result_stl.seasonal.head().values)
print("Residual (first 5):", result_stl.resid.head().values)`}
        />

        <h3>Stationarity Testing, Differencing, and Log Transforms</h3>
        <CodeBlock
          language="python"
          title="stationarity_testing.py"
          code={`import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def adf_test(series, name=""):
    """Run Augmented Dickey-Fuller test and print results."""
    result = adfuller(series.dropna(), autolag="AIC")
    print(f"--- ADF Test: {name} ---")
    print(f"  Test Statistic: {result[0]:.4f}")
    print(f"  p-value:        {result[1]:.4f}")
    print(f"  Lags Used:      {result[2]}")
    for key, val in result[4].items():
        print(f"  Critical ({key}): {val:.4f}")
    stationary = result[1] < 0.05
    print(f"  Conclusion:     {'STATIONARY' if stationary else 'NON-STATIONARY'}\\n")
    return stationary

# Create non-stationary data (random walk + trend)
np.random.seed(42)
n = 500
random_walk = np.cumsum(np.random.randn(n)) + 100
trend = 0.1 * np.arange(n)
y = pd.Series(random_walk + trend)

# Test 1: Original series (should be non-stationary)
adf_test(y, "Original Series")

# Fix 1: First differencing removes trend/unit root
y_diff = y.diff()
adf_test(y_diff, "First Difference")

# Fix 2: Log transform (stabilizes variance for multiplicative data)
y_positive = y - y.min() + 1  # ensure positive
y_log = np.log(y_positive)
adf_test(y_log, "Log Transform")

# Fix 3: Log + differencing (common combo)
y_log_diff = y_log.diff()
adf_test(y_log_diff, "Log + First Difference")

# Warning: Don't over-difference!
y_diff2 = y_diff.diff()
adf_test(y_diff2, "Second Difference (be careful!)")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Always plot the raw data first</strong> &mdash; before any modeling, visually inspect for trend, seasonality, outliers, and structural breaks. A plot tells you more than any test.</li>
          <li><strong>Use multiplicative decomposition when seasonal amplitude grows</strong> &mdash; if seasonal swings get bigger as the level increases (e.g., retail sales), use multiplicative. If they stay constant, use additive.</li>
          <li><strong>Quick rule for choosing decomposition type</strong> &mdash; plot the data. If the &quot;envelope&quot; of seasonal fluctuations widens over time, it&apos;s multiplicative. Alternatively, take the log and see if the additive decomposition looks better.</li>
          <li><strong>Differencing for stationarity</strong> &mdash; one difference usually removes a linear trend. Seasonal differencing (lag = period) removes seasonal patterns. Rarely need more than two differences total.</li>
          <li><strong>Log transform for multiplicative seasonality</strong> &mdash; <InlineMath math="\log(T \times S \times R) = \log T + \log S + \log R" />, converting multiplicative to additive.</li>
          <li><strong>STL over classical decomposition</strong> &mdash; STL handles outliers better, allows seasonal patterns to evolve, and works for any period. Use <code>robust=True</code> when outliers are present.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Assuming stationarity without testing</strong> &mdash; many time series (stock prices, GDP, temperature) are non-stationary. Always run the ADF test before fitting ARIMA-type models.</li>
          <li><strong>Wrong decomposition type</strong> &mdash; using additive decomposition on multiplicative data produces residuals that grow over time. Check if residual variance is constant.</li>
          <li><strong>Confusing trend with drift</strong> &mdash; a deterministic trend (<InlineMath math="y_t = \alpha + \beta t + \varepsilon_t" />) is fixed and predictable; drift (<InlineMath math="y_t = y_{t-1} + c + \varepsilon_t" />) is a random walk with a constant. They look similar but require different treatments.</li>
          <li><strong>Over-differencing</strong> &mdash; differencing a stationary series introduces artificial autocorrelation. If ADF already says stationary, don&apos;t difference. Check the ACF of the differenced series &mdash; a large negative spike at lag 1 suggests over-differencing.</li>
          <li><strong>Ignoring structural breaks</strong> &mdash; a sudden change in the data-generating process (e.g., COVID, policy change) can fool decomposition methods. Consider splitting the series at the break point.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Given a retail sales time series, walk through your decomposition approach.</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Plot the raw data.</strong> I&apos;d first visualize the series looking for upward/downward trend, seasonal patterns (weekly, monthly, yearly), outliers, and any structural breaks (like COVID or store openings).</li>
          <li><strong>Identify the decomposition type.</strong> If seasonal swings grow proportionally with sales volume (likely for retail), I&apos;d use multiplicative decomposition. If they&apos;re roughly constant, additive.</li>
          <li><strong>Apply STL decomposition</strong> with <code>robust=True</code> since retail data often has outliers (Black Friday spikes). I&apos;d set the period to 12 for monthly data (or 52 for weekly).</li>
          <li><strong>Examine each component.</strong> Trend: is growth accelerating or decelerating? Seasonal: which months/weeks peak? Residual: are there unexplained spikes that correspond to promotions or events?</li>
          <li><strong>Test stationarity</strong> on the residuals with ADF. If non-stationary, there may be additional structure to model (e.g., autoregressive behavior in the residuals).</li>
          <li><strong>Transform if needed.</strong> For multiplicative data, apply log transform to make it additive. Difference to remove remaining trend if pursuing ARIMA-type modeling.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Hyndman &amp; Athanasopoulos &quot;Forecasting: Principles and Practice&quot;</strong> &mdash; The gold standard free online textbook. Chapters 3 and 5 cover decomposition and stationarity beautifully.</li>
          <li><strong>Cleveland et al. (1990) &quot;STL: A Seasonal-Trend Decomposition Procedure Based on Loess&quot;</strong> &mdash; The original STL paper. Explains the iterative LOESS algorithm in detail.</li>
          <li><strong>Augmented Dickey-Fuller test</strong> &mdash; Said &amp; Dickey (1984). The standard unit root test with its extensions for serial correlation.</li>
          <li><strong>statsmodels documentation on time series</strong> &mdash; Practical guide to decomposition, stationarity tests, and transformations in Python.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
