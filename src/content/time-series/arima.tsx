"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function ArimaSarima() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          ARIMA predicts the future from the past using three simple ideas combined:
        </p>
        <ul>
          <li><strong>Autoregression (AR)</strong> &mdash; past values predict future values. If today&apos;s temperature is 70&deg;F, tomorrow&apos;s is probably close to 70 too. The further back you look, the more context you have.</li>
          <li><strong>Integration (I)</strong> &mdash; differencing the data to remove trends and make it stationary. Instead of predicting the price, predict the <em>change</em> in price.</li>
          <li><strong>Moving Average (MA)</strong> &mdash; past forecast <em>errors</em> help correct future predictions. If you consistently overestimated last week, adjust downward this week.</li>
        </ul>
        <p>
          <strong>SARIMA</strong> adds seasonal versions of each component. If you&apos;re predicting monthly airline passengers, SARIMA captures both the general trend and the fact that July is always a peak month.
        </p>
        <p>
          The <strong>Box-Jenkins methodology</strong> is the classic framework: (1) identify tentative model orders by looking at ACF/PACF plots, (2) estimate the parameters, (3) check diagnostics (are residuals white noise?). If not, iterate.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Autoregressive Model AR(p)</h3>
        <p>The current value depends on the previous <InlineMath math="p" /> values:</p>
        <BlockMath math="y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \varepsilon_t" />
        <p>where <InlineMath math="\varepsilon_t" /> is white noise and <InlineMath math="\phi_1, \ldots, \phi_p" /> are the autoregressive coefficients.</p>

        <h3>Moving Average Model MA(q)</h3>
        <p>The current value depends on the previous <InlineMath math="q" /> forecast errors:</p>
        <BlockMath math="y_t = c + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \cdots + \theta_q \varepsilon_{t-q}" />

        <h3>ARIMA(p, d, q)</h3>
        <p>Combine AR and MA on the <InlineMath math="d" />-times differenced series. If <InlineMath math="d=1" />, we model <InlineMath math="\Delta y_t = y_t - y_{t-1}" />:</p>
        <BlockMath math="\Delta^d y_t = c + \sum_{i=1}^{p} \phi_i \Delta^d y_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j} + \varepsilon_t" />

        <h3>SARIMA(p, d, q)(P, D, Q)<sub>m</sub></h3>
        <p>Adds seasonal AR, differencing, and MA with period <InlineMath math="m" />:</p>
        <BlockMath math="\Phi(B^m)\,\phi(B)\,\Delta^d \Delta_m^D\, y_t = c + \Theta(B^m)\,\theta(B)\,\varepsilon_t" />
        <p>where <InlineMath math="B" /> is the backshift operator (<InlineMath math="B y_t = y_{t-1}" />), <InlineMath math="\Delta_m^D" /> is seasonal differencing, and uppercase <InlineMath math="\Phi, \Theta" /> are the seasonal polynomials.</p>

        <h3>Model Selection: AIC and BIC</h3>
        <BlockMath math="\text{AIC} = -2 \ln(\hat{L}) + 2k" />
        <BlockMath math="\text{BIC} = -2 \ln(\hat{L}) + k \ln(n)" />
        <p>where <InlineMath math="\hat{L}" /> is the maximized likelihood, <InlineMath math="k" /> is the number of parameters, and <InlineMath math="n" /> is the number of observations. Lower is better. BIC penalizes complexity more heavily than AIC.</p>

        <h3>Box-Jenkins Methodology</h3>
        <ol>
          <li><strong>Identify:</strong> Use ACF/PACF plots to determine <InlineMath math="p, d, q" />.</li>
          <li><strong>Estimate:</strong> Fit the model via maximum likelihood.</li>
          <li><strong>Diagnose:</strong> Check residuals are white noise (Ljung-Box test, ACF of residuals).</li>
        </ol>
      </TopicSection>

      <TopicSection type="code">
        <h3>AR Model from Scratch (OLS Fitting)</h3>
        <CodeBlock
          language="python"
          title="ar_from_scratch.py"
          code={`import numpy as np

class ARModel:
    """Autoregressive model fitted via OLS."""

    def __init__(self, p):
        self.p = p
        self.coeffs = None
        self.intercept = None

    def _build_lag_matrix(self, y):
        """Create matrix of lagged values."""
        n = len(y)
        X = np.zeros((n - self.p, self.p))
        for i in range(self.p):
            X[:, i] = y[self.p - i - 1 : n - i - 1]
        target = y[self.p:]
        return X, target

    def fit(self, y):
        """Fit AR(p) model using OLS: y_t = c + phi_1*y_{t-1} + ... + phi_p*y_{t-p}."""
        X, target = self._build_lag_matrix(y)
        # Add intercept column
        X_b = np.column_stack([np.ones(len(X)), X])
        # OLS solution
        params = np.linalg.lstsq(X_b, target, rcond=None)[0]
        self.intercept = params[0]
        self.coeffs = params[1:]
        return self

    def predict(self, y, steps=1):
        """Forecast 'steps' ahead."""
        history = list(y[-self.p:])
        forecasts = []
        for _ in range(steps):
            lags = np.array(history[-self.p:][::-1])  # most recent first
            y_hat = self.intercept + self.coeffs @ lags
            forecasts.append(y_hat)
            history.append(y_hat)
        return np.array(forecasts)

# Example: AR(2) process
np.random.seed(42)
n = 500
y = np.zeros(n)
for t in range(2, n):
    y[t] = 0.6 * y[t-1] - 0.2 * y[t-2] + np.random.randn() * 0.5

model = ARModel(p=2)
model.fit(y)
print(f"Fitted coefficients: {model.coeffs}")  # should be close to [0.6, -0.2]
print(f"Intercept: {model.intercept:.4f}")

forecast = model.predict(y, steps=5)
print(f"Next 5 predictions: {forecast}")`}
        />

        <h3>ARIMA with statsmodels &mdash; AIC Grid Search</h3>
        <CodeBlock
          language="python"
          title="arima_grid_search.py"
          code={`import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from itertools import product

warnings.filterwarnings("ignore")

# Generate non-stationary data (random walk + AR behavior)
np.random.seed(42)
n = 300
y_raw = np.cumsum(np.random.randn(n) * 0.5) + 50
y = pd.Series(y_raw, index=pd.date_range("2020-01-01", periods=n, freq="D"))

# Step 1: Determine d (differencing order)
def find_d(series, max_d=2):
    for d in range(max_d + 1):
        s = series.copy()
        for _ in range(d):
            s = s.diff().dropna()
        pval = adfuller(s, autolag="AIC")[1]
        if pval < 0.05:
            return d
    return max_d

d = find_d(y)
print(f"Differencing order d = {d}")

# Step 2: Grid search over p and q
best_aic = np.inf
best_order = None
best_model = None

for p, q in product(range(5), range(5)):
    try:
        model = ARIMA(y, order=(p, d, q))
        fitted = model.fit()
        if fitted.aic < best_aic:
            best_aic = fitted.aic
            best_order = (p, d, q)
            best_model = fitted
    except Exception:
        continue

print(f"Best ARIMA order: {best_order}, AIC: {best_aic:.2f}")
print(best_model.summary().tables[1])

# Step 3: Forecast
forecast = best_model.get_forecast(steps=30)
pred = forecast.predicted_mean
ci = forecast.conf_int()
print(f"\\nNext 5 day forecast: {pred.values[:5]}")`}
        />

        <h3>SARIMA on Airline Passengers</h3>
        <CodeBlock
          language="python"
          title="sarima_airline.py"
          code={`import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Classic airline passengers dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, parse_dates=["Month"], index_col="Month")
y = df["Passengers"]

# Train/test split (NEVER random for time series!)
train = y[:"1958"]
test = y["1959":]

# Plot ACF and PACF of seasonally differenced data
y_sdiff = train.diff(12).dropna()  # seasonal differencing
y_sdiff_diff = y_sdiff.diff().dropna()  # + first difference

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(y_sdiff_diff, lags=36, ax=axes[0])
plot_pacf(y_sdiff_diff, lags=36, ax=axes[1])
plt.tight_layout()
plt.show()

# Fit SARIMA(0,1,1)(0,1,1)_12 (classic airline model)
model = SARIMAX(
    train,
    order=(0, 1, 1),
    seasonal_order=(0, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
fitted = model.fit(disp=False)
print(fitted.summary())

# Diagnostics: residuals should be white noise
residuals = fitted.resid
lb_test = acorr_ljungbox(residuals, lags=[12, 24], return_df=True)
print("\\nLjung-Box test (p > 0.05 means white noise):")
print(lb_test)

# Forecast with confidence intervals
forecast = fitted.get_forecast(steps=len(test))
pred = forecast.predicted_mean
ci = forecast.conf_int()

plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Test", color="gray")
plt.plot(pred.index, pred, label="Forecast", color="red")
plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.2, color="red")
plt.legend()
plt.title("SARIMA(0,1,1)(0,1,1)_12 Forecast")
plt.show()

# Walk-forward validation (the right way to evaluate)
from sklearn.metrics import mean_absolute_error

history = list(train)
predictions = []
for t in range(len(test)):
    model = SARIMAX(history, order=(0,1,1), seasonal_order=(0,1,1,12))
    fit = model.fit(disp=False)
    yhat = fit.forecast(steps=1)[0]
    predictions.append(yhat)
    history.append(test.iloc[t])

mae = mean_absolute_error(test, predictions)
print(f"\\nWalk-forward MAE: {mae:.2f}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Box-Jenkins methodology</strong> &mdash; (1) Plot the data and check stationarity, (2) difference if needed, (3) examine ACF/PACF to identify model orders, (4) fit the model, (5) check residual diagnostics. Repeat if residuals show structure.</li>
          <li>
            <strong>ACF/PACF interpretation cheat sheet:</strong>
            <ul>
              <li>PACF cuts off after lag <InlineMath math="p" />, ACF decays &rarr; AR(p)</li>
              <li>ACF cuts off after lag <InlineMath math="q" />, PACF decays &rarr; MA(q)</li>
              <li>Both decay &rarr; ARMA(p, q), use AIC to choose orders</li>
              <li>Significant spike at seasonal lag in ACF &rarr; seasonal component needed</li>
            </ul>
          </li>
          <li><strong>Use auto_arima from pmdarima</strong> &mdash; in practice, manual ACF/PACF reading is error-prone. <code>pmdarima.auto_arima()</code> automates the grid search with stepwise AIC optimization.</li>
          <li><strong>Walk-forward validation</strong> &mdash; never use random train/test splits for time series. Train on data up to time <InlineMath math="t" />, predict <InlineMath math="t+1" />, expand training set, repeat.</li>
          <li><strong>Start simple</strong> &mdash; ARIMA(1,1,1) or ARIMA(0,1,1) are surprisingly good baselines. Only add complexity if diagnostics demand it.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Not checking stationarity first</strong> &mdash; fitting ARIMA on non-stationary data without differencing produces spurious results. Always run ADF before choosing <InlineMath math="p" /> and <InlineMath math="q" />.</li>
          <li><strong>Using random train/test split</strong> &mdash; this is the single most common mistake. Time series data has temporal ordering. Random splits leak future information into training and give unrealistically good results.</li>
          <li><strong>Overfitting with too many parameters</strong> &mdash; ARIMA(5,2,5) almost certainly overfits. Prefer parsimonious models. If AIC keeps improving with higher orders, something is wrong (likely non-stationarity or structural breaks).</li>
          <li><strong>Ignoring residual diagnostics</strong> &mdash; after fitting, check: (1) ACF of residuals shows no significant spikes, (2) Ljung-Box test p-value is above 0.05, (3) residuals look normally distributed. If any fail, the model is misspecified.</li>
          <li><strong>Forecasting too far ahead</strong> &mdash; ARIMA forecasts revert to the mean. Confidence intervals widen rapidly. Don&apos;t trust forecasts beyond a few seasonal cycles.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> ACF shows significant spikes at lag 1 and lag 12, PACF cuts off after lag 1. What model do you suggest?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>PACF cuts off after lag 1 &rarr; the non-seasonal component is <strong>AR(1)</strong>.</li>
          <li>ACF significant at lag 12 &rarr; there&apos;s a <strong>seasonal component with period 12</strong> (monthly data).</li>
          <li>ACF spike at lag 1 is consistent with the AR(1) part decaying. The lag-12 spike suggests a seasonal MA or AR term.</li>
          <li>I&apos;d start with <strong>SARIMA(1, d, 0)(0, D, 1)<sub>12</sub></strong> &mdash; AR(1) for the non-seasonal part, seasonal MA(1) for the lag-12 behavior. The <InlineMath math="d" /> and <InlineMath math="D" /> depend on stationarity tests.</li>
          <li>I&apos;d verify by fitting a few nearby models (e.g., SARIMA(1,1,0)(1,1,0)<sub>12</sub>) and comparing AIC. Then I&apos;d check residual diagnostics with the Ljung-Box test.</li>
          <li>If this is a real interview, I&apos;d also mention that in practice I&apos;d use <code>pmdarima.auto_arima()</code> with <code>seasonal=True, m=12</code> to confirm.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Box &amp; Jenkins &quot;Time Series Analysis: Forecasting and Control&quot;</strong> &mdash; The original text that defined the ARIMA methodology. Dense but foundational.</li>
          <li><strong>Hyndman &amp; Athanasopoulos FPP Chapter 9</strong> &mdash; The most accessible modern treatment of ARIMA. Free online at otexts.com/fpp3.</li>
          <li><strong>pmdarima documentation</strong> &mdash; Practical auto_arima guide with examples. The stepwise algorithm is much faster than exhaustive grid search.</li>
          <li><strong>Ljung-Box test</strong> &mdash; Ljung &amp; Box (1978). Essential for residual diagnostics &mdash; tests whether autocorrelations in residuals are jointly zero.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
