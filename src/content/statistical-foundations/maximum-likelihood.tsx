"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function MaximumLikelihoodEstimation() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          <strong>Maximum Likelihood Estimation (MLE)</strong> answers a simple question: given the data
          I observed, what parameter values make that data most probable? You pick the parameters that
          maximize the <em>likelihood</em> — the probability of your data as a function of the
          parameters.
        </p>
        <p>
          Here&apos;s the intuition. Suppose you flip a coin 100 times and get 73 heads. What&apos;s the
          most likely bias of the coin? MLE says: find the value of <InlineMath math="p" /> that makes
          73 heads out of 100 flips as probable as possible. The answer is <InlineMath math="p = 0.73" /> —
          no other value of <InlineMath math="p" /> makes the observed data more likely.
        </p>
        <p>
          MLE is the <strong>workhorse of statistical estimation</strong>. Logistic regression, neural
          networks (with cross-entropy loss), Gaussian mixture models, and hidden Markov models all use
          MLE under the hood. When you train a model by minimizing negative log-likelihood (or
          equivalently, cross-entropy), you are doing MLE.
        </p>
        <p>
          The key properties that make MLE so popular: it is <strong>consistent</strong> (converges to
          the true parameter as <InlineMath math="n \to \infty" />), <strong>asymptotically efficient</strong>
          (achieves the lowest possible variance among consistent estimators), and
          <strong>asymptotically normal</strong> (the sampling distribution of the MLE is approximately
          Gaussian for large <InlineMath math="n" />).
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>The Likelihood Function</h3>
        <p>
          Given independent observations <InlineMath math="x_1, x_2, \ldots, x_n" /> from a distribution
          with parameter(s) <InlineMath math="\theta" />, the <strong>likelihood function</strong> is:
        </p>
        <BlockMath math="L(\theta) = \prod_{i=1}^n f(x_i \mid \theta)" />
        <p>
          The <strong>log-likelihood</strong> is more convenient (sums instead of products, numerically
          stable):
        </p>
        <BlockMath math="\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log f(x_i \mid \theta)" />
        <p>
          The MLE is: <InlineMath math="\hat{\theta}_{\text{MLE}} = \arg\max_\theta \, \ell(\theta)" />.
          Since <InlineMath math="\log" /> is monotone, maximizing <InlineMath math="\ell" /> is
          equivalent to maximizing <InlineMath math="L" />.
        </p>

        <h3>MLE for the Gaussian Distribution</h3>
        <p>
          Given <InlineMath math="x_1, \ldots, x_n \sim \mathcal{N}(\mu, \sigma^2)" />, the log-likelihood is:
        </p>
        <BlockMath math="\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2" />
        <p>Taking the derivative with respect to <InlineMath math="\mu" /> and setting it to zero:</p>
        <BlockMath math="\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0 \implies \hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}" />
        <p>Taking the derivative with respect to <InlineMath math="\sigma^2" />:</p>
        <BlockMath math="\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n (x_i - \mu)^2 = 0 \implies \hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2" />
        <p>
          Note: the MLE for variance divides by <InlineMath math="n" />, not <InlineMath math="n-1" />.
          This is <strong>biased</strong> — it systematically underestimates the true variance. The
          unbiased estimator <InlineMath math="s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2" /> is not the
          MLE. This is a known trade-off: MLE optimizes likelihood, not unbiasedness.
        </p>

        <h3>MLE for the Bernoulli Distribution</h3>
        <p>
          Given <InlineMath math="x_1, \ldots, x_n \sim \text{Bernoulli}(p)" /> where each <InlineMath math="x_i \in \{0, 1\}" />:
        </p>
        <BlockMath math="\ell(p) = \sum_{i=1}^n \left[ x_i \log p + (1 - x_i) \log(1 - p) \right]" />
        <p>Taking the derivative and solving:</p>
        <BlockMath math="\frac{\partial \ell}{\partial p} = \frac{\sum x_i}{p} - \frac{n - \sum x_i}{1 - p} = 0 \implies \hat{p}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}" />
        <p>
          The MLE for the Bernoulli parameter is simply the sample proportion — the fraction of 1s in
          the data. This is exactly what logistic regression maximizes (where <InlineMath math="p" /> is
          modeled as a function of features).
        </p>

        <h3>MLE for the Poisson Distribution</h3>
        <p>
          Given <InlineMath math="x_1, \ldots, x_n \sim \text{Poisson}(\lambda)" />:
        </p>
        <BlockMath math="\ell(\lambda) = \sum_{i=1}^n \left[ x_i \log \lambda - \lambda - \log(x_i!) \right] = \log\lambda \sum x_i - n\lambda - \sum \log(x_i!)" />
        <p>Taking the derivative:</p>
        <BlockMath math="\frac{\partial \ell}{\partial \lambda} = \frac{\sum x_i}{\lambda} - n = 0 \implies \hat{\lambda}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}" />
        <p>
          Again, the MLE is the sample mean — a beautiful and intuitive result since <InlineMath math="\lambda" /> is
          both the mean and variance of the Poisson distribution.
        </p>

        <h3>Asymptotic Properties</h3>
        <p>Under regularity conditions, as <InlineMath math="n \to \infty" />:</p>
        <BlockMath math="\hat{\theta}_{\text{MLE}} \xrightarrow{d} \mathcal{N}\!\left(\theta_0, \frac{1}{I(\theta_0)}\right)" />
        <p>
          where <InlineMath math="I(\theta)" /> is the <strong>Fisher information</strong>:
        </p>
        <BlockMath math="I(\theta) = -E\!\left[\frac{\partial^2 \ell}{\partial \theta^2}\right] = E\!\left[\left(\frac{\partial \ell}{\partial \theta}\right)^2\right]" />
        <p>
          The <strong>Cram&eacute;r-Rao lower bound</strong> states that no unbiased estimator can have
          variance less than <InlineMath math="1 / I(\theta)" />. MLE achieves this bound asymptotically,
          making it <em>efficient</em>.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <CodeBlock
          language="python"
          title="mle_gaussian.py"
          code={`import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy import stats

np.random.seed(42)

# ── MLE for Gaussian: analytical vs numerical ──
true_mu, true_sigma = 5.0, 2.0
data = np.random.normal(true_mu, true_sigma, size=200)

# Analytical MLE
mu_mle = np.mean(data)
sigma2_mle = np.var(data, ddof=0)  # ddof=0 = MLE (divide by n)
print(f"True params:      mu={true_mu}, sigma^2={true_sigma**2}")
print(f"Analytical MLE:   mu={mu_mle:.4f}, sigma^2={sigma2_mle:.4f}")

# Numerical MLE via optimization
def neg_log_lik_gaussian(params, data):
    mu, log_sigma = params  # use log(sigma) to ensure sigma > 0
    sigma = np.exp(log_sigma)
    n = len(data)
    return 0.5 * n * np.log(2 * np.pi * sigma**2) + \\
           np.sum((data - mu)**2) / (2 * sigma**2)

result = minimize(neg_log_lik_gaussian, x0=[0, 0], args=(data,), method="Nelder-Mead")
mu_num, sigma_num = result.x[0], np.exp(result.x[1])
print(f"Numerical MLE:    mu={mu_num:.4f}, sigma^2={sigma_num**2:.4f}")

# scipy.stats makes this even easier
mu_scipy, sigma_scipy = stats.norm.fit(data)
print(f"scipy.stats.fit:  mu={mu_scipy:.4f}, sigma^2={sigma_scipy**2:.4f}")`}
        />

        <CodeBlock
          language="python"
          title="mle_bernoulli_poisson.py"
          code={`import numpy as np
from scipy import stats

np.random.seed(42)

# ── MLE for Bernoulli ──
true_p = 0.35
coin_flips = np.random.binomial(1, true_p, size=500)  # 0s and 1s

p_mle = np.mean(coin_flips)  # MLE is just the sample proportion
# Standard error of MLE (from Fisher information)
se_p = np.sqrt(p_mle * (1 - p_mle) / len(coin_flips))
print(f"Bernoulli MLE: p_hat = {p_mle:.4f} (true: {true_p})")
print(f"  95% CI: [{p_mle - 1.96*se_p:.4f}, {p_mle + 1.96*se_p:.4f}]")

# ── MLE for Poisson ──
true_lambda = 3.7
counts = np.random.poisson(true_lambda, size=300)

lambda_mle = np.mean(counts)  # MLE is the sample mean
# Fisher information for Poisson: I(lambda) = n / lambda
se_lambda = np.sqrt(lambda_mle / len(counts))
print(f"\\nPoisson MLE: lambda_hat = {lambda_mle:.4f} (true: {true_lambda})")
print(f"  95% CI: [{lambda_mle - 1.96*se_lambda:.4f}, {lambda_mle + 1.96*se_lambda:.4f}]")

# ── Verify with likelihood profile ──
import matplotlib.pyplot as plt

lambdas = np.linspace(2.5, 5.0, 200)
log_liks = [np.sum(stats.poisson.logpmf(counts, lam)) for lam in lambdas]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(lambdas, log_liks, linewidth=2)
ax.axvline(lambda_mle, color="red", linestyle="--", label=f"MLE = {lambda_mle:.2f}")
ax.axvline(true_lambda, color="green", linestyle=":", label=f"True = {true_lambda}")
ax.set_xlabel("lambda")
ax.set_ylabel("Log-Likelihood")
ax.set_title("Poisson Log-Likelihood Profile")
ax.legend()
plt.tight_layout()
plt.show()`}
        />

        <CodeBlock
          language="python"
          title="mle_vs_map.py"
          code={`import numpy as np
from scipy import stats

np.random.seed(42)

# ── MLE vs MAP (Maximum A Posteriori) ──
# Suppose we have very little data: 3 coin flips, all heads
data = np.array([1, 1, 1])
n = len(data)
k = np.sum(data)  # number of successes

# MLE: p_hat = k/n = 3/3 = 1.0 (certainty of always heads!)
p_mle = k / n
print(f"MLE:  p_hat = {p_mle:.2f}  (overfits to small sample)")

# MAP with Beta(2, 2) prior (mild preference for fairness)
# Posterior: Beta(alpha + k, beta + n - k) = Beta(2 + 3, 2 + 0) = Beta(5, 2)
alpha_prior, beta_prior = 2, 2
alpha_post = alpha_prior + k
beta_post = beta_prior + (n - k)
p_map = (alpha_post - 1) / (alpha_post + beta_post - 2)  # mode of Beta
print(f"MAP:  p_hat = {p_map:.2f}  (regularized by prior)")

# MAP with stronger prior Beta(10, 10)
alpha_post2 = 10 + k
beta_post2 = 10 + (n - k)
p_map2 = (alpha_post2 - 1) / (alpha_post2 + beta_post2 - 2)
print(f"MAP (strong prior): p_hat = {p_map2:.2f}")

# As n grows, MLE and MAP converge (data overwhelms the prior)
for n_sim in [3, 30, 300, 3000]:
    data_sim = np.random.binomial(1, 0.6, n_sim)
    k_sim = data_sim.sum()
    mle = k_sim / n_sim
    map_est = (2 + k_sim - 1) / (2 + 2 + n_sim - 2)
    print(f"n={n_sim:5d}: MLE={mle:.4f}, MAP={map_est:.4f}, "
          f"diff={abs(mle-map_est):.4f}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>Logistic regression IS MLE</strong>: The cost function in logistic regression is the
            negative log-likelihood of the Bernoulli model. When you call
            <code>LogisticRegression().fit()</code>, you are finding the MLE of the weight parameters.
          </li>
          <li>
            <strong>Cross-entropy loss is negative log-likelihood</strong>: In neural network
            classification, minimizing cross-entropy is equivalent to maximizing the likelihood of the
            observed labels under the model&apos;s predicted probabilities.
          </li>
          <li>
            <strong>Always work with log-likelihood</strong>: The likelihood itself involves products of
            potentially tiny probabilities, leading to numerical underflow. Log transforms products into
            sums, which are numerically stable.
          </li>
          <li>
            <strong>Use Fisher information for confidence intervals</strong>: The standard error of the
            MLE is approximately <InlineMath math="1/\sqrt{I(\hat{\theta})}" />, giving
            you confidence intervals for free: <InlineMath math="\hat{\theta} \pm z_{\alpha/2} / \sqrt{I(\hat{\theta})}" />.
          </li>
          <li>
            <strong>MLE can overfit with small data</strong>: With 3 heads in 3 flips, MLE says
            <InlineMath math="p = 1.0" />. In practice, use regularization (L2 = MAP with Gaussian
            prior) or Bayesian methods when data is scarce.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Confusing likelihood with probability</strong>: <InlineMath math="L(\theta \mid x)" />
            is <em>not</em> the probability of <InlineMath math="\theta" />. It is the probability of the
            data <InlineMath math="x" /> given <InlineMath math="\theta" />, viewed as a function
            of <InlineMath math="\theta" />. The likelihood does not integrate to 1 over
            <InlineMath math="\theta" />.
          </li>
          <li>
            <strong>Forgetting that the MLE variance estimator is biased</strong>: The Gaussian MLE
            divides by <InlineMath math="n" />, giving a biased estimate of variance. For small samples,
            this bias matters. Use <InlineMath math="n - 1" /> for the unbiased sample variance.
          </li>
          <li>
            <strong>Assuming MLE always exists and is unique</strong>: For some models (e.g., mixture
            models with a single data point in a cluster), the likelihood is unbounded and no finite
            MLE exists. For multimodal likelihoods, the optimizer may find a local maximum.
          </li>
          <li>
            <strong>Not checking the second derivative</strong>: Setting the gradient to zero finds
            stationary points, which could be maxima, minima, or saddle points. Verify that the Hessian
            is negative definite at the solution to confirm it&apos;s a maximum.
          </li>
          <li>
            <strong>Applying asymptotic results to small samples</strong>: The normality of the MLE is an
            asymptotic result. With <InlineMath math="n = 10" />, the sampling distribution may be far
            from Gaussian, and confidence intervals based on Fisher information may be inaccurate.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> Derive the MLE for the parameter <InlineMath math="\lambda" /> of
          an Exponential distribution given observations <InlineMath math="x_1, \ldots, x_n" />. Then
          compute the Fisher information and the asymptotic variance of the MLE.
        </p>
        <p><strong>Answer:</strong></p>
        <p>
          The Exponential PDF is <InlineMath math="f(x \mid \lambda) = \lambda e^{-\lambda x}" /> for
          <InlineMath math="x > 0" />.
        </p>
        <p><strong>Step 1: Write the log-likelihood.</strong></p>
        <BlockMath math="\ell(\lambda) = \sum_{i=1}^n \log(\lambda e^{-\lambda x_i}) = n \log \lambda - \lambda \sum_{i=1}^n x_i" />
        <p><strong>Step 2: Take the derivative and set to zero.</strong></p>
        <BlockMath math="\frac{\partial \ell}{\partial \lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0 \implies \hat{\lambda}_{\text{MLE}} = \frac{n}{\sum x_i} = \frac{1}{\bar{x}}" />
        <p>The MLE for the rate is the reciprocal of the sample mean.</p>
        <p><strong>Step 3: Fisher information.</strong></p>
        <BlockMath math="\frac{\partial^2 \ell}{\partial \lambda^2} = -\frac{n}{\lambda^2}" />
        <BlockMath math="I(\lambda) = -E\!\left[\frac{\partial^2 \ell}{\partial \lambda^2}\right] = \frac{n}{\lambda^2}" />
        <p><strong>Step 4: Asymptotic variance.</strong></p>
        <BlockMath math="\text{Var}(\hat{\lambda}) \approx \frac{1}{I(\lambda)} = \frac{\lambda^2}{n}" />
        <p>
          So the standard error is <InlineMath math="SE \approx \lambda / \sqrt{n}" />, and a 95%
          confidence interval is <InlineMath math="\hat{\lambda} \pm 1.96 \cdot \hat{\lambda} / \sqrt{n}" />.
          The relative precision improves as <InlineMath math="1/\sqrt{n}" />, which is a general property
          of MLE.
        </p>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Casella &amp; Berger, &quot;Statistical Inference&quot; (Ch. 7)</strong> — Rigorous treatment of MLE, sufficiency, and the Cram&eacute;r-Rao bound.</li>
          <li><strong>Larry Wasserman, &quot;All of Statistics&quot; (Ch. 9)</strong> — Concise and practical coverage of MLE with examples from ML applications.</li>
          <li><strong>Bishop, &quot;Pattern Recognition and Machine Learning&quot; (Ch. 2)</strong> — Connects MLE to Bayesian estimation and shows how regularization = MAP.</li>
          <li><strong>statsmodels GenericLikelihoodModel</strong> — Python class for defining custom likelihoods and running MLE with standard errors and confidence intervals.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
