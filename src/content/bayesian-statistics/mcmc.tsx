"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function MCMC() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Markov Chain Monte Carlo (MCMC) solves a fundamental problem in Bayesian statistics:
          <strong> how do you sample from a probability distribution you can&apos;t write down in closed form?</strong>
          For simple problems like the Beta-Binomial, we get a neat closed-form posterior. But for
          real-world models with many parameters and complex likelihoods, the posterior is an
          intractable high-dimensional integral that no formula can solve.
        </p>
        <p>
          MCMC&apos;s brilliant trick: construct a random walk (Markov chain) that, after enough steps,
          produces samples that look as if they came from your target distribution. You don&apos;t
          need to know the normalizing constant — you only need to evaluate the unnormalized
          posterior, which is just the likelihood times the prior. Run the chain long enough,
          throw away the initial &quot;burn-in&quot; samples (before the chain has converged), and the
          remaining samples approximate the posterior.
        </p>
        <p>
          The two foundational algorithms are <strong>Metropolis-Hastings</strong> and <strong>Gibbs
          sampling</strong>. Metropolis-Hastings proposes a move and accepts or rejects it based on
          how much better the new position is. Gibbs sampling breaks a high-dimensional problem
          into a sequence of one-dimensional conditional samples. Modern tools like PyMC and Stan
          use more advanced variants (NUTS — the No-U-Turn Sampler), but understanding M-H and
          Gibbs gives you the foundation to diagnose any MCMC problem.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Metropolis-Hastings Algorithm</h3>
        <p>
          Target distribution: <InlineMath math="\pi(\theta) \propto P(D \mid \theta) P(\theta)" /> (the unnormalized posterior).
          Proposal distribution: <InlineMath math="q(\theta^* \mid \theta_t)" />.
        </p>
        <ol>
          <li>Start at some <InlineMath math="\theta_0" /></li>
          <li>Propose a new state: <InlineMath math="\theta^* \sim q(\theta^* \mid \theta_t)" /></li>
          <li>Compute the acceptance ratio:
            <BlockMath math="\alpha = \min\left(1, \; \frac{\pi(\theta^*) \, q(\theta_t \mid \theta^*)}{\pi(\theta_t) \, q(\theta^* \mid \theta_t)}\right)" />
          </li>
          <li>Accept with probability <InlineMath math="\alpha" />: set <InlineMath math="\theta_{t+1} = \theta^*" />. Otherwise <InlineMath math="\theta_{t+1} = \theta_t" />.</li>
        </ol>
        <p>
          For a <strong>symmetric</strong> proposal (<InlineMath math="q(\theta^* \mid \theta_t) = q(\theta_t \mid \theta^*)" />,
          e.g., a Gaussian centered at the current state), the ratio simplifies to:
        </p>
        <BlockMath math="\alpha = \min\left(1, \; \frac{\pi(\theta^*)}{\pi(\theta_t)}\right)" />
        <p>
          This is the original <strong>Metropolis algorithm</strong>. If the proposed state has higher
          posterior density, always accept. If lower, accept with probability equal to the density ratio.
        </p>

        <h3>Gibbs Sampling</h3>
        <p>
          For a parameter vector <InlineMath math="\theta = (\theta_1, \theta_2, \ldots, \theta_d)" />,
          Gibbs sampling iterates:
        </p>
        <BlockMath math="\theta_1^{(t+1)} \sim P(\theta_1 \mid \theta_2^{(t)}, \theta_3^{(t)}, \ldots, \theta_d^{(t)})" />
        <BlockMath math="\theta_2^{(t+1)} \sim P(\theta_2 \mid \theta_1^{(t+1)}, \theta_3^{(t)}, \ldots, \theta_d^{(t)})" />
        <BlockMath math="\vdots" />
        <BlockMath math="\theta_d^{(t+1)} \sim P(\theta_d \mid \theta_1^{(t+1)}, \theta_2^{(t+1)}, \ldots, \theta_{d-1}^{(t+1)})" />
        <p>
          Each step samples from a <strong>full conditional distribution</strong> — the distribution of
          one parameter given all the others fixed. Gibbs is a special case of M-H with acceptance
          probability 1 (always accepted). It works when the full conditionals are easy to sample
          from (e.g., conjugate models).
        </p>

        <h3>Convergence Diagnostics</h3>
        <p>
          <strong>Gelman-Rubin <InlineMath math="\hat{R}" /> statistic</strong>: Run multiple chains from
          different starting points. Compare the between-chain variance <InlineMath math="B" /> to the
          within-chain variance <InlineMath math="W" />:
        </p>
        <BlockMath math="\hat{R} = \sqrt{\frac{\frac{n-1}{n}W + \frac{1}{n}B}{W}}" />
        <p>
          Target: <InlineMath math="\hat{R} < 1.01" />. Values substantially above 1 indicate the chains
          have not converged to the same distribution.
        </p>
        <p>
          <strong>Effective sample size (ESS)</strong>: Due to autocorrelation between successive samples,
          <InlineMath math="N" /> MCMC samples contain less information than <InlineMath math="N" /> independent
          samples. ESS estimates the equivalent number of independent samples. Aim
          for <InlineMath math="\text{ESS} > 400" /> for reliable posterior summaries.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <CodeBlock
          language="python"
          title="metropolis_hastings_from_scratch.py"
          code={`import numpy as np
from scipy import stats

np.random.seed(42)

# ============================================
# Metropolis-Hastings: infer mean of a Normal
# ============================================
# True model: X ~ Normal(mu=5, sigma=2)
# Prior: mu ~ Normal(0, 10)
# We want the posterior P(mu | data)

data = np.random.normal(5, 2, size=30)

def log_posterior(mu, data, prior_mean=0, prior_std=10, sigma=2):
    """Unnormalized log posterior: log P(data|mu) + log P(mu)."""
    log_likelihood = np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))
    log_prior = stats.norm.logpdf(mu, loc=prior_mean, scale=prior_std)
    return log_likelihood + log_prior

# MCMC settings
n_iterations = 50000
proposal_std = 0.5  # standard deviation of Gaussian proposal
chain = np.zeros(n_iterations)
chain[0] = 0.0  # arbitrary start
accepted = 0

for t in range(1, n_iterations):
    # Propose new state (symmetric Gaussian proposal)
    proposal = chain[t - 1] + np.random.normal(0, proposal_std)

    # Log acceptance ratio (work in log space for numerical stability)
    log_alpha = log_posterior(proposal, data) - log_posterior(chain[t - 1], data)

    # Accept or reject
    if np.log(np.random.uniform()) < log_alpha:
        chain[t] = proposal
        accepted += 1
    else:
        chain[t] = chain[t - 1]

burn_in = 10000
posterior_samples = chain[burn_in:]

print(f"Acceptance rate: {accepted / n_iterations:.3f} (target: 0.2-0.5)")
print(f"Posterior mean:  {posterior_samples.mean():.3f}")
print(f"Posterior std:   {posterior_samples.std():.3f}")
print(f"True data mean:  {data.mean():.3f}")
print(f"95% credible interval: ({np.percentile(posterior_samples, 2.5):.3f}, "
      f"{np.percentile(posterior_samples, 97.5):.3f})")`}
        />

        <CodeBlock
          language="python"
          title="gibbs_sampling.py"
          code={`import numpy as np
from scipy import stats

np.random.seed(42)

# =============================================
# Gibbs sampler: Normal model with unknown
# mean AND unknown variance
# =============================================
# Data: X_i ~ Normal(mu, sigma^2)
# Priors: mu ~ Normal(mu_0, tau^2), sigma^2 ~ InvGamma(a, b)

# Generate data
true_mu, true_sigma = 3.0, 2.0
data = np.random.normal(true_mu, true_sigma, size=50)
n = len(data)
x_bar = data.mean()
s2 = data.var()

# Hyperparameters
mu_0, tau2 = 0, 100   # vague prior on mu
a, b = 2, 1           # prior on sigma^2

# Gibbs sampler
n_iter = 20000
mu_samples = np.zeros(n_iter)
sigma2_samples = np.zeros(n_iter)

# Initialize
mu_samples[0] = 0
sigma2_samples[0] = 1

for t in range(1, n_iter):
    # Sample mu | sigma^2, data
    # Full conditional: Normal
    precision = n / sigma2_samples[t - 1] + 1 / tau2
    cond_mean = (n * x_bar / sigma2_samples[t - 1] + mu_0 / tau2) / precision
    cond_var = 1 / precision
    mu_samples[t] = np.random.normal(cond_mean, np.sqrt(cond_var))

    # Sample sigma^2 | mu, data
    # Full conditional: Inverse-Gamma
    a_post = a + n / 2
    b_post = b + np.sum((data - mu_samples[t]) ** 2) / 2
    sigma2_samples[t] = 1 / np.random.gamma(a_post, 1 / b_post)

burn_in = 5000
mu_post = mu_samples[burn_in:]
sigma2_post = sigma2_samples[burn_in:]

print(f"True mu: {true_mu:.2f}, Posterior mean: {mu_post.mean():.3f}")
print(f"True sigma^2: {true_sigma**2:.2f}, Posterior mean: {sigma2_post.mean():.3f}")
print(f"95% CI for mu: ({np.percentile(mu_post, 2.5):.3f}, "
      f"{np.percentile(mu_post, 97.5):.3f})")
print(f"95% CI for sigma: ({np.sqrt(np.percentile(sigma2_post, 2.5)):.3f}, "
      f"{np.sqrt(np.percentile(sigma2_post, 97.5)):.3f})")`}
        />

        <CodeBlock
          language="python"
          title="pymc_modern_mcmc.py"
          code={`import numpy as np
import pymc as pm
import arviz as az

np.random.seed(42)

# =============================================
# Modern MCMC with PyMC (uses NUTS sampler)
# =============================================

# Bayesian linear regression
n = 100
X = np.random.randn(n)
true_alpha, true_beta, true_sigma = 2.0, 3.5, 1.0
y = true_alpha + true_beta * X + np.random.normal(0, true_sigma, n)

with pm.Model() as model:
    # Priors
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)

    # Likelihood
    mu = alpha + beta * X
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    # Sample (NUTS sampler — the modern default)
    trace = pm.sample(2000, tune=1000, chains=4, random_seed=42)

# Diagnostics
print(az.summary(trace, var_names=["alpha", "beta", "sigma"]))

# Check convergence
print(f"\\nR-hat values:")
rhat = az.rhat(trace)
for var in ["alpha", "beta", "sigma"]:
    print(f"  {var}: {rhat[var].values:.4f}")

# Effective sample size
ess = az.ess(trace)
for var in ["alpha", "beta", "sigma"]:
    print(f"  ESS {var}: {ess[var].values:.0f}")

# Posterior predictive check
with model:
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)

# Plot diagnostics
az.plot_trace(trace, var_names=["alpha", "beta", "sigma"])
az.plot_posterior(trace, var_names=["alpha", "beta", "sigma"])`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Use PyMC or Stan, not hand-rolled MCMC</strong>: Modern probabilistic programming languages use NUTS (No-U-Turn Sampler), which automatically tunes step sizes and avoids the random-walk behavior of basic M-H. NUTS is orders of magnitude more efficient for most problems.</li>
          <li><strong>Run multiple chains</strong>: Always run at least 4 chains from different starting points. Check that they converge to the same distribution (<InlineMath math="\hat{R} < 1.01" />). Divergent chains are a red flag.</li>
          <li><strong>Tune the proposal distribution</strong>: For Metropolis-Hastings, the optimal acceptance rate is about 23% in high dimensions and 44% for one-dimensional targets. If your acceptance rate is too high, your proposal is too narrow (the chain isn&apos;t exploring). If too low, it&apos;s too wide (proposals keep getting rejected).</li>
          <li><strong>Thinning is usually unnecessary</strong>: Old advice said to keep every k-th sample to reduce autocorrelation. Modern consensus: just run the chain longer. Thinning throws away information. Report ESS to quantify how many effectively independent samples you have.</li>
          <li><strong>Posterior predictive checks are essential</strong>: Generate simulated data from your posterior and compare it to real data. If they don&apos;t match, your model is misspecified — no amount of MCMC tuning will fix a wrong model.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Not discarding burn-in</strong>: The initial samples (before convergence) are biased by the starting point. Always discard at least the first 25-50% of samples. Use trace plots to visually verify the chain has &quot;forgotten&quot; its starting point.</li>
          <li><strong>Declaring convergence too early</strong>: A chain can appear converged for thousands of iterations before finding a second mode. <InlineMath math="\hat{R}" /> helps, but also check trace plots visually. Multi-modal posteriors are especially treacherous.</li>
          <li><strong>Ignoring divergences in NUTS/HMC</strong>: PyMC and Stan report divergent transitions — these indicate the sampler is struggling with the geometry of the posterior. Do not ignore these warnings. Common fixes: reparameterize the model (use non-centered parameterization), increase <InlineMath math="\texttt{target\_accept}" /> to 0.95+, or simplify the model.</li>
          <li><strong>Using MCMC when you don&apos;t need to</strong>: If you have a conjugate model, the posterior is available in closed form — no MCMC needed. Similarly, variational inference (VI) can be faster for large datasets where approximate posteriors are acceptable.</li>
          <li><strong>Confusing prior predictive vs. posterior predictive</strong>: Prior predictive checks (simulating data from the prior) help you choose sensible priors. Posterior predictive checks (simulating data from the posterior) validate the fitted model. Both are important.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Explain the Metropolis-Hastings algorithm. Why does it work? What determines how fast it converges?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Algorithm</strong>: From the current state <InlineMath math="\theta_t" />, propose a new state <InlineMath math="\theta^*" /> from a proposal distribution. Accept with probability <InlineMath math="\alpha = \min(1, \frac{\pi(\theta^*) q(\theta_t \mid \theta^*)}{\pi(\theta_t) q(\theta^* \mid \theta_t)})" />. If rejected, stay at <InlineMath math="\theta_t" />.</li>
          <li><strong>Why it works</strong>: The acceptance criterion ensures <strong>detailed balance</strong>: <InlineMath math="\pi(\theta_t) T(\theta^* \mid \theta_t) = \pi(\theta^*) T(\theta_t \mid \theta^*)" />, where <InlineMath math="T" /> is the transition kernel. This guarantees that <InlineMath math="\pi" /> is a stationary distribution of the Markov chain. By ergodicity, the chain converges to <InlineMath math="\pi" /> regardless of starting point.</li>
          <li><strong>Convergence speed depends on</strong>:
            <ul>
              <li><strong>Proposal scale</strong>: Too small &rarr; high acceptance but slow exploration (random walk). Too large &rarr; frequent rejections, also slow.</li>
              <li><strong>Posterior geometry</strong>: Correlated parameters create narrow &quot;ridges&quot; that are hard to traverse. Reparameterization or Hamiltonian methods help.</li>
              <li><strong>Dimensionality</strong>: Basic M-H scales poorly. In <InlineMath math="d" /> dimensions, mixing time grows as <InlineMath math="O(d^2)" />. HMC/NUTS scales as <InlineMath math="O(d^{5/4})" />.</li>
              <li><strong>Multi-modality</strong>: M-H struggles to jump between well-separated modes. Tempering or parallel tempering can help.</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Betancourt (2017) &quot;A Conceptual Introduction to Hamiltonian Monte Carlo&quot;</strong> — The best explanation of HMC and why NUTS is the modern default. Focuses on geometric intuition.</li>
          <li><strong>Gelman et al. (2013) &quot;Bayesian Data Analysis&quot; Ch. 11-12</strong> — Thorough treatment of MCMC methods and convergence diagnostics.</li>
          <li><strong>PyMC documentation</strong> — Excellent tutorials covering model building, MCMC diagnostics, and posterior analysis with ArviZ.</li>
          <li><strong>Stan User&apos;s Guide</strong> — The Stan documentation is a masterclass in practical Bayesian modeling, with detailed advice on reparameterization and debugging.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
