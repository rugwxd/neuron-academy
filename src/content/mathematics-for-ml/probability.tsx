"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function Probability() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Probability is the math of <strong>uncertainty</strong>. A <strong>sample space</strong> is
          the set of all possible outcomes of an experiment (flip a coin: heads or tails). An{" "}
          <strong>event</strong> is a subset of those outcomes you care about.
        </p>
        <p>
          <strong>Conditional probability</strong> answers: &quot;given that I already know something,
          how does that change my belief?&quot; If you know a patient tested positive, what&apos;s the
          actual chance they&apos;re sick? That&apos;s where <strong>Bayes&apos; theorem</strong> comes in
          &mdash; it&apos;s the formal way to update your beliefs when you get new evidence.
        </p>
        <p>
          <strong>Why this is the foundation of ML:</strong> Every model is making probabilistic
          predictions. Logistic regression outputs a probability. Naive Bayes literally applies
          Bayes&apos; theorem. Neural network softmax outputs are probabilities. Bayesian deep
          learning puts distributions over the <em>weights themselves</em>. If you understand
          probability, you understand what every model is actually doing under the hood.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Axioms of Probability</h3>
        <p>
          For a sample space <InlineMath math="\Omega" /> and events <InlineMath math="A, B \subseteq \Omega" />:
        </p>
        <BlockMath math="0 \le P(A) \le 1, \quad P(\Omega) = 1, \quad P(A \cup B) = P(A) + P(B) \text{ if } A \cap B = \emptyset" />

        <h3>Conditional Probability and Bayes&apos; Theorem</h3>
        <p>
          Conditional probability of <InlineMath math="A" /> given <InlineMath math="B" />:
        </p>
        <BlockMath math="P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}" />
        <p>
          Derived from the definition <InlineMath math="P(A \mid B) = \frac{P(A \cap B)}{P(B)}" /> and
          noting <InlineMath math="P(A \cap B) = P(B \mid A) P(A)" />. The full form with the law of total probability:
        </p>
        <BlockMath math="P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B \mid A) P(A) + P(B \mid \neg A) P(\neg A)}" />
        <p>
          Here <InlineMath math="P(A)" /> is the <strong>prior</strong>, <InlineMath math="P(B \mid A)" /> is
          the <strong>likelihood</strong>, and <InlineMath math="P(A \mid B)" /> is the <strong>posterior</strong>.
        </p>

        <h3>Common Distributions</h3>
        <p><strong>Bernoulli</strong> &mdash; single binary trial:</p>
        <BlockMath math="P(X = k) = p^k (1-p)^{1-k}, \quad k \in \{0, 1\}" />
        <p><strong>Binomial</strong> &mdash; number of successes in <InlineMath math="n" /> trials:</p>
        <BlockMath math="P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}" />
        <p><strong>Poisson</strong> &mdash; count of rare events in a fixed interval:</p>
        <BlockMath math="P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}" />
        <p><strong>Gaussian (Normal)</strong> &mdash; the bell curve:</p>
        <BlockMath math="f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)" />
        <p><strong>Exponential</strong> &mdash; time between Poisson events:</p>
        <BlockMath math="f(x) = \lambda e^{-\lambda x}, \quad x \ge 0" />
        <p><strong>Beta</strong> &mdash; distribution over probabilities (conjugate prior for Bernoulli):</p>
        <BlockMath math="f(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad x \in [0, 1]" />

        <h3>Expectation, Variance, Covariance</h3>
        <BlockMath math="E[X] = \sum_x x \, P(X = x), \quad \text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2" />
        <BlockMath math="\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]" />

        <h3>Joint, Marginal, and Conditional Distributions</h3>
        <p>
          For two random variables <InlineMath math="X" /> and <InlineMath math="Y" />:
        </p>
        <BlockMath math="P(X = x) = \sum_y P(X = x, Y = y) \quad \text{(marginalization)}" />
        <BlockMath math="P(X \mid Y) = \frac{P(X, Y)}{P(Y)} \quad \text{(conditioning)}" />

        <h3>Law of Large Numbers and CLT Preview</h3>
        <p>
          The <strong>Law of Large Numbers (LLN)</strong>: as sample size <InlineMath math="n \to \infty" />,
          the sample mean <InlineMath math="\bar{X}_n" /> converges to the true mean <InlineMath math="\mu" />.
        </p>
        <p>
          The <strong>Central Limit Theorem (CLT)</strong>: regardless of the original distribution, the
          distribution of sample means approaches a Gaussian as <InlineMath math="n" /> grows:
        </p>
        <BlockMath math="\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)" />
      </TopicSection>

      <TopicSection type="code">
        <h3>Bayes&apos; Theorem from Scratch &mdash; Medical Test Example</h3>
        <CodeBlock
          language="python"
          title="bayes_theorem.py"
          code={`import numpy as np

# ── Bayes' theorem from scratch ──────────────────────────────
# Scenario: A disease affects 1 in 1000 people.
# A test has 99% sensitivity (true positive rate)
# and 95% specificity (true negative rate).
# If someone tests positive, what's the probability they're sick?

def bayes_theorem(prior, likelihood, false_positive_rate):
    """
    Compute posterior probability using Bayes' theorem.

    P(Disease | Positive) = P(Pos | Disease) * P(Disease)
                            / P(Positive)
    where P(Positive) = P(Pos|Disease)*P(Disease) + P(Pos|No Disease)*P(No Disease)
    """
    p_positive = likelihood * prior + false_positive_rate * (1 - prior)
    posterior = (likelihood * prior) / p_positive
    return posterior

# Parameters
prior = 0.001           # P(Disease) = 1/1000
sensitivity = 0.99      # P(Positive | Disease)
specificity = 0.95      # P(Negative | No Disease)
false_positive_rate = 1 - specificity  # P(Positive | No Disease) = 0.05

posterior = bayes_theorem(prior, sensitivity, false_positive_rate)
print(f"Prior probability of disease:    {prior:.4f}")
print(f"Posterior after positive test:   {posterior:.4f}")
print(f"That's only about {posterior:.1%} — most positives are false positives!")
# Output:
# Prior probability of disease:    0.0010
# Posterior after positive test:   0.0194
# That's only about 1.9% — most positives are false positives!

# Simulate to verify
np.random.seed(42)
n_people = 1_000_000
has_disease = np.random.rand(n_people) < prior

# Test results
test_positive = np.where(
    has_disease,
    np.random.rand(n_people) < sensitivity,   # true positive
    np.random.rand(n_people) < false_positive_rate  # false positive
)

simulated_posterior = has_disease[test_positive].mean()
print(f"\\nSimulated posterior: {simulated_posterior:.4f}")
print(f"Analytical posterior: {posterior:.4f}")`}
        />

        <h3>Simulating Common Distributions</h3>
        <CodeBlock
          language="python"
          title="distributions.py"
          code={`import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# 1. Bernoulli (PMF)
p = 0.7
samples = np.random.binomial(1, p, size=10000)
axes[0, 0].bar([0, 1], [np.mean(samples == 0), np.mean(samples == 1)],
               color=['steelblue', 'coral'], width=0.4)
axes[0, 0].set_title(f"Bernoulli(p={p})")
axes[0, 0].set_xticks([0, 1])

# 2. Binomial (PMF)
n, p = 20, 0.4
samples = np.random.binomial(n, p, size=10000)
values, counts = np.unique(samples, return_counts=True)
axes[0, 1].bar(values, counts / len(samples), color='steelblue')
axes[0, 1].set_title(f"Binomial(n={n}, p={p})")

# 3. Poisson (PMF)
lam = 5
samples = np.random.poisson(lam, size=10000)
values, counts = np.unique(samples, return_counts=True)
axes[0, 2].bar(values, counts / len(samples), color='steelblue')
axes[0, 2].set_title(f"Poisson(\\u03bb={lam})")

# 4. Gaussian (PDF)
mu, sigma = 0, 1
samples = np.random.normal(mu, sigma, size=10000)
axes[1, 0].hist(samples, bins=50, density=True, alpha=0.7, color='steelblue')
x = np.linspace(-4, 4, 200)
pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
axes[1, 0].plot(x, pdf, 'r-', linewidth=2)
axes[1, 0].set_title(f"Gaussian(\\u03bc={mu}, \\u03c3={sigma})")

# 5. Exponential (PDF)
lam = 1.5
samples = np.random.exponential(1 / lam, size=10000)
axes[1, 1].hist(samples, bins=50, density=True, alpha=0.7, color='steelblue')
x = np.linspace(0, 5, 200)
pdf = lam * np.exp(-lam * x)
axes[1, 1].plot(x, pdf, 'r-', linewidth=2)
axes[1, 1].set_title(f"Exponential(\\u03bb={lam})")

# 6. Beta (PDF)
alpha, beta_param = 2, 5
samples = np.random.beta(alpha, beta_param, size=10000)
axes[1, 2].hist(samples, bins=50, density=True, alpha=0.7, color='steelblue')
from scipy.stats import beta as beta_dist
x = np.linspace(0, 1, 200)
axes[1, 2].plot(x, beta_dist.pdf(x, alpha, beta_param), 'r-', linewidth=2)
axes[1, 2].set_title(f"Beta(\\u03b1={alpha}, \\u03b2={beta_param})")

plt.tight_layout()
plt.savefig("distributions.png", dpi=150)
plt.show()`}
        />

        <h3>Law of Large Numbers Demonstration</h3>
        <CodeBlock
          language="python"
          title="law_of_large_numbers.py"
          code={`import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Roll a fair die many times
# True mean = (1+2+3+4+5+6)/6 = 3.5
true_mean = 3.5
n_rolls = 10000
rolls = np.random.randint(1, 7, size=n_rolls)

# Running average after each roll
running_avg = np.cumsum(rolls) / np.arange(1, n_rolls + 1)

plt.figure(figsize=(10, 5))
plt.plot(running_avg, color='steelblue', linewidth=0.8, label='Running average')
plt.axhline(y=true_mean, color='red', linestyle='--', linewidth=2, label=f'True mean = {true_mean}')
plt.xlabel('Number of rolls')
plt.ylabel('Running average')
plt.title('Law of Large Numbers: Fair Die')
plt.legend()
plt.xscale('log')
plt.grid(True, alpha=0.3)

# Print convergence at different sample sizes
for n in [10, 100, 1000, 10000]:
    avg = rolls[:n].mean()
    print(f"n={n:>5d}:  sample mean = {avg:.4f}  (error = {abs(avg - true_mean):.4f})")
# Output:
# n=   10:  sample mean = 3.5000  (error = 0.0000)
# n=  100:  sample mean = 3.3900  (error = 0.1100)
# n= 1000:  sample mean = 3.4830  (error = 0.0170)
# n=10000:  sample mean = 3.5015  (error = 0.0015)

plt.savefig("law_of_large_numbers.png", dpi=150)
plt.show()`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Bernoulli &amp; Binomial</strong>: Binary classification (spam/not spam, click/no click). Logistic regression models a Bernoulli distribution.</li>
          <li><strong>Poisson</strong>: Count data &mdash; website visits per hour, number of defects, rare event modeling.</li>
          <li><strong>Gaussian</strong>: Shows up <em>everywhere</em> because of the Central Limit Theorem. Noise modeling, Gaussian processes, normalization assumptions, VAE latent spaces.</li>
          <li><strong>Exponential</strong>: Time-to-event models, survival analysis, waiting times between events.</li>
          <li><strong>Beta</strong>: The conjugate prior for Bernoulli/Binomial. Perfect for modeling uncertainty about probabilities themselves (A/B testing, Thompson sampling).</li>
          <li><strong>Conjugate priors preview</strong>: When your prior and posterior come from the same family, updates are simple. Beta-Bernoulli is the classic example: start with <InlineMath math="\text{Beta}(\alpha, \beta)" />, observe <InlineMath math="k" /> successes in <InlineMath math="n" /> trials, posterior is <InlineMath math="\text{Beta}(\alpha + k, \beta + n - k)" />.</li>
          <li><strong>Why Gaussian is everywhere</strong>: CLT guarantees that averages of <em>any</em> distribution tend toward Gaussian. Since most measurements are averages or sums of many small effects, Gaussians naturally arise.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Confusing prior and posterior</strong>: The prior is your belief <em>before</em> seeing data. The posterior is <em>after</em>. Bayes&apos; theorem tells you how to go from one to the other.</li>
          <li><strong>Base rate fallacy (base rate neglect)</strong>: Ignoring the prior probability. Even a 99%-accurate test gives mostly false positives when the disease prevalence is 0.1%. Always multiply by the prior!</li>
          <li><strong>Assuming independence when it doesn&apos;t hold</strong>: Naive Bayes assumes features are conditionally independent. This is almost never true, yet the model often works surprisingly well. Know the assumption you&apos;re making.</li>
          <li><strong>Confusing PDF value with probability</strong>: For continuous distributions, <InlineMath math="f(x)" /> is a <em>density</em>, not a probability. <InlineMath math="P(X = x) = 0" /> for continuous variables. Probability comes from integrating: <InlineMath math="P(a \le X \le b) = \int_a^b f(x)\,dx" />. Densities can exceed 1.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> A rare disease affects 1 in 10,000 people. A test has 99% sensitivity and 99% specificity. A patient tests positive. What is the probability they actually have the disease?</p>
        <p><strong>Solution:</strong></p>
        <ol>
          <li>Define: <InlineMath math="P(D) = 0.0001" />, <InlineMath math="P(+ \mid D) = 0.99" />, <InlineMath math="P(+ \mid \neg D) = 0.01" /></li>
          <li>Apply Bayes&apos; theorem:
            <BlockMath math="P(D \mid +) = \frac{P(+ \mid D) \cdot P(D)}{P(+ \mid D) \cdot P(D) + P(+ \mid \neg D) \cdot P(\neg D)}" />
          </li>
          <li>Plug in:
            <BlockMath math="P(D \mid +) = \frac{0.99 \times 0.0001}{0.99 \times 0.0001 + 0.01 \times 0.9999} = \frac{0.000099}{0.000099 + 0.009999} \approx 0.0098" />
          </li>
          <li>Result: only about <strong>0.98%</strong> &mdash; less than 1 in 100!</li>
        </ol>
        <p>
          <strong>Key insight:</strong> Even with a very accurate test, when the condition is rare, the
          false positives vastly outnumber the true positives. This is why screening tests use multiple
          rounds of confirmation.
        </p>
        <CodeBlock
          language="python"
          code={`# Quick verification
prior = 0.0001
sensitivity = 0.99
false_positive_rate = 0.01

p_positive = sensitivity * prior + false_positive_rate * (1 - prior)
posterior = (sensitivity * prior) / p_positive
print(f"P(Disease | Positive) = {posterior:.4f}")
# Output: P(Disease | Positive) = 0.0098`}
        />
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Jaynes, &quot;Probability Theory: The Logic of Science&quot;</strong> &mdash; The deepest treatment of probability as extended logic, essential for understanding Bayesian reasoning</li>
          <li><strong>Blitzstein &amp; Hwang, &quot;Introduction to Probability&quot;</strong> &mdash; Modern, intuitive textbook with excellent exercises (Harvard Stat 110)</li>
          <li><strong>3Blue1Brown &quot;Bayes theorem&quot; video</strong> &mdash; Beautiful visual explanation of updating beliefs</li>
          <li><strong>Bishop, &quot;Pattern Recognition and Machine Learning&quot; Ch. 1-2</strong> &mdash; Probability foundations for ML, introduction to Bayesian inference</li>
        </ul>
      </TopicSection>
    </div>
  );
}
