"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function BayesTheorem() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Bayes&apos; theorem is the mathematically correct way to <strong>update your beliefs
          when you see new evidence</strong>. You start with a prior belief (how likely something
          is before you look at data), observe some evidence, and arrive at a posterior belief
          (how likely something is after you see the data).
        </p>
        <p>
          Here&apos;s a concrete example. A medical test for a rare disease has 99% sensitivity
          (catches 99% of true cases) and 95% specificity (correctly clears 95% of healthy
          people). You test positive. Your gut says &quot;I almost certainly have it&quot; — but
          if the disease affects only 1 in 1,000 people, the math tells a different story.
          Most positive results are false positives because the disease is so rare. Bayes&apos;
          theorem makes this precise: your probability of actually having the disease is only
          about 2%, not 99%.
        </p>
        <p>
          This pattern of reasoning extends far beyond medical testing. In machine learning,
          Bayesian thinking lets you incorporate prior knowledge into models, quantify
          uncertainty in predictions, and avoid overfitting by treating model parameters as
          random variables with distributions rather than fixed unknowns. Naive Bayes
          classifiers, Bayesian optimization, and Bayesian neural networks all flow from this
          single theorem.
        </p>
        <p>
          The Bayesian approach also gives a natural framework for sequential learning: as
          each new data point arrives, your posterior becomes the prior for the next update.
          This makes Bayesian methods ideal for online learning, adaptive experiments, and
          any setting where data accumulates over time.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Bayes&apos; Theorem</h3>
        <BlockMath math="P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}" />
        <p>In the context of inference about a parameter <InlineMath math="\theta" /> given data <InlineMath math="D" />:</p>
        <BlockMath math="P(\theta \mid D) = \frac{P(D \mid \theta) \cdot P(\theta)}{P(D)}" />
        <ul>
          <li><InlineMath math="P(\theta)" /> — <strong>Prior</strong>: your belief about <InlineMath math="\theta" /> before seeing data</li>
          <li><InlineMath math="P(D \mid \theta)" /> — <strong>Likelihood</strong>: how probable the data is given <InlineMath math="\theta" /></li>
          <li><InlineMath math="P(\theta \mid D)" /> — <strong>Posterior</strong>: your updated belief about <InlineMath math="\theta" /> after seeing data</li>
          <li><InlineMath math="P(D)" /> — <strong>Evidence (marginal likelihood)</strong>: a normalizing constant</li>
        </ul>
        <p>Since <InlineMath math="P(D)" /> is constant with respect to <InlineMath math="\theta" />:</p>
        <BlockMath math="\underbrace{P(\theta \mid D)}_{\text{posterior}} \propto \underbrace{P(D \mid \theta)}_{\text{likelihood}} \cdot \underbrace{P(\theta)}_{\text{prior}}" />

        <h3>Example: Coin Flip Inference</h3>
        <p>
          Let <InlineMath math="\theta" /> be the probability of heads. We observe <InlineMath math="k" /> heads in <InlineMath math="n" /> flips.
        </p>
        <p><strong>Likelihood</strong> (Binomial):</p>
        <BlockMath math="P(D \mid \theta) = \binom{n}{k} \theta^k (1 - \theta)^{n - k}" />
        <p><strong>Prior</strong>: Use the conjugate prior — the Beta distribution <InlineMath math="\text{Beta}(\alpha, \beta)" />:</p>
        <BlockMath math="P(\theta) = \frac{\theta^{\alpha - 1}(1 - \theta)^{\beta - 1}}{B(\alpha, \beta)}" />
        <p><strong>Posterior</strong> (also a Beta distribution — this is conjugacy):</p>
        <BlockMath math="P(\theta \mid D) = \text{Beta}(\alpha + k, \; \beta + n - k)" />
        <p>
          The posterior mean is:
        </p>
        <BlockMath math="E[\theta \mid D] = \frac{\alpha + k}{\alpha + \beta + n}" />
        <p>
          This is a weighted average of the prior mean <InlineMath math="\frac{\alpha}{\alpha + \beta}" /> and the MLE <InlineMath math="\frac{k}{n}" />,
          with weights determined by how much data you have versus how strong your prior is.
          As <InlineMath math="n \to \infty" />, the posterior concentrates at the MLE — data overwhelms the prior.
        </p>

        <h3>Credible Intervals</h3>
        <p>
          The Bayesian analog of confidence intervals. A 95% <strong>credible interval</strong> is the
          range <InlineMath math="[a, b]" /> such that:
        </p>
        <BlockMath math="P(a \leq \theta \leq b \mid D) = 0.95" />
        <p>
          Unlike frequentist CIs, you <em>can</em> say &quot;there&apos;s a 95% probability the parameter
          is in this interval&quot; (conditional on the model and prior).
        </p>
      </TopicSection>

      <TopicSection type="code">
        <CodeBlock
          language="python"
          title="bayesian_coin_inference.py"
          code={`import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# =============================================
# Bayesian inference for a coin's bias
# =============================================

# Prior: Beta(2, 2) — mild belief that coin is fair
alpha_prior, beta_prior = 2, 2

# Data: flip coin 20 times, get 14 heads
n_flips = 20
n_heads = 14

# Posterior: Beta(alpha + heads, beta + tails)
alpha_post = alpha_prior + n_heads
beta_post = beta_prior + (n_flips - n_heads)

print(f"Prior:     Beta({alpha_prior}, {beta_prior})")
print(f"Data:      {n_heads} heads in {n_flips} flips")
print(f"Posterior: Beta({alpha_post}, {beta_post})")
print(f"Prior mean:     {alpha_prior/(alpha_prior+beta_prior):.3f}")
print(f"MLE:            {n_heads/n_flips:.3f}")
print(f"Posterior mean:  {alpha_post/(alpha_post+beta_post):.3f}")

# 95% credible interval
ci = stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)
print(f"95% credible interval: ({ci[0]:.3f}, {ci[1]:.3f})")

# Probability that coin is biased toward heads (theta > 0.5)
p_biased = 1 - stats.beta.cdf(0.5, alpha_post, beta_post)
print(f"P(theta > 0.5 | data) = {p_biased:.3f}")

# Plot prior, likelihood, and posterior
theta = np.linspace(0, 1, 1000)
prior = stats.beta.pdf(theta, alpha_prior, beta_prior)
likelihood = stats.binom.pmf(n_heads, n_flips, theta)
likelihood = likelihood / likelihood.max()  # scale for plotting
posterior = stats.beta.pdf(theta, alpha_post, beta_post)

plt.figure(figsize=(10, 5))
plt.plot(theta, prior / prior.max(), 'b--', label='Prior')
plt.plot(theta, likelihood, 'g:', linewidth=2, label='Likelihood (scaled)')
plt.plot(theta, posterior / posterior.max(), 'r-', linewidth=2, label='Posterior')
plt.axvline(n_heads/n_flips, color='gray', linestyle=':', alpha=0.5, label='MLE')
plt.xlabel('theta (probability of heads)')
plt.ylabel('Density (normalized)')
plt.legend()
plt.title('Bayesian Updating: Prior x Likelihood = Posterior')
plt.show()`}
        />

        <CodeBlock
          language="python"
          title="medical_test_bayes.py"
          code={`from scipy import stats

# =============================================
# Medical test problem (base rate fallacy)
# =============================================

prevalence = 0.001        # P(disease) = 1 in 1000
sensitivity = 0.99        # P(positive | disease) = 99%
specificity = 0.95        # P(negative | healthy) = 95%
false_positive_rate = 1 - specificity  # P(positive | healthy) = 5%

# Apply Bayes' theorem
# P(disease | positive) = P(positive | disease) * P(disease) / P(positive)
p_positive = (sensitivity * prevalence +
              false_positive_rate * (1 - prevalence))

p_disease_given_positive = (sensitivity * prevalence) / p_positive

print(f"P(disease | positive test) = {p_disease_given_positive:.4f}")
print(f"That is only {p_disease_given_positive*100:.1f}%!")
print(f"Most positive results ({(1-p_disease_given_positive)*100:.1f}%) are FALSE positives.")

# Sequential updates: what if you test positive TWICE (independent tests)?
# First positive: posterior becomes prior for second test
prior_2 = p_disease_given_positive
p_positive_2 = (sensitivity * prior_2 +
                false_positive_rate * (1 - prior_2))
p_disease_given_two_positives = (sensitivity * prior_2) / p_positive_2
print(f"\\nP(disease | TWO positive tests) = {p_disease_given_two_positives:.4f}")
print(f"Now it is {p_disease_given_two_positives*100:.1f}% — much more convincing.")`}
        />

        <CodeBlock
          language="python"
          title="bayesian_ab_test.py"
          code={`import numpy as np
from scipy import stats

np.random.seed(42)

# =============================================
# Bayesian A/B test for conversion rates
# =============================================

# Observed data
control_conversions, control_visitors = 120, 2000
treatment_conversions, treatment_visitors = 145, 2000

# Prior: Beta(1, 1) = Uniform (uninformative)
alpha_prior, beta_prior = 1, 1

# Posterior distributions for each group
alpha_c = alpha_prior + control_conversions
beta_c = beta_prior + control_visitors - control_conversions
alpha_t = alpha_prior + treatment_conversions
beta_t = beta_prior + treatment_visitors - treatment_conversions

print(f"Control:   {control_conversions}/{control_visitors} = "
      f"{control_conversions/control_visitors:.3f}")
print(f"Treatment: {treatment_conversions}/{treatment_visitors} = "
      f"{treatment_conversions/treatment_visitors:.3f}")

# Monte Carlo: P(treatment > control)
n_samples = 100000
samples_c = np.random.beta(alpha_c, beta_c, n_samples)
samples_t = np.random.beta(alpha_t, beta_t, n_samples)

p_treatment_wins = (samples_t > samples_c).mean()
print(f"\\nP(treatment > control) = {p_treatment_wins:.4f}")

# Expected lift
lift_samples = (samples_t - samples_c) / samples_c
print(f"Expected relative lift: {lift_samples.mean()*100:.2f}%")
print(f"95% credible interval for lift: "
      f"({np.percentile(lift_samples, 2.5)*100:.2f}%, "
      f"{np.percentile(lift_samples, 97.5)*100:.2f}%)")

# Risk analysis: expected loss if we pick the wrong variant
loss_if_pick_treatment = np.maximum(samples_c - samples_t, 0).mean()
loss_if_pick_control = np.maximum(samples_t - samples_c, 0).mean()
print(f"\\nExpected loss if we ship treatment: {loss_if_pick_treatment:.6f}")
print(f"Expected loss if we keep control:  {loss_if_pick_control:.6f}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Bayesian A/B testing</strong>: Instead of p-values, Bayesian A/B tests give you <InlineMath math="P(\text{treatment better})" /> — the probability that treatment wins. This is what stakeholders actually want to know. You can also compute expected loss to make risk-aware decisions.</li>
          <li><strong>Choosing priors</strong>: Use weakly informative priors (e.g., <InlineMath math="\text{Beta}(1, 1)" /> for proportions, <InlineMath math="\mathcal{N}(0, 10)" /> for regression coefficients). They prevent pathological results without imposing strong assumptions. If you have genuine domain knowledge, encode it — that&apos;s the power of Bayesian methods.</li>
          <li><strong>Conjugate priors simplify computation</strong>: Beta-Binomial, Normal-Normal, and Gamma-Poisson are conjugate pairs where the posterior is the same family as the prior. This gives closed-form posteriors without MCMC.</li>
          <li><strong>Bayesian vs. frequentist in practice</strong>: For well-powered, pre-registered confirmatory tests, frequentist methods are simpler and widely understood. Bayesian methods shine in sequential testing (no need for fixed sample sizes), incorporating prior information, and quantifying uncertainty.</li>
          <li><strong>Prior sensitivity analysis</strong>: Always check how your conclusions change with different reasonable priors. If the posterior is sensitive to the prior, you need more data.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Confusing prior and posterior probabilities</strong>: &quot;The test is 99% accurate so I probably have the disease&quot; ignores the base rate. Always start with the prior (prevalence) and update with the likelihood (test accuracy).</li>
          <li><strong>Flat priors are not always &quot;objective&quot;</strong>: A flat prior on <InlineMath math="\theta" /> is an informative prior on <InlineMath math="\log(\theta)" /> and vice versa. The &quot;uninformative&quot; prior depends on the parameterization. Jeffreys priors are invariant to reparameterization.</li>
          <li><strong>Ignoring the base rate (base rate fallacy)</strong>: This is the most common Bayesian mistake in everyday reasoning. Rare events with imperfect tests produce many false positives. Always consider <InlineMath math="P(A)" /> before computing <InlineMath math="P(A \mid B)" />.</li>
          <li><strong>Thinking more data always helps equally</strong>: Due to the <InlineMath math="1/\sqrt{n}" /> relationship, going from 100 to 200 observations helps more than going from 10,000 to 10,100. Early data is most informative relative to the prior.</li>
          <li><strong>Not checking model fit</strong>: Bayesian methods give beautiful posteriors even when the model is wrong. Use posterior predictive checks — sample from the posterior, generate fake data, and see if it looks like your real data.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> A spam filter uses Bayesian reasoning. 1% of emails are spam. A spam email contains the word &quot;free&quot; 80% of the time. A legitimate email contains &quot;free&quot; 10% of the time. An email arrives containing &quot;free.&quot; What is the probability it&apos;s spam? If the email also contains &quot;winner&quot; (found in 50% of spam, 1% of legitimate emails), what is the updated probability?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>Define: <InlineMath math="P(S) = 0.01" />, <InlineMath math="P(\text{free} \mid S) = 0.80" />, <InlineMath math="P(\text{free} \mid \neg S) = 0.10" /></li>
          <li>Apply Bayes&apos; theorem:
            <BlockMath math="P(S \mid \text{free}) = \frac{0.80 \times 0.01}{0.80 \times 0.01 + 0.10 \times 0.99} = \frac{0.008}{0.107} = 0.0748" />
          </li>
          <li>About 7.5% chance it&apos;s spam — still mostly likely legitimate because spam is rare.</li>
          <li>Now update with &quot;winner&quot; (assuming conditional independence given spam/not-spam). Use the posterior from step 2 as the new prior: <InlineMath math="P(S) = 0.0748" />
            <BlockMath math="P(S \mid \text{free}, \text{winner}) = \frac{0.50 \times 0.0748}{0.50 \times 0.0748 + 0.01 \times 0.9252} = \frac{0.0374}{0.0467} = 0.801" />
          </li>
          <li>After two words, the probability jumps to ~80%. Each piece of evidence updates the belief sequentially — this is exactly how Naive Bayes classifiers work.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>McElreath (2020) &quot;Statistical Rethinking&quot;</strong> — The best introduction to Bayesian statistics, with intuitive explanations and code in R/Python.</li>
          <li><strong>Gelman et al. (2013) &quot;Bayesian Data Analysis&quot;</strong> — The comprehensive reference for Bayesian methods (BDA3). Available free online.</li>
          <li><strong>Kruschke (2014) &quot;Doing Bayesian Data Analysis&quot;</strong> — A gentle, example-driven introduction known as the &quot;puppy book.&quot;</li>
          <li><strong>3Blue1Brown &quot;Bayes theorem, the geometry of changing beliefs&quot;</strong> — Excellent visual explanation on YouTube.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
