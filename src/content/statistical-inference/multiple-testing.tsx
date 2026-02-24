"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function MultipleTesting() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          When you run a single hypothesis test at <InlineMath math="\alpha = 0.05" />, you have a 5%
          chance of a false positive. But what happens when you run 20 tests? The probability
          of <em>at least one</em> false positive jumps to <InlineMath math="1 - 0.95^{20} = 0.64" /> —
          a 64% chance. This is the <strong>multiple testing problem</strong>, and it&apos;s one of the
          most common sources of false discoveries in science and industry.
        </p>
        <p>
          Imagine you&apos;re analyzing an A/B test and you look at 15 metrics: conversion rate,
          revenue, session duration, pages per visit, bounce rate, and so on. Even if the
          treatment has zero effect on everything, you&apos;d expect about one metric to show
          <InlineMath math="p < 0.05" /> purely by chance. This is why headlines like &quot;eating
          chocolate prevents cancer&quot; appear — researchers test enough outcomes that something
          will be &quot;significant&quot; by accident.
        </p>
        <p>
          There are two main philosophies for correction. <strong>Family-wise error rate (FWER)</strong> methods
          like Bonferroni and Holm control the probability of making <em>even one</em> false
          discovery — they&apos;re conservative. <strong>False discovery rate (FDR)</strong> methods like
          Benjamini-Hochberg control the expected <em>proportion</em> of false discoveries — they&apos;re
          more permissive but still provide meaningful guarantees. FDR is usually the right
          choice in exploratory settings where some false positives are acceptable.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>The Problem</h3>
        <p>
          Running <InlineMath math="m" /> independent tests at level <InlineMath math="\alpha" />,
          the family-wise error rate (FWER) is:
        </p>
        <BlockMath math="\text{FWER} = P(\text{at least one false positive}) = 1 - (1 - \alpha)^m" />

        <h3>Bonferroni Correction</h3>
        <p>The simplest FWER control: reject <InlineMath math="H_{0,i}" /> only if:</p>
        <BlockMath math="p_i \leq \frac{\alpha}{m}" />
        <p>
          This guarantees <InlineMath math="\text{FWER} \leq \alpha" /> regardless of dependence
          between tests. It&apos;s valid but conservative — as <InlineMath math="m" /> grows, the
          threshold becomes very stringent and you lose power.
        </p>

        <h3>Holm-Bonferroni (Step-Down) Procedure</h3>
        <p>
          A uniformly more powerful alternative to Bonferroni that also controls FWER at level <InlineMath math="\alpha" />:
        </p>
        <ol>
          <li>Order the p-values: <InlineMath math="p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}" /></li>
          <li>Find the smallest <InlineMath math="k" /> such that <InlineMath math="p_{(k)} > \frac{\alpha}{m - k + 1}" /></li>
          <li>Reject all hypotheses <InlineMath math="H_{(1)}, \ldots, H_{(k-1)}" /></li>
        </ol>
        <p>
          The thresholds are <InlineMath math="\frac{\alpha}{m}, \frac{\alpha}{m-1}, \frac{\alpha}{m-2}, \ldots" /> — progressively
          less strict. Holm is always at least as powerful as Bonferroni and there&apos;s no reason not to use it.
        </p>

        <h3>Benjamini-Hochberg (FDR Control)</h3>
        <p>
          Controls the <strong>false discovery rate</strong>: the expected proportion of rejected hypotheses that are false positives:
        </p>
        <BlockMath math="\text{FDR} = E\left[\frac{\text{false positives}}{\text{total rejections}}\right]" />
        <p>Procedure:</p>
        <ol>
          <li>Order p-values: <InlineMath math="p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}" /></li>
          <li>Find the largest <InlineMath math="k" /> such that <InlineMath math="p_{(k)} \leq \frac{k}{m} \cdot \alpha" /></li>
          <li>Reject all hypotheses <InlineMath math="H_{(1)}, \ldots, H_{(k)}" /></li>
        </ol>
        <p>
          Under independence (or positive regression dependence), this guarantees <InlineMath math="\text{FDR} \leq \alpha" />.
          BH is much more powerful than Bonferroni: with <InlineMath math="m = 1000" /> tests and 50 true
          effects, Bonferroni might find 5 while BH finds 40.
        </p>

        <h3>Adjusted P-Values</h3>
        <p>
          Instead of adjusting the threshold, you can adjust the p-values and compare to the
          original <InlineMath math="\alpha" />:
        </p>
        <BlockMath math="p_i^{\text{Bonf}} = \min(m \cdot p_i, \; 1)" />
        <BlockMath math="p_{(i)}^{\text{BH}} = \min\left(\frac{m}{i} \cdot p_{(i)}, \; 1\right) \quad \text{(monotonized)}" />
      </TopicSection>

      <TopicSection type="code">
        <CodeBlock
          language="python"
          title="multiple_testing_demo.py"
          code={`import numpy as np
from scipy import stats

np.random.seed(42)

# ===========================================================
# Simulate the multiple testing problem
# 20 tests, only 3 have real effects
# ===========================================================
m = 20
n_true_effects = 3
effect_size = 0.5  # Cohen's d

p_values = []
truths = []  # True = H0 is false (real effect)

for i in range(m):
    n = 100
    group_a = np.random.normal(0, 1, n)
    if i < n_true_effects:
        group_b = np.random.normal(effect_size, 1, n)  # real effect
        truths.append(True)
    else:
        group_b = np.random.normal(0, 1, n)  # no effect
        truths.append(False)
    _, p = stats.ttest_ind(group_a, group_b)
    p_values.append(p)

p_values = np.array(p_values)
truths = np.array(truths)

# No correction: which are "significant"?
sig_uncorrected = p_values < 0.05
print("=== No Correction ===")
print(f"Rejected: {sig_uncorrected.sum()}")
print(f"True positives:  {(sig_uncorrected & truths).sum()}")
print(f"False positives: {(sig_uncorrected & ~truths).sum()}")`}
        />

        <CodeBlock
          language="python"
          title="correction_methods.py"
          code={`import numpy as np
from statsmodels.stats.multitest import multipletests

# p_values and truths from previous block
# (Continuing the simulation above)

# ------ Bonferroni ------
reject_bonf, pvals_bonf, _, _ = multipletests(p_values, alpha=0.05,
                                                method='bonferroni')
print("=== Bonferroni (FWER control) ===")
print(f"Rejected: {reject_bonf.sum()}")
print(f"True positives:  {(reject_bonf & truths).sum()}")
print(f"False positives: {(reject_bonf & ~truths).sum()}")

# ------ Holm-Bonferroni ------
reject_holm, pvals_holm, _, _ = multipletests(p_values, alpha=0.05,
                                               method='holm')
print("\\n=== Holm-Bonferroni (FWER control, more powerful) ===")
print(f"Rejected: {reject_holm.sum()}")
print(f"True positives:  {(reject_holm & truths).sum()}")
print(f"False positives: {(reject_holm & ~truths).sum()}")

# ------ Benjamini-Hochberg ------
reject_bh, pvals_bh, _, _ = multipletests(p_values, alpha=0.05,
                                            method='fdr_bh')
print("\\n=== Benjamini-Hochberg (FDR control) ===")
print(f"Rejected: {reject_bh.sum()}")
print(f"True positives:  {(reject_bh & truths).sum()}")
print(f"False positives: {(reject_bh & ~truths).sum()}")

# Compare adjusted p-values
print("\\n=== Adjusted p-values for top 5 tests ===")
order = np.argsort(p_values)
print(f"{'Test':>6} {'Raw p':>10} {'Bonf':>10} {'Holm':>10} {'BH':>10} {'True?':>6}")
for i in order[:5]:
    print(f"{i:6d} {p_values[i]:10.6f} {pvals_bonf[i]:10.6f} "
          f"{pvals_holm[i]:10.6f} {pvals_bh[i]:10.6f} {'Yes' if truths[i] else 'No':>6}")`}
        />

        <CodeBlock
          language="python"
          title="bh_from_scratch.py"
          code={`import numpy as np

def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg procedure from scratch."""
    m = len(p_values)
    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]

    # BH threshold for each rank
    thresholds = np.arange(1, m + 1) / m * alpha

    # Find largest k where p_(k) <= k/m * alpha
    below = sorted_pvals <= thresholds
    if not below.any():
        return np.zeros(m, dtype=bool), np.ones(m)

    k = np.max(np.where(below)[0])

    # Reject all hypotheses with rank <= k
    rejected = np.zeros(m, dtype=bool)
    rejected[sorted_indices[:k + 1]] = True

    # Compute adjusted p-values (q-values)
    adjusted = np.zeros(m)
    adjusted[sorted_indices[-1]] = sorted_pvals[-1]
    for i in range(m - 2, -1, -1):
        adjusted[sorted_indices[i]] = min(
            adjusted[sorted_indices[i + 1]],
            sorted_pvals[i] * m / (i + 1)
        )

    return rejected, np.minimum(adjusted, 1.0)

# Test it
np.random.seed(42)
p_vals = np.array([0.001, 0.008, 0.039, 0.041, 0.23, 0.38, 0.52, 0.61, 0.79, 0.91])
rejected, q_values = benjamini_hochberg(p_vals, alpha=0.05)

print("p-value  | q-value  | Rejected")
print("-" * 35)
for p, q, r in zip(p_vals, q_values, rejected):
    print(f"{p:.3f}    | {q:.4f}   | {'Yes' if r else 'No'}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>A/B tests with multiple metrics</strong>: If you track 10 metrics in an A/B test, apply BH correction. Better yet, pre-register a single primary metric (no correction needed) and treat the rest as secondary/exploratory with FDR control.</li>
          <li><strong>Genomics and feature selection</strong>: In gene expression studies, you test thousands of genes simultaneously. BH-adjusted p-values (q-values) are standard. A q-value of 0.05 means that among all genes called significant, about 5% are expected to be false positives.</li>
          <li><strong>Bonferroni is too harsh for most applications</strong>: With 1000 tests, the Bonferroni threshold is 0.00005 — you&apos;ll miss most real effects. Use BH unless you absolutely cannot tolerate any false positives (safety-critical applications).</li>
          <li><strong>Correlated tests</strong>: BH is valid under positive dependence (PRDS), which covers most practical cases. For arbitrary dependence, use Benjamini-Yekutieli (BY), which divides by an extra <InlineMath math="\sum_{i=1}^{m} 1/i" /> factor.</li>
          <li><strong>Pre-registration kills p-hacking</strong>: The most powerful multiple testing correction is not testing multiple hypotheses in the first place. Pre-register your primary hypothesis before seeing data.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Not correcting at all</strong>: The most common mistake. If you test 50 features for correlation with the outcome and report the 3 with <InlineMath math="p < 0.05" />, you haven&apos;t found anything — you&apos;d expect 2.5 false positives by chance.</li>
          <li><strong>Correcting for unrelated tests</strong>: You don&apos;t need to correct across unrelated studies or completely separate analyses. Correction applies to a &quot;family&quot; of related tests — those that address the same overall question.</li>
          <li><strong>Using Bonferroni when BH is appropriate</strong>: In exploratory analysis, FDR control (BH) is usually what you want. FWER control (Bonferroni) is for confirmatory settings where each individual claim needs to be reliable.</li>
          <li><strong>Confusing q-values with p-values</strong>: A BH-adjusted p-value (q-value) of 0.05 means that 5% of findings <em>at this threshold or below</em> are expected to be false. It does not mean there&apos;s a 5% chance <em>this particular</em> finding is false.</li>
          <li><strong>Forgetting implicit multiplicity</strong>: Even if you run &quot;one test,&quot; if you chose which variable to test after looking at the data, you&apos;ve implicitly performed multiple tests. This is the essence of p-hacking.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You run an A/B test and measure 8 metrics. Three come back with p-values of 0.012, 0.034, and 0.048. The rest are all above 0.2. Can you claim these three are significant findings?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Without correction</strong>: All three are below 0.05, so they look significant. But with 8 tests, we expect <InlineMath math="8 \times 0.05 = 0.4" /> false positives on average.</li>
          <li><strong>Bonferroni</strong>: The adjusted threshold is <InlineMath math="0.05 / 8 = 0.00625" />. None of the three pass. This is overly conservative.</li>
          <li><strong>Holm</strong>: Sort p-values. Compare <InlineMath math="0.012" /> to <InlineMath math="0.05/8 = 0.00625" /> — fails. So Holm also rejects nothing. Still conservative.</li>
          <li><strong>Benjamini-Hochberg</strong>: Sort all 8 p-values. The BH thresholds are <InlineMath math="k/8 \times 0.05" />. Checking: <InlineMath math="p_{(1)} = 0.012 \leq 0.00625" />? No. <InlineMath math="p_{(1)} = 0.012 \leq 1/8 \times 0.05 = 0.00625" /> fails, so BH also rejects nothing here.</li>
          <li><strong>Best practice</strong>: Designate one <em>primary</em> metric before the experiment. Test it at <InlineMath math="\alpha = 0.05" /> with no correction. Treat the remaining 7 as exploratory and apply BH. This gives maximum power for the question you care most about.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Benjamini &amp; Hochberg (1995) &quot;Controlling the False Discovery Rate&quot;</strong> — The seminal paper introducing FDR, one of the most cited statistics papers ever.</li>
          <li><strong>Storey (2003) &quot;The positive false discovery rate&quot;</strong> — Introduces the q-value and Storey&apos;s FDR estimation, more powerful than BH when many nulls are true.</li>
          <li><strong>Holm (1979) &quot;A simple sequentially rejective multiple test procedure&quot;</strong> — The original step-down procedure that dominates Bonferroni.</li>
          <li><strong>XKCD #882 &quot;Significant&quot;</strong> — A memorable comic illustrating the multiple testing problem with jelly beans and acne.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
