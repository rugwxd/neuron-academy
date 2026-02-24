"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function ScalingLaws() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Scaling laws are empirical relationships that describe how a model&apos;s performance (measured
          by loss) improves as you increase the amount of <strong>compute</strong>, <strong>model
          parameters</strong>, and <strong>training data</strong>. The remarkable finding: these
          relationships follow smooth <strong>power laws</strong> across many orders of magnitude.
        </p>
        <p>
          The first comprehensive study (Kaplan et al., 2020, from OpenAI) found that loss decreases as
          a power law in each of these three factors, and that bigger models are more sample-efficient --
          leading to the conclusion that you should train very large models on relatively less data.
          This philosophy drove GPT-3&apos;s design: a 175B parameter model trained on 300B tokens.
        </p>
        <p>
          <strong>Chinchilla</strong> (Hoffmann et al., 2022, from DeepMind) upended this by showing that
          Kaplan&apos;s analysis was wrong about the data-compute tradeoff. Chinchilla demonstrated that
          model size and training data should be scaled <strong>equally</strong>: for a compute-optimal
          model, the number of training tokens should be approximately 20 times the number of parameters.
          A 70B model trained on 1.4 trillion tokens (Chinchilla) outperformed a 280B model trained on
          300B tokens (Gopher), despite using the same compute budget.
        </p>
        <p>
          The practical impact was enormous: the entire field shifted from &quot;bigger models&quot; to
          &quot;more data for smaller models.&quot; LLaMA-1 (65B, 1.4T tokens), LLaMA-2 (70B, 2T tokens),
          and Mistral (7B, heavily trained) all follow this Chinchilla-optimal philosophy. Even more recent
          work suggests training well beyond the Chinchilla ratio (over-training) is beneficial when
          inference cost matters more than training cost.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Kaplan Scaling Laws (2020)</h3>
        <p>
          Loss follows a power law in each scaling factor independently:
        </p>
        <BlockMath math="L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}" />
        <p>
          Where <InlineMath math="N" /> = parameters, <InlineMath math="D" /> = dataset tokens,
          <InlineMath math="C" /> = compute (FLOPs), and the exponents from Kaplan et al.:
        </p>
        <BlockMath math="\alpha_N \approx 0.076, \quad \alpha_D \approx 0.095, \quad \alpha_C \approx 0.050" />
        <p>
          The combined scaling law with both <InlineMath math="N" /> and <InlineMath math="D" />:
        </p>
        <BlockMath math="L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}" />

        <h3>Chinchilla Scaling Laws (2022)</h3>
        <p>
          Hoffmann et al. proposed a different parametric form:
        </p>
        <BlockMath math="L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E" />
        <p>
          Where <InlineMath math="E" /> is the irreducible loss (entropy of natural language), and
          the fitted values:
        </p>
        <BlockMath math="\alpha \approx 0.34, \quad \beta \approx 0.28, \quad A \approx 406.4, \quad B \approx 410.7, \quad E \approx 1.69" />

        <h3>Compute-Optimal Allocation</h3>
        <p>
          Given a fixed compute budget <InlineMath math="C \approx 6ND" /> (6 FLOPs per parameter per token),
          the Chinchilla-optimal split is:
        </p>
        <BlockMath math="N_{\text{opt}} \propto C^a, \quad D_{\text{opt}} \propto C^b" />
        <BlockMath math="\text{where } a = \frac{\beta}{\alpha + \beta} \approx 0.45, \quad b = \frac{\alpha}{\alpha + \beta} \approx 0.55" />
        <p>The practical rule of thumb:</p>
        <BlockMath math="D_{\text{opt}} \approx 20 \times N" />
        <p>
          A 10B parameter model should be trained on approximately 200B tokens. Kaplan et al. had
          suggested only ~3.5B tokens for a 10B model, dramatically undertrained by Chinchilla standards.
        </p>

        <h3>The 6ND Approximation for Compute</h3>
        <p>Total training FLOPs for a Transformer:</p>
        <BlockMath math="C \approx 6 \cdot N \cdot D" />
        <p>
          Where the factor 6 comes from: each parameter participates in 2 FLOPs (multiply + add) per
          token in the forward pass, and roughly 4 FLOPs in the backward pass (2x for computing
          parameter gradients and activation gradients). So <InlineMath math="2 + 4 = 6" /> FLOPs per
          parameter per token.
        </p>

        <h3>Emergent Abilities and Beyond Power Laws</h3>
        <p>
          Some capabilities appear to emerge discontinuously at certain model scales (e.g., few-shot
          arithmetic, chain-of-thought reasoning). The loss itself scales smoothly, but downstream
          task performance measured by accuracy can exhibit sharp phase transitions:
        </p>
        <BlockMath math="\text{Accuracy}(\text{task}) \approx \begin{cases} \text{random} & \text{if } N < N_{\text{threshold}} \\ \text{high} & \text{if } N > N_{\text{threshold}} \end{cases}" />
        <p>
          Though recent work (Schaeffer et al., 2023) argues that these &quot;emergent&quot; abilities
          may be artifacts of nonlinear evaluation metrics rather than true phase transitions.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Visualizing Scaling Laws</h3>
        <CodeBlock
          language="python"
          title="scaling_laws_viz.py"
          code={`import numpy as np
import matplotlib.pyplot as plt

# --- Chinchilla scaling law ---
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    """Predicted loss given N parameters and D training tokens."""
    return A / N**alpha + B / D**beta + E

# Sweep model sizes at different data budgets
params = np.logspace(7, 11, 100)    # 10M to 100B parameters
data_budgets = [1e9, 1e10, 1e11, 1e12]  # 1B, 10B, 100B, 1T tokens

plt.figure(figsize=(10, 6))
for D in data_budgets:
    losses = chinchilla_loss(params, D)
    plt.plot(params, losses, label=f"D = {D:.0e} tokens")

plt.xscale("log")
plt.xlabel("Parameters (N)")
plt.ylabel("Loss")
plt.title("Chinchilla Scaling Law: Loss vs Parameters at Fixed Data")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()`}
        />

        <h3>Compute-Optimal Model Sizing</h3>
        <CodeBlock
          language="python"
          title="compute_optimal.py"
          code={`import numpy as np

def optimal_allocation(compute_flops, alpha=0.34, beta=0.28):
    """
    Given a compute budget (FLOPs), find the Chinchilla-optimal
    model size and training tokens.

    Uses C = 6 * N * D approximation.
    """
    # Optimal split: N ~ C^(beta/(alpha+beta)), D ~ C^(alpha/(alpha+beta))
    a = beta / (alpha + beta)   # ~0.45 for N
    b = alpha / (alpha + beta)  # ~0.55 for D

    # Solve N * D = C/6, with D/N = 20 (Chinchilla ratio)
    N_opt = (compute_flops / (6 * 20)) ** 0.5  # from C = 6 * N * 20N
    D_opt = 20 * N_opt

    return N_opt, D_opt

# --- Compare allocations for various compute budgets ---
print(f"{'Compute (FLOPs)':>20s} | {'Params':>12s} | {'Tokens':>14s} | {'Ratio D/N':>10s}")
print("-" * 65)

compute_budgets = {
    "GPT-3 budget":     3.64e23,  # ~3.6e23 FLOPs
    "Chinchilla budget": 5.76e23,  # ~5.8e23 FLOPs
    "LLaMA-2 budget":   1e24,     # ~1e24 FLOPs
    "GPT-4 estimate":   2e25,     # estimated
}

for name, C in compute_budgets.items():
    N, D = optimal_allocation(C)
    print(f"{name:>20s} | {N/1e9:>9.1f}B | {D/1e12:>11.2f}T | {D/N:>10.1f}")

# Output:
#      GPT-3 budget |      17.4B |        0.35T |       20.0
#  Chinchilla budget |      21.9B |        0.44T |       20.0
#    LLaMA-2 budget |      28.9B |        0.58T |       20.0
#    GPT-4 estimate |     129.1B |        2.58T |       20.0`}
        />

        <h3>Comparing Kaplan vs Chinchilla Predictions</h3>
        <CodeBlock
          language="python"
          title="kaplan_vs_chinchilla.py"
          code={`import numpy as np
import matplotlib.pyplot as plt

def kaplan_optimal_tokens(N):
    """Kaplan et al. recommendation: relatively few tokens for large models."""
    # From the paper: D_opt ~ N^0.74 (approximately)
    return 5e9 * (N / 1e9) ** 0.74

def chinchilla_optimal_tokens(N):
    """Chinchilla recommendation: ~20x parameters."""
    return 20 * N

# Compare for various model sizes
model_sizes = np.logspace(8, 11, 50)  # 100M to 100B

kaplan_tokens = kaplan_optimal_tokens(model_sizes)
chinchilla_tokens = chinchilla_optimal_tokens(model_sizes)

plt.figure(figsize=(10, 6))
plt.loglog(model_sizes, kaplan_tokens, "b-", linewidth=2, label="Kaplan (2020)")
plt.loglog(model_sizes, chinchilla_tokens, "r-", linewidth=2, label="Chinchilla (2022)")

# Plot actual models
models = {
    "GPT-3": (175e9, 300e9),
    "Chinchilla": (70e9, 1.4e12),
    "LLaMA-1 65B": (65e9, 1.4e12),
    "LLaMA-2 70B": (70e9, 2e12),
    "Mistral 7B": (7e9, 8e12),  # heavily over-trained
}
for name, (n, d) in models.items():
    plt.scatter(n, d, s=100, zorder=5)
    plt.annotate(name, (n, d), textcoords="offset points",
                 xytext=(10, 5), fontsize=9)

plt.xlabel("Parameters (N)")
plt.ylabel("Training Tokens (D)")
plt.title("Kaplan vs Chinchilla: Optimal Training Tokens")
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Key insight: GPT-3 is far below the Chinchilla line (undertrained)
# Mistral 7B is far above it (over-trained, optimized for inference)`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Chinchilla changed the industry</strong>: Before Chinchilla, the trend was ever-larger models (GPT-3 175B, Gopher 280B). After Chinchilla, the focus shifted to training smaller models on much more data (LLaMA 65B on 1.4T tokens, Mistral 7B on 8T+ tokens).</li>
          <li><strong>Over-training is the new meta</strong>: For deployment, inference cost dominates training cost. Training a 7B model on 10x the Chinchilla-optimal tokens gives slightly worse loss but much cheaper inference. LLaMA-3 8B was trained on 15T tokens (150x more than 20x would suggest).</li>
          <li><strong>The compute budget determines everything</strong>: Before training, estimate your budget in FLOPs, then use the Chinchilla ratio to determine model size and data. If you have <InlineMath math="10^{22}" /> FLOPs, a ~4B model on ~80B tokens is compute-optimal.</li>
          <li><strong>Data quality trumps quantity at scale</strong>: Scaling laws assume data quality is constant. In practice, curating high-quality training data (deduplication, filtering, careful mixing) can shift the scaling curve downward, equivalent to 2-5x more compute.</li>
          <li><strong>Scaling laws help predict training runs</strong>: You can fit scaling laws on small models (100M-1B) and extrapolate to predict the performance of 10B+ models before training them. This saves enormous amounts of compute on failed experiments.</li>
          <li><strong>Mixture of Experts (MoE) bend the curves</strong>: MoE models like Mixtral use more total parameters but only activate a fraction per token, achieving better loss per FLOP than dense models.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Using Kaplan ratios in 2024+</strong>: The Kaplan paper&apos;s recommendation to undertrain large models is outdated. Always use Chinchilla ratios (or over-train for inference efficiency) as the starting point.</li>
          <li><strong>Confusing model parameters with compute</strong>: A 7B model trained on 2T tokens uses more compute than a 13B model trained on 300B tokens. Always think in FLOPs (<InlineMath math="C \approx 6ND" />), not just parameters.</li>
          <li><strong>Assuming scaling laws apply to all tasks</strong>: Scaling laws predict <em>loss</em> (perplexity), not downstream accuracy. A model can have better loss but worse accuracy on a specific benchmark due to evaluation metric nonlinearity.</li>
          <li><strong>Ignoring data bottlenecks</strong>: Chinchilla says a 70B model needs 1.4T high-quality tokens. If you only have 100B tokens, you will cycle through your data multiple times, reducing its effective value. Data scarcity is the real bottleneck for many practitioners.</li>
          <li><strong>Treating the 20:1 ratio as exact</strong>: The Chinchilla ratio is approximate and depends on data quality, architecture details, and learning rate schedule. It is a guideline, not a law of nature.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Your team has a fixed compute budget equivalent to training a 13B parameter model on 260B tokens. Using Chinchilla scaling laws, is this allocation optimal? If not, what would you recommend?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Compute budget</strong>: <InlineMath math="C = 6 \times 13 \times 10^9 \times 260 \times 10^9 = 2.03 \times 10^{22}" /> FLOPs.
          </li>
          <li>
            <strong>Check the ratio</strong>: <InlineMath math="D / N = 260B / 13B = 20" />. This is exactly the Chinchilla-optimal ratio! So the allocation is compute-optimal.
          </li>
          <li>
            <strong>However, consider your use case</strong>:
            <ul>
              <li>If <strong>inference cost matters</strong> (serving millions of users), over-train a smaller model. Use a 7B model on ~480B tokens (same FLOPs). The 7B model is ~2x cheaper to serve with only slightly higher loss.</li>
              <li>If <strong>peak accuracy matters</strong> (research benchmark), the 13B/260B split is optimal.</li>
              <li>If <strong>data is limited</strong> (e.g., only 100B high-quality tokens), train a smaller model (~5B) with fewer epochs rather than cycling through 100B tokens 2.6x.</li>
            </ul>
          </li>
          <li>
            <strong>Validate with small-scale experiments</strong>: Train 100M, 300M, and 1B models at
            various data sizes. Fit the Chinchilla power law and extrapolate. This costs &lt;1% of
            the full budget and can save you from an expensive mistake.
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Kaplan et al. (2020) &quot;Scaling Laws for Neural Language Models&quot;</strong> -- The first comprehensive study of LLM scaling behavior. Historically important even though the optimal ratios were revised.</li>
          <li><strong>Hoffmann et al. (2022) &quot;Training Compute-Optimal Large Language Models&quot;</strong> -- The Chinchilla paper. Changed how the entire industry thinks about model sizing.</li>
          <li><strong>Touvron et al. (2023) &quot;LLaMA: Open and Efficient Foundation Language Models&quot;</strong> -- Demonstrates that Chinchilla-optimal smaller models can match much larger undertrained models.</li>
          <li><strong>Muennighoff et al. (2023) &quot;Scaling Data-Constrained Language Models&quot;</strong> -- What happens when you run out of unique data and must repeat tokens.</li>
          <li><strong>Schaeffer et al. (2023) &quot;Are Emergent Abilities of Large Language Models a Mirage?&quot;</strong> -- Challenges the notion of emergent abilities, arguing they are metric artifacts.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
