"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function RLHF() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          A pre-trained language model can generate fluent text, but it doesn&apos;t know what humans <strong>want</strong>. It might
          produce toxic content, hallucinate facts, or give unhelpful responses — because its only objective was to predict the next
          token in internet text. <strong>Alignment</strong> is the process of steering the model to be helpful, harmless, and honest.
        </p>
        <p>
          <strong>RLHF (Reinforcement Learning from Human Feedback)</strong> is the dominant alignment technique. It works in three stages:
          (1) <strong>Supervised fine-tuning (SFT)</strong> on high-quality demonstrations of desired behavior,
          (2) <strong>Reward modeling</strong> — train a model to predict human preferences by comparing pairs of responses, and
          (3) <strong>RL optimization</strong> — use Proximal Policy Optimization (PPO) to fine-tune the SFT model to maximize the
          learned reward while staying close to the original model.
        </p>
        <p>
          <strong>DPO (Direct Preference Optimization)</strong> simplifies this pipeline by eliminating the reward model and RL loop entirely.
          It directly optimizes the language model on preference pairs using a clever reformulation of the RLHF objective as a simple
          classification loss. DPO is much simpler to implement, more stable to train, and has become the preferred method for many teams.
        </p>
        <p>
          The alignment pipeline is what turns a base model (like LLaMA) into a helpful assistant (like ChatGPT). Without it, even the
          most capable base model is difficult to use — it will continue text rather than follow instructions, and has no notion of
          safety or helpfulness.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Stage 1: Supervised Fine-Tuning (SFT)</h3>
        <p>
          Fine-tune the pre-trained model <InlineMath math="\pi_{\text{base}}" /> on demonstrations of desired behavior
          (instruction-response pairs) using the standard causal LM objective:
        </p>
        <BlockMath math="\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{demo}}} \left[ \sum_{t=1}^{|y|} \log \pi_{\text{SFT}}(y_t \mid x, y_{<t}) \right]" />

        <h3>Stage 2: Reward Modeling</h3>
        <p>
          Given a prompt <InlineMath math="x" /> and two completions <InlineMath math="y_w" /> (preferred) and <InlineMath math="y_l" /> (rejected),
          train a reward model <InlineMath math="r_\phi" /> using the Bradley-Terry preference model:
        </p>
        <BlockMath math="P(y_w \succ y_l \mid x) = \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)" />
        <p>The loss over the preference dataset <InlineMath math="\mathcal{D}" />:</p>
        <BlockMath math="\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right) \right]" />

        <h3>Stage 3: PPO Optimization</h3>
        <p>
          The RL objective maximizes the reward while staying close to the SFT model via a KL penalty:
        </p>
        <BlockMath math="\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \, D_{\text{KL}}\left(\pi_\theta(\cdot|x) \| \pi_{\text{SFT}}(\cdot|x)\right) \right]" />
        <p>
          The KL constraint is critical — without it, the model collapses to degenerate high-reward outputs (reward hacking).
          <InlineMath math="\beta" /> controls the tradeoff: higher <InlineMath math="\beta" /> keeps outputs closer to the SFT model.
        </p>

        <p>The PPO clipped surrogate objective for each token:</p>
        <BlockMath math="\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)} \hat{A}_t, \; \text{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right)\hat{A}_t\right)\right]" />
        <p>
          where <InlineMath math="\hat{A}_t" /> is the advantage estimate (computed from the reward model&apos;s scores),
          <InlineMath math="\epsilon" /> is the clipping range (typically 0.2), and actions <InlineMath math="a_t" /> are token choices.
        </p>

        <h3>DPO (Direct Preference Optimization)</h3>
        <p>
          DPO shows that the optimal policy under the RLHF objective has a closed-form relationship with the reward:
        </p>
        <BlockMath math="r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + C(x)" />
        <p>Substituting into the Bradley-Terry model yields a loss directly on the policy, with no reward model needed:</p>
        <BlockMath math="\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right) \right]" />
        <p>
          This is just a binary cross-entropy loss on log-probability ratios. No RL, no reward model, no value function —
          just a supervised loss on preference pairs.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Training a Reward Model</h3>
        <CodeBlock
          language="python"
          title="reward_model.py"
          code={`import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class RewardModel(nn.Module):
    """Reward model based on a pre-trained LM with a scalar head."""
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf"):
        super().__init__()
        # Use the LM backbone with a single-output classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            torch_dtype=torch.float16,
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)  # scalar reward per sequence

def reward_loss(reward_model, chosen_ids, chosen_mask, rejected_ids, rejected_mask):
    """Bradley-Terry preference loss."""
    r_chosen = reward_model(chosen_ids, chosen_mask)
    r_rejected = reward_model(rejected_ids, rejected_mask)

    # Loss: -log(sigmoid(r_chosen - r_rejected))
    loss = -torch.log(torch.sigmoid(r_chosen - r_rejected) + 1e-10).mean()

    # Accuracy: how often does the model rank the chosen response higher?
    accuracy = (r_chosen > r_rejected).float().mean()
    return loss, accuracy

# Training loop sketch
# for batch in preference_dataloader:
#     loss, acc = reward_loss(
#         reward_model, batch["chosen_ids"], batch["chosen_mask"],
#         batch["rejected_ids"], batch["rejected_mask"]
#     )
#     loss.backward()
#     optimizer.step()`}
        />

        <h3>RLHF with PPO using TRL</h3>
        <CodeBlock
          language="python"
          title="rlhf_ppo.py"
          code={`from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
import torch

# 1. Load SFT model and add a value head for PPO
model = AutoModelForCausalLMWithValueHead.from_pretrained("my-sft-model")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("my-sft-model")
tokenizer = AutoTokenizer.from_pretrained("my-sft-model")
tokenizer.pad_token = tokenizer.eos_token

# 2. Load reward model (trained separately)
reward_model = ...  # your trained reward model

# 3. PPO config
ppo_config = PPOConfig(
    batch_size=64,
    mini_batch_size=16,
    learning_rate=1.5e-5,
    ppo_epochs=4,                # PPO optimization epochs per batch
    init_kl_coef=0.2,           # beta: KL penalty coefficient
    target_kl=6.0,              # adaptive KL target
    cliprange=0.2,              # PPO clipping epsilon
    gamma=1.0,                  # no discounting (full episode = one response)
    lam=0.95,                   # GAE lambda
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# 4. Training loop
for batch in prompt_dataloader:
    query_tensors = batch["input_ids"]

    # Generate responses from current policy
    response_tensors = ppo_trainer.generate(
        query_tensors, max_new_tokens=256, temperature=0.7
    )

    # Score with reward model
    texts = [tokenizer.decode(r) for r in response_tensors]
    rewards = [reward_model.score(t) for t in texts]
    rewards = [torch.tensor(r) for r in rewards]

    # PPO step: update policy to maximize reward with KL constraint
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    print(f"Mean reward: {stats['ppo/mean_scores']:.3f}, "
          f"KL: {stats['objective/kl']:.3f}")`}
        />

        <h3>DPO: The Simpler Alternative</h3>
        <CodeBlock
          language="python"
          title="dpo_training.py"
          code={`from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load the SFT model (this becomes both policy and reference)
model = AutoModelForCausalLM.from_pretrained(
    "my-sft-model", torch_dtype=torch.float16, device_map="auto"
)
ref_model = AutoModelForCausalLM.from_pretrained(
    "my-sft-model", torch_dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("my-sft-model")
tokenizer.pad_token = tokenizer.eos_token

# Load preference dataset
# Each example: {"prompt": str, "chosen": str, "rejected": str}
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

# DPO config
dpo_config = DPOConfig(
    output_dir="./dpo-model",
    beta=0.1,                    # KL constraint strength
    learning_rate=5e-7,          # very low LR for stability
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    warmup_ratio=0.1,
    fp16=True,
    logging_steps=10,
    max_length=512,
    max_prompt_length=256,
)

# DPO trainer — no reward model needed!
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

dpo_trainer.train()

# That's it! No reward model, no RL loop, no value function.
# The model directly learns from preference pairs.`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Data quality is paramount</strong>: The quality of human preference data determines alignment quality. Inconsistent or noisy labels (disagreeing annotators) make the reward model unreliable, which cascades into bad RL optimization.</li>
          <li><strong>Reward hacking is real</strong>: The model will find ways to maximize reward that don&apos;t align with human intent. Common failure: producing verbose, confident-sounding text that gets high reward scores but is factually wrong. The KL penalty is the main defense.</li>
          <li><strong>DPO is winning in practice</strong>: Most open-source alignment projects (Zephyr, Mistral, etc.) use DPO over PPO because it&apos;s dramatically simpler and produces comparable results. PPO requires maintaining 4 models simultaneously (policy, reference, reward, value) vs. DPO&apos;s 2 (policy, reference).</li>
          <li><strong>Iterative RLHF</strong>: State-of-the-art alignment often uses multiple rounds: SFT, then RLHF, then collect new preferences on the RLHF model, then train a new reward model, then more RLHF. Each round improves the model and the quality of feedback.</li>
          <li><strong>Constitutional AI</strong>: Anthropic&apos;s approach uses AI feedback instead of human feedback for some stages — the model critiques and revises its own responses according to a constitution of principles.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Skipping the SFT stage</strong>: Applying RLHF directly to a base model usually fails. The base model doesn&apos;t know how to follow instructions, so it can&apos;t generate responses for the reward model to score. SFT is essential to bootstrap instruction-following.</li>
          <li><strong>Too low KL penalty (beta)</strong>: The model over-optimizes the reward model, finding adversarial outputs that score high but are gibberish or repetitive. If generated text degrades during training, increase beta.</li>
          <li><strong>Too high KL penalty</strong>: The model barely moves from the SFT checkpoint. Outputs remain unchanged despite training. The reward scores don&apos;t improve because the policy can&apos;t explore.</li>
          <li><strong>Reward model scale mismatch</strong>: If reward scores are very large, the sigmoid saturates and gradients vanish. Normalize reward model outputs or use a smaller learning rate.</li>
          <li><strong>Using DPO with bad reference model</strong>: In DPO, the reference model defines the baseline. If it&apos;s too far from the training distribution (e.g., using a base model as reference instead of the SFT model), training is unstable.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Walk through the RLHF pipeline from start to finish. What is the role of each stage? How does DPO simplify this, and what are the tradeoffs?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Stage 1 — SFT</strong>: Fine-tune the base model on high-quality (prompt, response) demonstrations. This teaches the model to follow instructions and produce well-formatted responses. Output: <InlineMath math="\pi_{\text{SFT}}" />.
          </li>
          <li>
            <strong>Stage 2 — Reward Modeling</strong>: Collect human preference data by showing annotators pairs of responses and asking which is better. Train a reward model <InlineMath math="r_\phi" /> using the Bradley-Terry loss: <InlineMath math="-\log \sigma(r(y_w) - r(y_l))" />. The reward model scores any response on a scalar scale.
          </li>
          <li>
            <strong>Stage 3 — PPO</strong>: Use the reward model to optimize the policy. For each prompt, generate a response, score it, and compute the PPO update. The KL penalty <InlineMath math="\beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{SFT}})" /> prevents the model from drifting too far.
          </li>
          <li>
            <strong>DPO simplification</strong>: DPO eliminates stages 2 and 3 entirely. It shows that the optimal policy satisfies <InlineMath math="r^*(x,y) = \beta \log(\pi^*(y|x)/\pi_{\text{ref}}(y|x)) + C" />, so we can directly optimize the policy on preference pairs with a supervised loss. No reward model, no RL loop.
          </li>
          <li>
            <strong>Tradeoffs</strong>:
            <ul>
              <li>PPO is more flexible — can use the reward model for rejection sampling, best-of-n, or iterative refinement.</li>
              <li>DPO is simpler, more stable, and requires less compute and engineering effort.</li>
              <li>PPO can explore beyond the training data distribution (on-policy). DPO is purely offline (off-policy), limited to the quality of the collected preference pairs.</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Ouyang et al. (2022) &quot;Training language models to follow instructions with human feedback&quot;</strong> — The InstructGPT paper that introduced the 3-stage RLHF pipeline.</li>
          <li><strong>Rafailov et al. (2023) &quot;Direct Preference Optimization: Your Language Model is Secretly a Reward Model&quot;</strong> — The DPO paper that eliminates the RL loop.</li>
          <li><strong>Schulman et al. (2017) &quot;Proximal Policy Optimization Algorithms&quot;</strong> — The original PPO paper from OpenAI.</li>
          <li><strong>Bai et al. (2022) &quot;Constitutional AI: Harmlessness from AI Feedback&quot;</strong> — Anthropic&apos;s approach to alignment using AI-generated feedback.</li>
          <li><strong>HuggingFace TRL library</strong> — Production-quality implementations of SFT, reward modeling, PPO, and DPO trainers.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
