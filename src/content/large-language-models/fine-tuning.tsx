"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function FineTuning() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          A pre-trained language model is a general-purpose text predictor. <strong>Fine-tuning</strong> adapts it to a specific task
          or domain — like turning a general practitioner into a specialist. Instead of training from scratch, you start with the
          pre-trained weights and continue training on a much smaller, task-specific dataset with a lower learning rate.
        </p>
        <p>
          <strong>Full fine-tuning</strong> updates every parameter in the model. This works well but is expensive for large models:
          a 7B parameter model requires ~28 GB just for the parameters in FP32, plus 2-3x that for optimizer states and gradients.
          For a 70B model, you&apos;d need a cluster of high-memory GPUs.
        </p>
        <p>
          <strong>Parameter-efficient fine-tuning (PEFT)</strong> methods solve this by updating only a small fraction of parameters.
          <strong>LoRA</strong> (Low-Rank Adaptation) is the most popular: it freezes the original weights and injects small trainable
          rank-decomposition matrices into each layer. Instead of updating a <InlineMath math="d \times d" /> weight matrix, you learn
          two small matrices <InlineMath math="A \in \mathbb{R}^{d \times r}" /> and <InlineMath math="B \in \mathbb{R}^{r \times d}" /> where <InlineMath math="r \ll d" /> (typically 8-64). This reduces trainable parameters by 100-1000x.
        </p>
        <p>
          <strong>QLoRA</strong> goes further by quantizing the frozen base model to 4-bit precision, making it possible to fine-tune
          a 65B parameter model on a single 48GB GPU. The combination of quantization + LoRA has democratized LLM fine-tuning, making
          it accessible to anyone with a consumer GPU.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Full Fine-tuning</h3>
        <p>
          Given a pre-trained model with parameters <InlineMath math="\theta_0" />, full fine-tuning optimizes:
        </p>
        <BlockMath math="\theta^* = \arg\min_\theta \mathcal{L}_{\text{task}}(\theta) + \lambda \|\theta - \theta_0\|^2" />
        <p>
          The regularization term <InlineMath math="\lambda \|\theta - \theta_0\|^2" /> (weight decay toward initialization) prevents
          <strong>catastrophic forgetting</strong> — drifting too far from the pre-trained knowledge.
        </p>

        <h3>LoRA (Low-Rank Adaptation)</h3>
        <p>
          For a pre-trained weight matrix <InlineMath math="W_0 \in \mathbb{R}^{d \times k}" />, LoRA parametrizes the update as a low-rank decomposition:
        </p>
        <BlockMath math="W = W_0 + \Delta W = W_0 + BA" />
        <p>
          where <InlineMath math="B \in \mathbb{R}^{d \times r}" />, <InlineMath math="A \in \mathbb{R}^{r \times k}" />, and <InlineMath math="r \ll \min(d, k)" />.
          The forward pass becomes:
        </p>
        <BlockMath math="h = W_0 x + \frac{\alpha}{r} BAx" />
        <p>
          where <InlineMath math="\alpha" /> is a scaling hyperparameter. Key details:
        </p>
        <ul>
          <li><InlineMath math="A" /> is initialized from a random Gaussian, <InlineMath math="B" /> is initialized to zero — so <InlineMath math="\Delta W = 0" /> at the start of training.</li>
          <li>Only <InlineMath math="A" /> and <InlineMath math="B" /> are trained. <InlineMath math="W_0" /> remains frozen.</li>
          <li>Trainable parameters per layer: <InlineMath math="r(d + k)" /> vs. <InlineMath math="dk" /> for full fine-tuning. For <InlineMath math="d = k = 4096" /> and <InlineMath math="r = 16" />: 131K vs. 16.8M (128x reduction).</li>
        </ul>

        <h3>Why Low-Rank Works</h3>
        <p>
          Aghajanyan et al. (2021) showed that pre-trained models have a low <strong>intrinsic dimensionality</strong> — the weight
          updates during fine-tuning lie in a low-dimensional subspace:
        </p>
        <BlockMath math="\text{rank}(\Delta W) \ll \min(d, k)" />
        <p>
          This means a rank-<InlineMath math="r" /> approximation captures most of the adaptation signal. Empirically, <InlineMath math="r = 8" /> to <InlineMath math="64" /> works well for most tasks.
        </p>

        <h3>QLoRA Quantization</h3>
        <p>
          QLoRA combines three innovations:
        </p>
        <ul>
          <li><strong>4-bit NormalFloat (NF4)</strong>: A data type optimally distributed for normally distributed weights: <InlineMath math="Q_{\text{NF4}}(w) = \arg\min_{q_i \in \text{NF4}} |w - q_i|" /></li>
          <li><strong>Double quantization</strong>: The quantization constants themselves are quantized from FP32 to FP8, saving additional memory.</li>
          <li><strong>Paged optimizers</strong>: Uses CPU RAM for optimizer states that don&apos;t fit in GPU memory, with automatic page transfers.</li>
        </ul>
        <p>Memory comparison for a 7B parameter model:</p>
        <ul>
          <li>Full fine-tuning (FP32): ~112 GB (params + grads + optimizer)</li>
          <li>LoRA (FP16 base + FP32 adapters): ~14 GB + small adapter overhead</li>
          <li>QLoRA (4-bit base + FP16 adapters): ~5 GB + small adapter overhead</li>
        </ul>
      </TopicSection>

      <TopicSection type="code">
        <h3>Full Fine-tuning with HuggingFace</h3>
        <CodeBlock
          language="python"
          title="full_finetune.py"
          code={`from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer
)
from datasets import load_dataset

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

# Load instruction-tuning dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

def format_and_tokenize(example):
    prompt = f"### Instruction:\\n{example['instruction']}\\n\\n### Response:\\n{example['output']}"
    return tokenizer(prompt, truncation=True, max_length=512, padding="max_length")

tokenized = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir="./llama2-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,   # effective batch = 4 * 8 = 32
    learning_rate=2e-5,              # low LR for fine-tuning
    warmup_ratio=0.03,
    weight_decay=0.01,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)
trainer.train()`}
        />

        <h3>LoRA Fine-tuning with PEFT</h3>
        <CodeBlock
          language="python"
          title="lora_finetune.py"
          code={`from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                     # rank of the update matrices
    lora_alpha=32,            # scaling factor (alpha/r applied to updates)
    lora_dropout=0.05,
    target_modules=[          # which layers to apply LoRA to
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj",      # MLP
    ],
    bias="none",
)

# Wrap model with LoRA adapters
model = get_peft_model(model, lora_config)

# See how few parameters we're training
model.print_trainable_parameters()
# trainable params: 13,631,488 || all params: 6,751,350,784 || trainable%: 0.2019

# Training proceeds exactly like full fine-tuning
# but only the LoRA parameters are updated
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./llama2-lora",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # can use larger batch with LoRA
    learning_rate=2e-4,             # higher LR is OK for LoRA
    warmup_ratio=0.03,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
)

# After training, save only the adapter weights (~50 MB vs 14 GB)
model.save_pretrained("./llama2-lora-adapter")

# To load later:
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, "./llama2-lora-adapter")`}
        />

        <h3>QLoRA: Fine-tuning in 4-bit</h3>
        <CodeBlock
          language="python"
          title="qlora_finetune.py"
          code={`from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

model_name = "meta-llama/Llama-2-7b-hf"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # NormalFloat4 — optimal for normal distributions
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,     # quantize the quantization constants too
)

# Load model in 4-bit (~3.5 GB for 7B model)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare for k-bit training (handles gradient checkpointing, layer norms)
model = prepare_model_for_kbit_training(model)

# Apply LoRA on top of the quantized model
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 6,815,744 || all params: 3,507,456,000 || trainable%: 0.1943

# Now train on a single GPU with <8 GB VRAM!
# (use the same Trainer setup as before)`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>LoRA rank selection</strong>: Start with <InlineMath math="r = 16" />. Increase to 32-64 for complex tasks or larger models. Rank 8 works for simple classification tasks. Higher rank = more expressiveness but more parameters.</li>
          <li><strong>Which layers to target</strong>: Applying LoRA to attention projections (Q, K, V, O) is standard. Adding MLP layers (gate, up, down) often helps. Applying to all linear layers gives the best results but increases parameter count.</li>
          <li><strong>Learning rate for LoRA</strong>: LoRA adapters tolerate much higher learning rates than full fine-tuning — 1e-4 to 3e-4 vs. 1e-5 to 3e-5. This is because only the small adapters are being updated.</li>
          <li><strong>Merging adapters</strong>: After training, you can merge LoRA weights into the base model: <InlineMath math="W = W_0 + BA" />. This adds zero inference latency — the merged model runs identically to a fully fine-tuned one.</li>
          <li><strong>Multiple adapters</strong>: You can train separate LoRA adapters for different tasks and swap them at inference time. This enables one base model to serve many specialized tasks.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Using too high a learning rate for full fine-tuning</strong>: Pre-trained weights are sensitive. Start with 1-3e-5 for full fine-tuning. Too high and you&apos;ll catastrophically forget pre-trained knowledge.</li>
          <li><strong>Not freezing the base model in LoRA</strong>: The whole point is that <InlineMath math="W_0" /> stays frozen. If you accidentally set requires_grad=True on the base parameters, you&apos;re doing full fine-tuning at LoRA learning rates, which may diverge.</li>
          <li><strong>Confusing lora_alpha and rank</strong>: The actual scaling is <InlineMath math="\alpha / r" />. Doubling both <InlineMath math="\alpha" /> and <InlineMath math="r" /> keeps the same effective scaling. The convention is to set <InlineMath math="\alpha = 2r" /> and tune <InlineMath math="r" /> alone.</li>
          <li><strong>Forgetting to format data properly</strong>: Instruction fine-tuning requires a specific prompt format (Alpaca, ChatML, etc.). If the format doesn&apos;t match what the model expects, fine-tuning will underperform or produce garbled output.</li>
          <li><strong>Evaluating only on training distribution</strong>: Fine-tuning can overfit quickly on small datasets. Always hold out a validation set and monitor for performance degradation on the base model&apos;s original capabilities.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Explain how LoRA works mathematically. Why is it effective despite training so few parameters? When would you prefer full fine-tuning over LoRA?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>LoRA mechanics</strong>:
            <ul>
              <li>For each target weight matrix <InlineMath math="W_0 \in \mathbb{R}^{d \times k}" />, inject a trainable low-rank update <InlineMath math="\Delta W = BA" /> where <InlineMath math="B \in \mathbb{R}^{d \times r}" /> and <InlineMath math="A \in \mathbb{R}^{r \times k}" />.</li>
              <li><InlineMath math="B" /> is initialized to zero (so training starts from the pre-trained model), <InlineMath math="A" /> from Gaussian.</li>
              <li>Forward pass: <InlineMath math="h = W_0 x + (\alpha/r) \cdot BAx" />. Only <InlineMath math="A" /> and <InlineMath math="B" /> receive gradients.</li>
            </ul>
          </li>
          <li>
            <strong>Why it works</strong>:
            <ul>
              <li>Fine-tuning updates live in a low-dimensional subspace — the <strong>intrinsic dimensionality</strong> of adaptation is much lower than the full parameter count.</li>
              <li>A rank-16 decomposition of a 4096x4096 matrix has 131K parameters but can express any direction in a 16-dimensional subspace of the 4096-dimensional space.</li>
              <li>Empirically, LoRA matches full fine-tuning on most benchmarks within 1-2% accuracy.</li>
            </ul>
          </li>
          <li>
            <strong>When to use full fine-tuning</strong>:
            <ul>
              <li>When the target domain is very different from pre-training data (e.g., adapting an English model to code or a rare language).</li>
              <li>When you have abundant compute and data and need maximum performance.</li>
              <li>When training a smaller model where the parameter savings of LoRA are less impactful.</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Hu et al. (2021) &quot;LoRA: Low-Rank Adaptation of Large Language Models&quot;</strong> — The original LoRA paper.</li>
          <li><strong>Dettmers et al. (2023) &quot;QLoRA: Efficient Finetuning of Quantized Language Models&quot;</strong> — 4-bit quantization + LoRA for accessible fine-tuning.</li>
          <li><strong>Aghajanyan et al. (2021) &quot;Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning&quot;</strong> — The theoretical foundation for why LoRA works.</li>
          <li><strong>HuggingFace PEFT library documentation</strong> — Practical guides for LoRA, QLoRA, and other adapter methods.</li>
          <li><strong>Houlsby et al. (2019) &quot;Parameter-Efficient Transfer Learning for NLP&quot;</strong> — The original adapter layers paper that preceded LoRA.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
