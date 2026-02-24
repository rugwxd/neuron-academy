"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function DistributedTraining() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Modern deep learning models — GPT-4, LLaMA 3, Gemini — are far too large to train on a single GPU.
          GPT-3 has 175 billion parameters, which at FP16 requires 350GB of memory just for the weights (not
          counting optimizer states, gradients, or activations). A single A100 GPU has only 80GB. <strong>Distributed
          training</strong> splits the workload across many GPUs and many machines to make training feasible.
        </p>
        <p>
          There are three fundamental strategies. <strong>Data parallelism</strong> replicates the entire model
          on every GPU and splits the data — each GPU processes a different mini-batch and they synchronize
          gradients. <strong>Model parallelism</strong> splits the model itself across GPUs — different layers
          (or different parts of a layer) live on different devices. <strong>Pipeline parallelism</strong> is a
          form of model parallelism where different stages of the model process different micro-batches
          simultaneously, like an assembly line.
        </p>
        <p>
          In practice, large-scale training uses a combination of all three — this is called <strong>3D parallelism</strong>.
          Frameworks like PyTorch FSDP (Fully Sharded Data Parallel) and DeepSpeed ZeRO make this practical by
          automatically sharding model parameters, gradients, and optimizer states across GPUs while maintaining
          the simplicity of single-GPU code.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Data Parallelism</h3>
        <p>
          With <InlineMath math="N" /> GPUs, each GPU computes gradients on a local batch of size <InlineMath math="B/N" />.
          The gradients are then averaged (all-reduce) to get the global gradient:
        </p>
        <BlockMath math="g_{\text{global}} = \frac{1}{N} \sum_{i=1}^{N} g_i, \quad g_i = \nabla_\theta \mathcal{L}\left(\theta; \mathcal{B}_i\right)" />
        <p>
          This is mathematically equivalent to training with effective batch size <InlineMath math="B" />.
          The learning rate typically needs to scale linearly: <InlineMath math="\eta_N = N \cdot \eta_1" /> (with
          warmup to avoid instability).
        </p>

        <h3>Communication Cost</h3>
        <p>
          An all-reduce of <InlineMath math="M" /> parameters across <InlineMath math="N" /> GPUs using the
          ring all-reduce algorithm takes:
        </p>
        <BlockMath math="T_{\text{all-reduce}} = 2(N-1) \cdot \frac{M}{N} \cdot \left(\alpha + \frac{1}{\beta}\right)" />
        <p>
          where <InlineMath math="\alpha" /> is latency and <InlineMath math="\beta" /> is bandwidth. The key insight:
          the cost scales with <InlineMath math="M" /> (model size), not with <InlineMath math="N" /> (number of GPUs),
          so adding more GPUs does not significantly increase communication time.
        </p>

        <h3>ZeRO Memory Optimization</h3>
        <p>
          For a model with <InlineMath math="M" /> parameters in mixed precision, the memory per GPU is:
        </p>
        <ul>
          <li><strong>No sharding (DDP)</strong>: <InlineMath math="2M + 2M + (4M + 4M + 4M) = 16M" /> bytes (params + grads + optimizer)</li>
          <li><strong>ZeRO Stage 1</strong>: Shard optimizer states → <InlineMath math="4M + 12M/N" /> bytes</li>
          <li><strong>ZeRO Stage 2</strong>: + Shard gradients → <InlineMath math="2M + (2M + 12M)/N" /> bytes</li>
          <li><strong>ZeRO Stage 3 / FSDP</strong>: + Shard parameters → <InlineMath math="16M/N" /> bytes</li>
        </ul>
        <p>
          With 64 GPUs, ZeRO-3 reduces memory per GPU by <InlineMath math="64\times" />, enabling models that
          would otherwise not fit at all.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>PyTorch DistributedDataParallel (DDP)</h3>
        <CodeBlock
          language="python"
          title="ddp_training.py"
          code={`import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup(rank, world_size):
    """Initialize the distributed process group."""
    dist.init_process_group(
        backend="nccl",         # Use NCCL for GPU communication
        init_method="env://",   # Get MASTER_ADDR and MASTER_PORT from env
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)

def train(rank, world_size):
    setup(rank, world_size)

    # Model — wrapped in DDP
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
        num_layers=6,
    ).to(rank)
    model = DDP(model, device_ids=[rank])

    # Data — DistributedSampler ensures each GPU gets different data
    dataset = MyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch
        for batch in loader:
            x, y = batch[0].to(rank), batch[1].to(rank)

            loss = loss_fn(model(x), y)
            loss.backward()        # Gradients auto-synced by DDP (all-reduce)
            optimizer.step()
            optimizer.zero_grad()

        if rank == 0:
            print(f"Epoch {epoch} complete")

    dist.destroy_process_group()

# Launch: torchrun --nproc_per_node=4 ddp_training.py`}
        />

        <h3>PyTorch FSDP (Fully Sharded Data Parallel)</h3>
        <CodeBlock
          language="python"
          title="fsdp_training.py"
          code={`import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

def train_fsdp(rank, world_size):
    setup(rank, world_size)

    # Define wrapping policy — shard at the TransformerEncoderLayer level
    wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={nn.TransformerEncoderLayer},
    )

    # Mixed precision for memory efficiency
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Create model and wrap with FSDP
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=4096, nhead=32, batch_first=True),
        num_layers=32,
    ).to(rank)

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO Stage 3
        mixed_precision=mixed_precision,
        auto_wrap_policy=wrap_policy,
        device_id=rank,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):
        for batch in loader:
            x, y = batch[0].to(rank), batch[1].to(rank)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Save model — FSDP handles gathering shards
    if rank == 0:
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            torch.save(model.state_dict(), "model_fsdp.pt")

    dist.destroy_process_group()

# Launch: torchrun --nproc_per_node=8 fsdp_training.py`}
        />

        <h3>DeepSpeed ZeRO</h3>
        <CodeBlock
          language="python"
          title="deepspeed_training.py"
          code={`import deepspeed
import torch
import torch.nn as nn

# ds_config.json:
# {
#   "train_batch_size": 256,
#   "gradient_accumulation_steps": 4,
#   "fp16": {"enabled": true},
#   "zero_optimization": {
#     "stage": 2,
#     "offload_optimizer": {"device": "cpu"},
#     "contiguous_gradients": true,
#     "overlap_comm": true
#   },
#   "optimizer": {
#     "type": "AdamW",
#     "params": {"lr": 1e-4, "weight_decay": 0.01}
#   }
# }

model = MyLargeModel()

# DeepSpeed wraps model, optimizer, and dataloader
model_engine, optimizer, train_loader, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    training_data=train_dataset,
    config="ds_config.json",
)

for epoch in range(10):
    for batch in train_loader:
        x, y = batch[0].to(model_engine.device), batch[1].to(model_engine.device)

        loss = model_engine(x, y)    # Forward
        model_engine.backward(loss)  # Backward (handles gradient sync)
        model_engine.step()          # Optimizer step (handles ZeRO sharding)

# Launch: deepspeed --num_gpus=8 deepspeed_training.py`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Start with DDP</strong>: If your model fits on one GPU, use DistributedDataParallel for multi-GPU speedups. It&apos;s the simplest and fastest option for models that fit in memory.</li>
          <li><strong>Use FSDP or DeepSpeed when memory is tight</strong>: If your model does not fit on one GPU (or barely fits without room for decent batch sizes), switch to FSDP (PyTorch native) or DeepSpeed ZeRO.</li>
          <li><strong>Scale learning rate with batch size</strong>: When scaling from 1 GPU to N GPUs, multiply the learning rate by N and use a linear warmup over the first 1-5% of training steps to avoid early instability.</li>
          <li><strong>Gradient accumulation for large effective batches</strong>: If you need a 256 batch size but only fit 32 per GPU, accumulate gradients over 8 steps. Combined with 4 GPUs, you get <InlineMath math="32 \times 4 \times 8 = 1024" /> effective batch size.</li>
          <li><strong>Profile communication overhead</strong>: Use <code>torch.profiler</code> or NVIDIA Nsight to ensure computation and communication overlap. If GPUs are idle waiting for all-reduce, you have a bottleneck.</li>
          <li><strong>Use bf16 over fp16</strong>: bfloat16 has the same range as fp32 (8 exponent bits), so it rarely needs loss scaling. fp16 has only 5 exponent bits and requires careful loss scaling to avoid overflow/underflow.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Forgetting DistributedSampler</strong>: Without it, every GPU trains on the same data, wasting compute. The sampler ensures each GPU gets a unique shard of the data.</li>
          <li><strong>Not calling sampler.set_epoch(epoch)</strong>: Without this, the same shard assignment is used every epoch, reducing effective data diversity.</li>
          <li><strong>Ignoring the linear scaling rule</strong>: Doubling the effective batch size without increasing the learning rate effectively halves the update magnitude, slowing convergence.</li>
          <li><strong>Saving the model from all ranks</strong>: Only rank 0 should save checkpoints. All ranks writing simultaneously causes corruption and wastes storage.</li>
          <li><strong>Not overlapping communication and computation</strong>: DDP overlaps all-reduce with backward computation by default. Custom training loops that break this pattern lose significant performance.</li>
          <li><strong>Using gloo backend for GPU training</strong>: Always use NCCL for GPU-to-GPU communication. Gloo is for CPU-only distributed training.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You need to train a 13B parameter LLM on a cluster of 64 A100 GPUs (8 nodes, 8 GPUs each). Describe your distributed training strategy.</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Memory calculation</strong>: 13B params in bf16 = 26GB for weights. With AdamW optimizer states (2 momentum buffers in fp32) = 26 + 52 + 52 = 130GB. Plus activations. Does not fit on one 80GB A100.</li>
          <li><strong>Strategy — FSDP or DeepSpeed ZeRO Stage 3</strong>: Shard parameters, gradients, and optimizer states across all 64 GPUs. Per-GPU memory: 130GB / 64 ~ 2GB for model state, leaving ~78GB for activations and batch data.</li>
          <li><strong>Intra-node vs inter-node</strong>: Within a node (8 GPUs connected by NVLink at 600GB/s), use FSDP with full sharding. Between nodes (connected by InfiniBand at ~200Gb/s), communication is slower, so use gradient accumulation to reduce sync frequency.</li>
          <li><strong>Mixed precision</strong>: Train in bf16 for compute, keep master weights in fp32 for numeric stability. Use activation checkpointing to trade compute for memory on attention layers.</li>
          <li><strong>Batch size</strong>: With 64 GPUs and micro-batch size 4, effective batch size = 256. Use 4 gradient accumulation steps for effective batch of 1024. Learning rate warmup for the first 2000 steps.</li>
          <li><strong>Fault tolerance</strong>: Save checkpoints every 1000 steps. Use elastic training (torchrun with rdzv_backend=c10d) to handle node failures gracefully.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>PyTorch FSDP tutorial</strong> — Official documentation for Fully Sharded Data Parallel training.</li>
          <li><strong>DeepSpeed ZeRO paper (Rajbhandari et al., 2020)</strong> — Memory optimization stages and the math behind sharding.</li>
          <li><strong>Megatron-LM paper (Shoeybi et al., 2019)</strong> — Tensor parallelism for training massive language models.</li>
          <li><strong>&quot;Efficient Large-Scale Language Model Training on GPU Clusters&quot; (Narayanan et al., 2021)</strong> — 3D parallelism combining data, tensor, and pipeline parallelism.</li>
          <li><strong>Goyal et al. &quot;Accurate, Large Minibatch SGD&quot; (2017)</strong> — Linear scaling rule and warmup for distributed training.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
