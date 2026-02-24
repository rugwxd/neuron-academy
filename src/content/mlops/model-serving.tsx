"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function ModelServing() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Training a machine learning model is only half the battle. <strong>Model serving</strong> is the
          discipline of making a trained model available for real-time or batch predictions in production.
          It bridges the gap between a <code>.pt</code> or <code>.pkl</code> file sitting on your laptop
          and an API endpoint that serves millions of requests per day.
        </p>
        <p>
          There are three dominant paradigms. <strong>FastAPI</strong> (or Flask) wraps your model in a
          Python web server — it&apos;s the simplest approach and works well for moderate traffic.
          <strong> TorchServe</strong> is PyTorch&apos;s native serving solution that handles model
          versioning, batching, and multi-model management out of the box. <strong>NVIDIA Triton
          Inference Server</strong> is the industrial-grade option — it supports multiple frameworks
          (ONNX, TensorRT, PyTorch, TensorFlow), dynamic batching, GPU scheduling, and model ensembles.
        </p>
        <p>
          The choice depends on your scale. A prototype or internal tool? FastAPI is fine. Serving multiple
          PyTorch models with decent traffic? TorchServe. Serving heterogeneous models at scale with GPU
          optimization? Triton. Understanding all three gives you the vocabulary to discuss production ML
          systems in interviews and at work.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Latency and Throughput</h3>
        <p>
          Model serving is governed by two key metrics: <strong>latency</strong> (time per request) and
          <strong>throughput</strong> (requests per second). These are often in tension.
        </p>
        <BlockMath math="\text{Throughput} = \frac{\text{Batch Size}}{\text{Latency per Batch}}" />
        <p>
          <strong>Dynamic batching</strong> groups incoming requests and processes them together on the GPU.
          If a single inference takes <InlineMath math="t_1" /> and a batch of <InlineMath math="B" /> takes
          <InlineMath math="t_B" />, the speedup comes from GPU parallelism:
        </p>
        <BlockMath math="t_B \ll B \cdot t_1 \quad \Rightarrow \quad \text{Throughput increases}" />
        <p>
          However, batching introduces a <strong>wait time</strong> while the server accumulates requests,
          so per-request latency may increase slightly. Triton lets you configure the maximum batch delay
          to control this tradeoff.
        </p>

        <h3>Quantization Speedup</h3>
        <p>
          Reducing precision from FP32 to INT8 roughly halves memory and doubles throughput on supported hardware:
        </p>
        <BlockMath math="\text{Memory}_{INT8} \approx \frac{\text{Memory}_{FP32}}{4}, \quad \text{Throughput}_{INT8} \approx 2\text{-}4 \times \text{Throughput}_{FP32}" />
      </TopicSection>

      <TopicSection type="code">
        <h3>FastAPI Model Server</h3>
        <CodeBlock
          language="python"
          title="serve_fastapi.py"
          code={`import torch
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Load model once at startup
model = torch.jit.load("model_scripted.pt")
model.eval()

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Convert input to tensor
    x = torch.tensor([request.features], dtype=torch.float32)

    # Inference (no gradient computation needed)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)

    pred_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, pred_class].item()

    return PredictionResponse(
        prediction=pred_class,
        confidence=round(confidence, 4),
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Run: uvicorn serve_fastapi:app --host 0.0.0.0 --port 8000`}
        />

        <h3>TorchServe Setup</h3>
        <CodeBlock
          language="bash"
          title="torchserve_setup.sh"
          code={`# 1. Archive the model
torch-model-archiver \\
  --model-name my_classifier \\
  --version 1.0 \\
  --serialized-file model_scripted.pt \\
  --handler image_classifier \\
  --export-path model_store

# 2. Start TorchServe
torchserve --start \\
  --model-store model_store \\
  --models my_classifier=my_classifier.mar \\
  --ncs

# 3. Query the model
curl -X POST http://localhost:8080/predictions/my_classifier \\
  -T input_image.jpg

# 4. Check model status
curl http://localhost:8081/models/my_classifier`}
        />

        <h3>Triton Inference Server</h3>
        <CodeBlock
          language="python"
          title="triton_client.py"
          code={`import tritonclient.http as httpclient
import numpy as np

# Connect to Triton
client = httpclient.InferenceServerClient(url="localhost:8000")

# Prepare input
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
inputs = [
    httpclient.InferInput("input__0", input_data.shape, "FP32")
]
inputs[0].set_data_from_numpy(input_data)

# Request inference
result = client.infer(
    model_name="resnet50",
    inputs=inputs,
)

output = result.as_numpy("output__0")
predicted_class = np.argmax(output, axis=1)
print(f"Predicted class: {predicted_class[0]}")

# --- Triton model repository structure ---
# model_repository/
#   resnet50/
#     config.pbtxt          # Model configuration
#     1/                    # Version 1
#       model.onnx          # The actual model file`}
        />

        <h3>Triton config.pbtxt</h3>
        <CodeBlock
          language="text"
          title="config.pbtxt"
          code={`name: "resnet50"
platform: "onnxruntime_onnx"
max_batch_size: 32

input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]

output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [1000]
  }
]

dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 100
}`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Start simple with FastAPI</strong>: For prototypes and internal tools, FastAPI + Uvicorn + Gunicorn is battle-tested. Add async workers for I/O-bound preprocessing.</li>
          <li><strong>Use TorchScript or ONNX for production</strong>: <code>torch.jit.script</code> or <code>torch.onnx.export</code> removes the Python dependency from inference, enabling C++ runtimes and much lower latency.</li>
          <li><strong>Dynamic batching is free throughput</strong>: Both TorchServe and Triton support it. A batch of 32 on a GPU is barely slower than a batch of 1, so you get ~32x throughput at a small latency cost.</li>
          <li><strong>Always add a health endpoint</strong>: Load balancers (ALB, nginx) need a <code>/health</code> route to know if your server is alive. Also expose <code>/metrics</code> for Prometheus scraping.</li>
          <li><strong>Model versioning is critical</strong>: Both TorchServe and Triton have built-in model versioning. For FastAPI, use a <code>/v1/predict</code> URL pattern or a model registry like MLflow.</li>
          <li><strong>GPU memory planning</strong>: Know your model size. A ResNet-50 is ~100MB, a BERT-base is ~440MB, GPT-2 is ~1.5GB. Plan GPU instances accordingly and consider multi-model serving.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Loading the model on every request</strong>: Model loading should happen once at server startup, not per request. This is the most common performance killer.</li>
          <li><strong>Forgetting torch.no_grad()</strong>: Without it, PyTorch tracks gradients during inference, wasting memory and compute. Always use <code>with torch.no_grad():</code> or <code>model.eval()</code>.</li>
          <li><strong>Not handling concurrent requests</strong>: A single-threaded Flask server will serialize GPU calls. Use async (FastAPI + Uvicorn) or multiple workers.</li>
          <li><strong>Skipping input validation</strong>: Production endpoints receive adversarial and malformed inputs. Validate shapes, dtypes, and value ranges before they hit the model.</li>
          <li><strong>Ignoring cold start</strong>: The first request after deployment (model loading, GPU warmup, JIT compilation) can be 10-100x slower. Send warm-up requests in your startup script.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You&apos;re deploying a BERT-based sentiment classifier that needs to serve 500 requests/second with p99 latency under 50ms. Walk through your architecture.</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Model optimization</strong>: Export BERT to ONNX, then convert to TensorRT for INT8 inference. This typically gives 4-8x speedup over vanilla PyTorch.</li>
          <li><strong>Serving</strong>: Use Triton Inference Server with dynamic batching (preferred batch sizes 8/16/32, max queue delay 5ms). This maximizes GPU utilization.</li>
          <li><strong>Infrastructure</strong>: Deploy on 2-3 GPU instances (e.g., T4 or A10G) behind an Application Load Balancer. Each T4 can handle ~200 req/s for BERT with TensorRT.</li>
          <li><strong>Input pipeline</strong>: Tokenization on CPU (pre-tokenize common inputs and cache), tensor creation, then GPU inference. Keep the tokenizer in the Triton ensemble pipeline or a sidecar.</li>
          <li><strong>Monitoring</strong>: Track p50/p95/p99 latency, throughput, GPU utilization, and queue depth. Alert if p99 exceeds 40ms (leave headroom). Use Prometheus + Grafana.</li>
          <li><strong>Autoscaling</strong>: Scale on GPU utilization (&gt;70%) or request queue depth. Use Kubernetes HPA with custom metrics.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>FastAPI documentation</strong> — Official docs with async patterns and dependency injection.</li>
          <li><strong>TorchServe documentation</strong> — Custom handlers, batching, and metrics configuration.</li>
          <li><strong>NVIDIA Triton Inference Server docs</strong> — Model repository, dynamic batching, model ensembles, and performance analyzer.</li>
          <li><strong>MLflow Model Serving</strong> — Model registry and deployment patterns for experiment tracking to production.</li>
          <li><strong>&quot;Designing Machine Learning Systems&quot; by Chip Huyen, Ch. 7</strong> — Deployment and prediction service patterns.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
