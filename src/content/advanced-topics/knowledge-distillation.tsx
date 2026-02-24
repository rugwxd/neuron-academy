"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function KnowledgeDistillation() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          You have a massive 175-billion-parameter model that achieves state-of-the-art accuracy, but it
          takes 500ms per inference and costs $0.01 per query. You need something that runs in 5ms on a
          phone. <strong>Knowledge distillation</strong> transfers the &quot;knowledge&quot; from a large
          <strong> teacher</strong> model into a small <strong>student</strong> model, producing a compact
          model that performs far better than training the small model from scratch.
        </p>
        <p>
          The key insight is about <strong>soft targets</strong>. When a teacher classifies an image of a
          BMW, it does not just output &quot;car&quot; — it also assigns small probabilities to
          &quot;truck&quot; (0.05), &quot;motorcycle&quot; (0.02), and &quot;bicycle&quot; (0.01). These soft
          probabilities contain rich information about which classes are similar. A student trained on these
          soft targets learns inter-class relationships that hard labels (just &quot;car&quot;) cannot convey.
        </p>
        <p>
          Beyond classification, distillation applies broadly: compressing BERT into DistilBERT (40% smaller,
          97% of the performance), distilling GPT-4 into smaller models, and even <strong>self-distillation</strong> where
          a model distills knowledge from its deeper layers into shallower ones. It is one of the most
          practical techniques for deploying ML models in resource-constrained environments.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Temperature-Scaled Softmax</h3>
        <p>
          The standard softmax produces very peaked distributions (one class near 1.0, everything else near 0).
          To reveal the teacher&apos;s &quot;dark knowledge&quot; about inter-class relationships, we soften
          the distribution with a temperature parameter <InlineMath math="T" />:
        </p>
        <BlockMath math="q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}" />
        <p>
          where <InlineMath math="z_i" /> are the logits (pre-softmax outputs). At <InlineMath math="T = 1" />,
          this is the standard softmax. As <InlineMath math="T \to \infty" />, the distribution approaches
          uniform. Typical distillation uses <InlineMath math="T \in [3, 20]" /> to produce informative soft
          targets.
        </p>

        <h3>Distillation Loss (Hinton et al., 2015)</h3>
        <p>
          The student is trained with a weighted combination of two losses:
        </p>
        <BlockMath math="\mathcal{L} = \alpha \cdot T^2 \cdot \text{KL}\!\left(q^T_{\text{teacher}} \;\|\; q^T_{\text{student}}\right) + (1 - \alpha) \cdot \text{CE}(y, q^1_{\text{student}})" />
        <p>
          where:
        </p>
        <ul>
          <li><InlineMath math="q^T_{\text{teacher}}" /> and <InlineMath math="q^T_{\text{student}}" /> are softmax outputs at temperature <InlineMath math="T" /></li>
          <li><InlineMath math="q^1_{\text{student}}" /> is the student&apos;s output at <InlineMath math="T=1" /> (standard softmax)</li>
          <li><InlineMath math="y" /> is the hard label and <InlineMath math="\text{CE}" /> is cross-entropy</li>
          <li><InlineMath math="\alpha" /> balances the two losses (typically 0.5&ndash;0.9)</li>
          <li>The <InlineMath math="T^2" /> factor compensates for the gradient magnitude shrinking when temperature is high</li>
        </ul>

        <h3>Why KL Divergence?</h3>
        <p>
          The KL divergence between teacher and student soft targets is:
        </p>
        <BlockMath math="\text{KL}(q_T \| q_S) = \sum_i q_{T,i} \log \frac{q_{T,i}}{q_{S,i}}" />
        <p>
          This penalizes the student more heavily when the teacher assigns high probability to a class but
          the student does not. It effectively makes the student mimic the teacher&apos;s full output
          distribution, not just the top-1 prediction.
        </p>

        <h3>Feature-Based Distillation (FitNets)</h3>
        <p>
          Instead of (or in addition to) matching output distributions, feature-based distillation matches
          intermediate representations. Given teacher feature maps <InlineMath math="F_T" /> and student
          feature maps <InlineMath math="F_S" />:
        </p>
        <BlockMath math="\mathcal{L}_{\text{feature}} = \| W_r \cdot F_S - F_T \|_2^2" />
        <p>
          where <InlineMath math="W_r" /> is a learned projection matrix that maps the student&apos;s
          (typically smaller) feature space to the teacher&apos;s dimension. This transfers not just what
          the teacher predicts, but <em>how</em> it represents the input internally.
        </p>

        <h3>Self-Distillation</h3>
        <p>
          In self-distillation, the teacher and student are the <strong>same architecture</strong>. A model
          trained normally becomes the teacher, and a fresh copy is trained as the student using the
          teacher&apos;s soft targets. Surprisingly, the student often <em>outperforms</em> the teacher.
          This works because the soft targets provide a regularization effect — the student sees a smoother
          label distribution that reduces overfitting:
        </p>
        <BlockMath math="\mathcal{L}_{\text{self}} = \text{KL}\!\left(f_{\theta_{\text{teacher}}}(x) \;\|\; f_{\theta_{\text{student}}}(x)\right)" />
      </TopicSection>

      <TopicSection type="code">
        <h3>Knowledge Distillation in PyTorch from Scratch</h3>
        <CodeBlock
          language="python"
          title="distillation.py"
          code={`import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Define Teacher and Student Networks ---
class TeacherNet(nn.Module):
    """Large model: ~1.2M parameters."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class StudentNet(nn.Module):
    """Small model: ~30K parameters (25x smaller)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# --- Distillation Loss ---
def distillation_loss(student_logits, teacher_logits, labels,
                      temperature=4.0, alpha=0.7):
    """
    Combined distillation + hard-label loss.

    Args:
        student_logits: raw logits from student (B, C)
        teacher_logits: raw logits from teacher (B, C)
        labels: ground truth labels (B,)
        temperature: softmax temperature (higher = softer)
        alpha: weight for distillation loss (1-alpha for hard loss)
    """
    # Soft targets from teacher (high temperature)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)

    # KL divergence on soft targets (multiply by T^2 per Hinton)
    kd_loss = F.kl_div(
        soft_student, soft_teacher, reduction="batchmean"
    ) * (temperature ** 2)

    # Standard cross-entropy on hard labels (temperature=1)
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * kd_loss + (1 - alpha) * hard_loss

# --- Data ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_data = datasets.MNIST("./data", train=True, download=True,
                             transform=transform)
test_data = datasets.MNIST("./data", train=False, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`}
        />

        <h3>Training the Teacher, Then Distilling to Student</h3>
        <CodeBlock
          language="python"
          title="train_distill.py"
          code={`def train_model(model, train_loader, epochs=10, lr=1e-3):
    """Standard training with hard labels."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {total_loss/total:.4f} | Acc: {acc:.4f}")
    return model

def evaluate(model, test_loader):
    """Evaluate accuracy on test set."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return correct / total

# --- Step 1: Train the teacher ---
print("=" * 50)
print("Training Teacher Network")
print("=" * 50)
teacher = TeacherNet().to(device)
teacher = train_model(teacher, train_loader, epochs=10)
teacher_acc = evaluate(teacher, test_loader)
print(f"Teacher Test Accuracy: {teacher_acc:.4f}")

# --- Step 2: Train student WITHOUT distillation (baseline) ---
print("\\n" + "=" * 50)
print("Training Student (No Distillation)")
print("=" * 50)
student_baseline = StudentNet().to(device)
student_baseline = train_model(student_baseline, train_loader, epochs=10)
baseline_acc = evaluate(student_baseline, test_loader)
print(f"Student Baseline Accuracy: {baseline_acc:.4f}")

# --- Step 3: Train student WITH distillation ---
print("\\n" + "=" * 50)
print("Training Student (With Distillation)")
print("=" * 50)
student_distilled = StudentNet().to(device)
optimizer = torch.optim.Adam(student_distilled.parameters(), lr=1e-3)
teacher.eval()  # Teacher is frozen

for epoch in range(10):
    student_distilled.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Get teacher's logits (no gradient needed)
        with torch.no_grad():
            teacher_logits = teacher(images)

        # Get student's logits
        student_logits = student_distilled(images)

        # Combined distillation loss
        loss = distillation_loss(
            student_logits, teacher_logits, labels,
            temperature=4.0, alpha=0.7
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (student_logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}/10 | "
          f"Loss: {total_loss/total:.4f} | Acc: {acc:.4f}")

distilled_acc = evaluate(student_distilled, test_loader)
print(f"\\nStudent Distilled Accuracy: {distilled_acc:.4f}")

# --- Comparison ---
teacher_params = sum(p.numel() for p in teacher.parameters())
student_params = sum(p.numel() for p in student_distilled.parameters())

print("\\n" + "=" * 50)
print("COMPARISON")
print("=" * 50)
print(f"Teacher:            {teacher_acc:.4f} acc | "
      f"{teacher_params:,} params")
print(f"Student (baseline): {baseline_acc:.4f} acc | "
      f"{student_params:,} params")
print(f"Student (distill):  {distilled_acc:.4f} acc | "
      f"{student_params:,} params")
print(f"Compression ratio:  {teacher_params/student_params:.1f}x")
print(f"Distillation gain:  "
      f"{(distilled_acc - baseline_acc)*100:+.2f}%")`}
        />

        <h3>Feature-Based Distillation</h3>
        <CodeBlock
          language="python"
          title="feature_distillation.py"
          code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDistillationLoss(nn.Module):
    """
    Combine output distillation with intermediate feature matching.
    The student learns to mimic both the teacher's predictions
    and its internal representations.
    """
    def __init__(self, teacher_dim, student_dim, temperature=4.0,
                 alpha=0.5, beta=0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha   # weight for KD loss
        self.beta = beta     # weight for feature loss
        # Projection to match dimensions
        self.align = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_logits, teacher_logits, labels,
                student_features, teacher_features):
        # 1. Standard distillation loss (output matching)
        T = self.temperature
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T ** 2)

        # 2. Feature matching loss (intermediate representations)
        # Project student features to teacher's dimension
        student_proj = self.align(student_features)
        # L2 normalize both for stable matching
        s_norm = F.normalize(student_proj, dim=-1)
        t_norm = F.normalize(teacher_features.detach(), dim=-1)
        feature_loss = F.mse_loss(s_norm, t_norm)

        # 3. Hard label loss
        hard_loss = F.cross_entropy(student_logits, labels)

        total = (self.alpha * kd_loss +
                 self.beta * feature_loss +
                 (1 - self.alpha - self.beta) * hard_loss)
        return total, {
            "kd_loss": kd_loss.item(),
            "feature_loss": feature_loss.item(),
            "hard_loss": hard_loss.item(),
        }

# --- Usage with hooks to extract intermediate features ---
def get_features(model, layer_name):
    """Register a forward hook to capture intermediate features."""
    features = {}
    def hook(module, input, output):
        features["out"] = output.detach()
    handle = dict(model.named_modules())[layer_name].register_forward_hook(hook)
    return features, handle

# Attach hooks to extract features from specific layers
teacher_feats, t_hook = get_features(teacher, "features.7")  # after 2nd block
student_feats, s_hook = get_features(student_distilled, "features.3")  # after 2nd block

criterion = FeatureDistillationLoss(
    teacher_dim=128, student_dim=32,
    temperature=4.0, alpha=0.5, beta=0.3
).to(device)

# Training loop would use criterion.forward() with both logits and features
# Don't forget to remove hooks after training:
# t_hook.remove(); s_hook.remove()`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Temperature is the most important hyperparameter</strong>: Too low (1-2) and the soft targets are nearly as hard as one-hot labels — you lose the dark knowledge. Too high (20+) and the distribution is nearly uniform — there is no signal. Start with <InlineMath math="T=4" /> and sweep [2, 4, 8, 16].</li>
          <li><strong>DistilBERT is the canonical success story</strong>: DistilBERT achieves 97% of BERT&apos;s performance with 40% fewer parameters and 60% faster inference. It was trained with a combination of output distillation, intermediate-layer distillation, and a cosine embedding loss.</li>
          <li><strong>The teacher does not need to be perfect</strong>: A teacher with 95% accuracy still provides valuable soft targets because the inter-class similarity information is present even in the teacher&apos;s errors. Even noisy teachers improve over training from scratch.</li>
          <li><strong>Distillation works best when the capacity gap is moderate</strong>: A student that is 100x smaller than the teacher may not have enough capacity to absorb the knowledge. If the gap is too large, use a chain: distill a large teacher into a medium model, then distill that into a small model (progressive distillation).</li>
          <li><strong>Combine with other compression techniques</strong>: Distillation pairs well with pruning and quantization. Distill first to get a good small model, then quantize to INT8 for further speedup. This is the standard production compression pipeline.</li>
          <li><strong>Data augmentation amplifies distillation</strong>: The student benefits from seeing more data than the teacher was trained on. Use aggressive augmentation during distillation — the teacher&apos;s soft targets on augmented data provide regularization that prevents the student from overfitting.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Forgetting the T-squared factor</strong>: The gradients of KL divergence on softened probabilities scale as <InlineMath math="1/T^2" />. Without multiplying the KD loss by <InlineMath math="T^2" />, the distillation signal vanishes at high temperatures and the student effectively trains only on hard labels.</li>
          <li><strong>Using teacher logits instead of probabilities</strong>: The KL divergence must be computed between probability distributions (after softmax), not raw logits. Pass logits through <code>softmax(logits / T)</code> before computing KL divergence.</li>
          <li><strong>Training the teacher simultaneously</strong>: The teacher should be frozen during distillation. If you update the teacher, the soft target distribution keeps changing and the student cannot converge. Train the teacher to completion first, then distill.</li>
          <li><strong>Setting alpha too low</strong>: If <InlineMath math="\alpha" /> is too small (e.g., 0.1), the student mostly trains on hard labels and gets little benefit from distillation. For classification, <InlineMath math="\alpha \in [0.5, 0.9]" /> typically works best. The soft targets are the primary signal.</li>
          <li><strong>Ignoring the student architecture</strong>: Distillation cannot make any architecture good — the student must be well-designed for its size. A poorly designed 1M-parameter student will still underperform a well-designed 1M-parameter student, even with a perfect teacher.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You have a BERT-large model in production with 340M parameters and 50ms latency. The team needs to cut latency to 10ms for mobile deployment. Walk through your approach.</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Step 1: Establish baseline metrics</strong>
            <ul>
              <li>Measure the teacher&apos;s accuracy, latency, and model size on the target hardware.</li>
              <li>Define the acceptable accuracy drop (e.g., no more than 2% regression).</li>
            </ul>
          </li>
          <li><strong>Step 2: Choose student architecture</strong>
            <ul>
              <li>Start with DistilBERT (6 layers, 66M params) as a proven compressed BERT variant.</li>
              <li>If that is still too large, consider TinyBERT (4 layers, 14.5M params) or a custom architecture.</li>
              <li>Profile latency on the target device — parameters alone do not determine speed (attention is quadratic in sequence length).</li>
            </ul>
          </li>
          <li><strong>Step 3: Distillation pipeline</strong>
            <ul>
              <li>Task-agnostic distillation: distill on a large general corpus (like the original pre-training data) first.</li>
              <li>Task-specific distillation: fine-tune the teacher on your task, then distill using task data with <InlineMath math="T=4" /> and <InlineMath math="\alpha=0.7" />.</li>
              <li>Use both output-layer and intermediate-layer (feature) distillation for best results.</li>
              <li>Augment the task data using the teacher to label unlabeled examples (data augmentation via pseudo-labeling).</li>
            </ul>
          </li>
          <li><strong>Step 4: Further compression</strong>
            <ul>
              <li>Quantize the distilled student to INT8 using dynamic quantization (PyTorch <code>torch.quantization</code>), typically giving 2-4x additional speedup.</li>
              <li>If still above latency budget, apply structured pruning to remove attention heads or FFN dimensions that contribute least.</li>
            </ul>
          </li>
          <li><strong>Step 5: Validate</strong>: Compare the compressed model against the teacher on the full test suite. Check for accuracy degradation on edge cases and minority classes, not just overall accuracy. Deploy with A/B testing.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Hinton, Vinyals, &amp; Dean (2015) &quot;Distilling the Knowledge in a Neural Network&quot;</strong> — The foundational paper that introduced soft targets, temperature scaling, and the distillation loss framework.</li>
          <li><strong>Sanh et al. (2019) &quot;DistilBERT, a distilled version of BERT&quot;</strong> — The most widely-used distilled language model. Demonstrates triple loss (distillation + masked LM + cosine embedding) for NLP.</li>
          <li><strong>Romero et al. (2015) &quot;FitNets: Hints for Thin Deep Nets&quot;</strong> — Introduces intermediate feature matching, showing that aligning hidden representations improves distillation beyond output-only matching.</li>
          <li><strong>Jiao et al. (2020) &quot;TinyBERT: Distilling BERT for Natural Language Understanding&quot;</strong> — Proposes attention-based distillation and a two-stage (general + task-specific) distillation framework.</li>
          <li><strong>Gou et al. (2021) &quot;Knowledge Distillation: A Survey&quot;</strong> — Comprehensive survey covering response-based, feature-based, and relation-based distillation methods across vision and NLP.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
