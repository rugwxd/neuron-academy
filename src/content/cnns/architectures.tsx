"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function Architectures() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          The history of CNN architectures is a story of increasingly clever ideas for going deeper, wider,
          and more efficient. Each landmark architecture introduced a key innovation that changed how we
          think about network design.
        </p>
        <p>
          <strong>LeNet-5 (1998)</strong> showed that a small stack of convolutions could recognize
          handwritten digits. <strong>AlexNet (2012)</strong> scaled this idea to ImageNet with ReLU
          activations, dropout, and GPU training, kicking off the deep learning revolution.
          <strong> VGGNet (2014)</strong> proved that going deeper with uniform 3x3 filters works better
          than large kernels. <strong>GoogLeNet/Inception (2014)</strong> introduced parallel &quot;inception
          modules&quot; that apply multiple filter sizes and concatenate the results.
        </p>
        <p>
          <strong>ResNet (2015)</strong> was the single most important breakthrough: skip connections
          (residual connections) let gradients flow directly through the network, enabling training of
          architectures with 50, 101, or even 152 layers. Without skip connections, networks this deep
          simply could not be trained due to vanishing gradients.
        </p>
        <p>
          After ResNet, the field moved toward <strong>efficiency</strong>. <strong>MobileNet (2017)</strong>
          introduced depthwise separable convolutions for mobile deployment. <strong>EfficientNet (2019)</strong>
          used neural architecture search and compound scaling to find the optimal balance of depth, width,
          and resolution. Today, Vision Transformers (ViT) compete with and often surpass CNNs, but
          ConvNeXt (2022) showed that a &quot;modernized&quot; ResNet with Transformer-inspired design
          choices remains competitive.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>The Residual Connection (ResNet)</h3>
        <p>
          The core idea: instead of learning <InlineMath math="H(x)" /> directly, learn the
          residual <InlineMath math="F(x) = H(x) - x" />, then add back the identity:
        </p>
        <BlockMath math="y = F(x, \{W_i\}) + x" />
        <p>
          During backpropagation, the gradient flows through both the residual path and the skip connection:
        </p>
        <BlockMath math="\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left(\frac{\partial F}{\partial x} + 1\right)" />
        <p>
          The <InlineMath math="+1" /> ensures gradients never completely vanish, no matter how deep the network.
          This is why ResNets can train successfully at 100+ layers while plain networks degrade after ~20.
        </p>

        <h3>Bottleneck Block</h3>
        <p>
          ResNet-50+ uses a bottleneck design to reduce computation: 1x1 conv to reduce channels, 3x3
          conv, then 1x1 conv to expand channels back:
        </p>
        <BlockMath math="x \xrightarrow{1 \times 1,\; 64} \xrightarrow{3 \times 3,\; 64} \xrightarrow{1 \times 1,\; 256} + x" />
        <p>
          The 1x1 convs reduce the number of channels entering the expensive 3x3 conv from 256 to 64,
          cutting FLOPs by approximately <InlineMath math="4\times" />.
        </p>

        <h3>Depthwise Separable Convolution (MobileNet)</h3>
        <p>
          A standard conv with <InlineMath math="C_{\text{in}}" /> input channels, <InlineMath math="C_{\text{out}}" /> output
          channels, and kernel <InlineMath math="k \times k" /> costs:
        </p>
        <BlockMath math="\text{Standard cost} = C_{\text{in}} \cdot C_{\text{out}} \cdot k^2 \cdot H \cdot W" />
        <p>Depthwise separable splits this into depthwise + pointwise:</p>
        <BlockMath math="\text{DW cost} = C_{\text{in}} \cdot k^2 \cdot H \cdot W + C_{\text{in}} \cdot C_{\text{out}} \cdot H \cdot W" />
        <p>The ratio of computational savings:</p>
        <BlockMath math="\frac{\text{DW cost}}{\text{Standard cost}} = \frac{1}{C_{\text{out}}} + \frac{1}{k^2}" />
        <p>For 3x3 kernels and 256 output channels, this is <InlineMath math="\approx 8\text{-}9\times" /> cheaper.</p>

        <h3>EfficientNet Compound Scaling</h3>
        <p>
          EfficientNet scales depth (<InlineMath math="\alpha" />), width (<InlineMath math="\beta" />),
          and resolution (<InlineMath math="\gamma" />) uniformly using a compound coefficient <InlineMath math="\phi" />:
        </p>
        <BlockMath math="\text{depth}: d = \alpha^\phi, \quad \text{width}: w = \beta^\phi, \quad \text{resolution}: r = \gamma^\phi" />
        <BlockMath math="\text{subject to} \quad \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2" />
        <p>
          The constraint ensures that doubling <InlineMath math="\phi" /> roughly doubles the total
          FLOPs. The base values <InlineMath math="\alpha, \beta, \gamma" /> are found by neural architecture search.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>ResNet Bottleneck Block from Scratch</h3>
        <CodeBlock
          language="python"
          title="resnet_block.py"
          code={`import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    """ResNet bottleneck: 1x1 -> 3x3 -> 1x1 with skip connection."""
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        out_channels = mid_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # adjusts skip dims if needed

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))   # 1x1 reduce
        out = self.relu(self.bn2(self.conv2(out)))  # 3x3 spatial
        out = self.bn3(self.conv3(out))             # 1x1 expand

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # THE skip connection
        out = self.relu(out)
        return out

# --- Build a ResNet-50-like stage ---
def make_layer(in_ch, mid_ch, num_blocks, stride=1):
    out_ch = mid_ch * BottleneckBlock.expansion
    downsample = None
    if stride != 1 or in_ch != out_ch:
        downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    layers = [BottleneckBlock(in_ch, mid_ch, stride, downsample)]
    for _ in range(1, num_blocks):
        layers.append(BottleneckBlock(out_ch, mid_ch))
    return nn.Sequential(*layers)

# ResNet-50 has [3, 4, 6, 3] blocks in stages 2-5
layer2 = make_layer(256, 64, num_blocks=3, stride=1)
layer3 = make_layer(256, 128, num_blocks=4, stride=2)

x = torch.randn(2, 256, 56, 56)
h = layer2(x)
print(f"After stage 2: {h.shape}")  # (2, 256, 56, 56)
h = layer3(h)
print(f"After stage 3: {h.shape}")  # (2, 512, 28, 28)`}
        />

        <h3>Using Pretrained Models in PyTorch</h3>
        <CodeBlock
          language="python"
          title="pretrained_models.py"
          code={`import torch
import torchvision.models as models

# --- Load pretrained architectures ---
resnet50 = models.resnet50(weights="DEFAULT")
efficientnet = models.efficientnet_b0(weights="DEFAULT")
mobilenet = models.mobilenet_v3_small(weights="DEFAULT")

# Compare parameter counts
for name, model in [("ResNet-50", resnet50),
                     ("EfficientNet-B0", efficientnet),
                     ("MobileNet-V3-S", mobilenet)]:
    params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {params / 1e6:.1f}M parameters")
# ResNet-50:       25.6M parameters
# EfficientNet-B0:  5.3M parameters
# MobileNet-V3-S:   2.5M parameters

# --- Fine-tune for a custom task (e.g., 10 classes) ---
import torch.nn as nn

model = models.resnet50(weights="DEFAULT")

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final FC layer (unfrozen by default)
model.fc = nn.Linear(model.fc.in_features, 10)

# Only the new FC layer will be trained
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable}")  # 20,490`}
        />

        <h3>Architecture Comparison Table</h3>
        <CodeBlock
          language="python"
          title="architecture_timeline.py"
          code={`# Key CNN architectures and their innovations
architectures = {
    "LeNet-5 (1998)":    {"depth": 5,   "params": "60K",    "innovation": "Conv + pool + FC pipeline"},
    "AlexNet (2012)":    {"depth": 8,   "params": "60M",    "innovation": "ReLU, dropout, GPU training"},
    "VGGNet (2014)":     {"depth": 19,  "params": "144M",   "innovation": "Uniform 3x3 filters"},
    "GoogLeNet (2014)":  {"depth": 22,  "params": "6.8M",   "innovation": "Inception module (multi-scale)"},
    "ResNet (2015)":     {"depth": 152, "params": "60M",    "innovation": "Skip connections"},
    "DenseNet (2017)":   {"depth": 201, "params": "20M",    "innovation": "Dense skip connections"},
    "MobileNet (2017)":  {"depth": 28,  "params": "3.4M",   "innovation": "Depthwise separable conv"},
    "EfficientNet (2019)":{"depth": 82, "params": "5.3M",   "innovation": "Compound scaling + NAS"},
    "ConvNeXt (2022)":   {"depth": 50,  "params": "89M",    "innovation": "Modernized ResNet (Transformer ideas)"},
}

for name, info in architectures.items():
    print(f"{name:25s} | {info['depth']:4d} layers | "
          f"{info['params']:>6s} params | {info['innovation']}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Start with a pretrained model</strong>: Almost never train a CNN from scratch. Use ImageNet-pretrained ResNet-50 or EfficientNet and fine-tune. Even for medical imaging or satellite data, transfer learning provides a massive head start.</li>
          <li><strong>ResNet is still the default backbone</strong>: For detection (Faster R-CNN), segmentation (U-Net, DeepLab), and most computer vision tasks, ResNet-50 remains the standard baseline.</li>
          <li><strong>Use EfficientNet for constrained budgets</strong>: EfficientNet-B0 achieves 77% ImageNet accuracy with 5.3M parameters vs ResNet-50&apos;s 25.6M for 76% accuracy.</li>
          <li><strong>MobileNet/MobileNetV3 for edge deployment</strong>: When inference must run on a phone or embedded device, MobileNet variants with depthwise separable convolutions are the go-to.</li>
          <li><strong>Consider Vision Transformers for large datasets</strong>: ViT outperforms CNNs when pretrained on very large datasets (ImageNet-21K, JFT-300M). For smaller datasets, CNNs still win due to their stronger inductive biases.</li>
          <li><strong>ConvNeXt bridges the gap</strong>: If you want CNN efficiency with Transformer-level accuracy, ConvNeXt adopts larger kernels (7x7), LayerNorm, GELU, and fewer activation functions.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Training from scratch on small datasets</strong>: A CNN with millions of parameters will overfit badly on a small dataset. Use pretrained weights and fine-tune only the last few layers.</li>
          <li><strong>Not using data augmentation</strong>: Random crops, flips, color jitter, and MixUp/CutMix are essential for generalization. PyTorch&apos;s <code>torchvision.transforms</code> makes this easy.</li>
          <li><strong>Choosing VGG for production</strong>: VGG-16 has 138M parameters and is extremely slow. It&apos;s historically important but should never be used in practice when ResNet or EfficientNet exists.</li>
          <li><strong>Forgetting that ResNet&apos;s skip connections require dimension matching</strong>: When the skip connection crosses a stride-2 layer or channel change, you need a 1x1 projection conv on the shortcut path.</li>
          <li><strong>Ignoring FLOPs when comparing models</strong>: Parameter count alone is misleading. A model with fewer parameters can still be slower if it has higher FLOPs (more multiply-add operations).</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Explain why ResNet can train 152-layer networks successfully while a plain (no-skip) 56-layer network performs worse than a 20-layer network. What is the degradation problem, and how do residual connections solve it?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>The degradation problem</strong>: Empirically, adding more layers to a plain deep network
            causes <em>training</em> error (not just test error) to increase. This is not overfitting; it is an
            optimization difficulty. In theory, a deeper network should be at least as good because extra layers
            could learn identity mappings, but in practice optimizers cannot find these solutions.
          </li>
          <li>
            <strong>Why gradients struggle</strong>: In a plain network, the gradient must flow through every
            layer via the chain rule: <InlineMath math="\frac{\partial \mathcal{L}}{\partial x_1} = \prod_{i=1}^{L} \frac{\partial x_{i+1}}{\partial x_i}" />.
            If any factor is consistently less than 1, the product shrinks exponentially (vanishing gradients).
          </li>
          <li>
            <strong>How residual connections fix this</strong>: With <InlineMath math="y = F(x) + x" />,
            the gradient becomes <InlineMath math="\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y}(1 + \frac{\partial F}{\partial x})" />.
            The <InlineMath math="+1" /> term provides a gradient &quot;highway&quot; that bypasses the nonlinear
            layers. Even if <InlineMath math="\frac{\partial F}{\partial x}" /> is small, the gradient
            is at least <InlineMath math="\frac{\partial \mathcal{L}}{\partial y}" />.
          </li>
          <li>
            <strong>Additionally</strong>: Learning a residual <InlineMath math="F(x) \approx 0" /> (identity)
            is much easier than learning <InlineMath math="H(x) \approx x" /> from scratch, so the network can
            easily add layers that &quot;do nothing&quot; when more depth is not needed.
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>He et al. (2016) &quot;Deep Residual Learning for Image Recognition&quot;</strong> -- The ResNet paper. One of the most cited papers in all of computer science.</li>
          <li><strong>Simonyan &amp; Zisserman (2015) &quot;Very Deep Convolutional Networks&quot;</strong> -- The VGGNet paper that established the 3x3 filter principle.</li>
          <li><strong>Tan &amp; Le (2019) &quot;EfficientNet: Rethinking Model Scaling&quot;</strong> -- Compound scaling and neural architecture search for CNNs.</li>
          <li><strong>Liu et al. (2022) &quot;A ConvNet for the 2020s&quot;</strong> -- ConvNeXt: modernizing ResNet with ideas from Transformers.</li>
          <li><strong>Dosovitskiy et al. (2021) &quot;An Image is Worth 16x16 Words&quot;</strong> -- Vision Transformer (ViT), the architecture challenging CNN dominance.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
