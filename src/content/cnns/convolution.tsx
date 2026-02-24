"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function Convolution() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          A convolution is the fundamental operation that gives CNNs their name. Think of it as a small
          &quot;window&quot; (called a <strong>filter</strong> or <strong>kernel</strong>) that slides across an
          image, computing a weighted sum at every position. The result is a <strong>feature map</strong> that
          highlights where certain patterns appear in the input.
        </p>
        <p>
          Why is this brilliant? Two reasons. First, <strong>parameter sharing</strong>: the same filter is
          used at every spatial position, so the network learns a pattern once and can detect it anywhere in
          the image. Second, <strong>local connectivity</strong>: each output pixel depends only on a small
          neighborhood of inputs, not the entire image. Together, these properties make CNNs vastly more
          efficient than fully-connected networks for image data.
        </p>
        <p>
          A typical CNN layer stacks three operations: (1) the convolution itself, which produces a feature
          map, (2) a nonlinear activation like ReLU, and (3) optional <strong>pooling</strong> (e.g., max
          pooling) to downsample the spatial dimensions. As you go deeper in the network, filters learn
          increasingly abstract features: edges in early layers, textures in middle layers, and object parts
          or entire objects in later layers.
        </p>
        <p>
          Key terminology: <strong>stride</strong> controls how far the filter moves at each step.
          <strong> Padding</strong> adds zeros around the border so the output size can match the input.
          <strong> Dilation</strong> inserts gaps in the filter to increase the receptive field without
          adding parameters.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>2D Discrete Convolution (Cross-Correlation)</h3>
        <p>
          Technically, what deep learning calls &quot;convolution&quot; is actually <strong>cross-correlation</strong>.
          Given input <InlineMath math="I" /> and kernel <InlineMath math="K" /> of size <InlineMath math="k_h \times k_w" />:
        </p>
        <BlockMath math="(I * K)[i, j] = \sum_{m=0}^{k_h - 1} \sum_{n=0}^{k_w - 1} I[i + m,\; j + n] \cdot K[m, n]" />
        <p>
          True mathematical convolution flips the kernel, but in practice we skip the flip because the
          kernel weights are learned anyway.
        </p>

        <h3>Output Size Formula</h3>
        <p>
          Given input size <InlineMath math="W" />, kernel size <InlineMath math="k" />,
          padding <InlineMath math="p" />, stride <InlineMath math="s" />, and
          dilation <InlineMath math="d" />:
        </p>
        <BlockMath math="W_{\text{out}} = \left\lfloor \frac{W + 2p - d(k - 1) - 1}{s} + 1 \right\rfloor" />

        <h3>Parameter Count</h3>
        <p>
          For a conv layer with <InlineMath math="C_{\text{in}}" /> input channels,
          <InlineMath math="C_{\text{out}}" /> output channels (filters), and kernel
          size <InlineMath math="k \times k" />:
        </p>
        <BlockMath math="\text{Parameters} = C_{\text{out}} \times (C_{\text{in}} \times k^2 + 1)" />
        <p>
          The <InlineMath math="+1" /> is the bias term per filter. Compare this with a fully-connected
          layer on a 224x224x3 image: <InlineMath math="224 \times 224 \times 3 \times N" /> parameters
          for <InlineMath math="N" /> outputs, which would be millions. A 3x3 conv with 64 filters
          on 3-channel input has only <InlineMath math="64 \times (3 \times 9 + 1) = 1{,}792" /> parameters.
        </p>

        <h3>Receptive Field</h3>
        <p>
          The receptive field of a neuron is the region of the input that influences its value. For
          a stack of <InlineMath math="L" /> conv layers each with kernel size <InlineMath math="k" /> and stride 1:
        </p>
        <BlockMath math="r = L(k - 1) + 1" />
        <p>
          Two 3x3 layers have the same receptive field as one 5x5 layer (<InlineMath math="r = 5" />) but with
          fewer parameters (<InlineMath math="2 \times 9 = 18" /> vs <InlineMath math="25" />) and an
          extra nonlinearity between them. This is why modern architectures favor small kernels.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Convolution from Scratch</h3>
        <CodeBlock
          language="python"
          title="conv2d_scratch.py"
          code={`import torch
import torch.nn.functional as F

def conv2d_scratch(image, kernel, stride=1, padding=0):
    """
    2D convolution from scratch (single image, single filter).

    Args:
        image: (C_in, H, W) tensor
        kernel: (C_in, k_h, k_w) tensor
        stride: step size
        padding: zero-padding around borders
    Returns:
        feature_map: (H_out, W_out) tensor
    """
    if padding > 0:
        image = F.pad(image, [padding] * 4)  # pad all sides

    C_in, H, W = image.shape
    k_h, k_w = kernel.shape[1], kernel.shape[2]
    H_out = (H - k_h) // stride + 1
    W_out = (W - k_w) // stride + 1

    output = torch.zeros(H_out, W_out)
    for i in range(H_out):
        for j in range(W_out):
            region = image[:, i*stride:i*stride+k_h, j*stride:j*stride+k_w]
            output[i, j] = (region * kernel).sum()

    return output

# Example: 3-channel 8x8 image, 3x3 kernel
image = torch.randn(3, 8, 8)
kernel = torch.randn(3, 3, 3)
out = conv2d_scratch(image, kernel, stride=1, padding=1)
print(f"Output shape: {out.shape}")  # (8, 8) -- same spatial size with padding=1`}
        />

        <h3>PyTorch Conv2d in Practice</h3>
        <CodeBlock
          language="python"
          title="conv2d_pytorch.py"
          code={`import torch
import torch.nn as nn

# --- Basic Conv2d usage ---
conv = nn.Conv2d(
    in_channels=3,     # RGB input
    out_channels=64,   # 64 filters
    kernel_size=3,     # 3x3 kernels
    stride=1,
    padding=1,         # "same" padding for stride=1
    bias=True,
)
x = torch.randn(8, 3, 224, 224)  # batch=8, 3 channels, 224x224
out = conv(x)
print(f"Output: {out.shape}")  # (8, 64, 224, 224)
print(f"Parameters: {sum(p.numel() for p in conv.parameters())}")  # 1,792

# --- A typical conv block ---
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              stride=stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Stack of conv blocks
block1 = ConvBlock(3, 64)
block2 = ConvBlock(64, 128, stride=2)  # downsamples by 2x

x = torch.randn(4, 3, 32, 32)
h = block1(x)
print(f"After block1: {h.shape}")  # (4, 64, 32, 32)
h = block2(h)
print(f"After block2: {h.shape}")  # (4, 128, 16, 16)

# --- Pooling operations ---
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
avgpool = nn.AdaptiveAvgPool2d(1)  # global average pool to 1x1

h = maxpool(torch.randn(1, 64, 16, 16))
print(f"After MaxPool2d: {h.shape}")      # (1, 64, 8, 8)
h = avgpool(torch.randn(1, 64, 8, 8))
print(f"After AdaptiveAvgPool: {h.shape}") # (1, 64, 1, 1)`}
        />

        <h3>Visualizing Filters and Feature Maps</h3>
        <CodeBlock
          language="python"
          title="visualize_features.py"
          code={`import torch
import torchvision.models as models
import matplotlib.pyplot as plt

# Load pretrained ResNet and extract first conv layer
model = models.resnet18(weights="DEFAULT")
first_conv = model.conv1.weight.data.clone()  # (64, 3, 7, 7)

# Visualize first 16 filters
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    # Normalize filter to [0, 1] for display
    filt = first_conv[i]
    filt = (filt - filt.min()) / (filt.max() - filt.min())
    ax.imshow(filt.permute(1, 2, 0).numpy())  # (H, W, 3)
    ax.set_title(f"Filter {i}")
    ax.axis("off")
plt.suptitle("First-layer filters of ResNet-18")
plt.tight_layout()
plt.show()`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Always use 3x3 kernels</strong>: Stacking two 3x3 convs gives a 5x5 receptive field with fewer parameters and more nonlinearity. This insight from VGGNet is universally adopted.</li>
          <li><strong>Batch normalization after conv</strong>: The standard pattern is Conv &rarr; BN &rarr; ReLU. BN stabilizes training and allows higher learning rates.</li>
          <li><strong>Use stride-2 convolution instead of pooling</strong>: Modern architectures (ResNet, etc.) often downsample with stride-2 convs rather than max pooling, as the model can learn what to keep.</li>
          <li><strong>1x1 convolutions</strong>: These are channel-wise linear projections. Used in Inception (bottleneck), ResNet (matching dimensions), and MobileNet (pointwise conv). They change the number of channels without affecting spatial dimensions.</li>
          <li><strong>Depthwise separable convolutions</strong>: Split a standard conv into depthwise (one filter per channel) + pointwise (1x1). Reduces parameters by a factor of <InlineMath math="\sim k^2" />. Used in MobileNet, EfficientNet, and Xception.</li>
          <li><strong>Global average pooling</strong>: Replace the final FC layer with AdaptiveAvgPool2d(1) + a small FC. This reduces parameters dramatically and acts as a structural regularizer.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Forgetting to account for padding</strong>: Without padding, each conv layer shrinks the spatial dimensions. For a 3x3 kernel with stride 1, use padding=1 to preserve size.</li>
          <li><strong>Mixing up &quot;valid&quot; and &quot;same&quot; padding</strong>: PyTorch uses explicit pixel counts (padding=1), not string labels. For &quot;same&quot; output with stride=1 and kernel k, use padding=k//2.</li>
          <li><strong>Ignoring the channel dimension</strong>: A 3x3 conv on a 64-channel input applies a 3x3x64 kernel, not just 3x3. Each filter spans <em>all</em> input channels.</li>
          <li><strong>Using bias with BatchNorm</strong>: BatchNorm already has a learnable bias (<InlineMath math="\beta" />). Adding bias in the conv layer is redundant. Use <code>bias=False</code> when followed by BN.</li>
          <li><strong>Not understanding receptive field growth</strong>: A 5-layer CNN with 3x3 kernels has a receptive field of only 11x11 pixels. If your task needs global context, consider dilated convolutions or attention mechanisms.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You have a 32x32 RGB image. You apply a Conv2d layer with 16 filters of size 5x5, stride=1, padding=0. What is the output shape? How many trainable parameters does this layer have?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Output shape</strong>:
            <ul>
              <li>Using the formula: <InlineMath math="W_{\text{out}} = \lfloor(32 + 2 \times 0 - 5) / 1 + 1\rfloor = 28" /></li>
              <li>We have 16 filters, so output is <InlineMath math="(N, 16, 28, 28)" /> where <InlineMath math="N" /> is the batch size.</li>
            </ul>
          </li>
          <li>
            <strong>Parameters</strong>:
            <ul>
              <li>Each filter: <InlineMath math="3 \times 5 \times 5 = 75" /> weights (3 input channels, 5x5 kernel)</li>
              <li>Plus 1 bias per filter</li>
              <li>Total: <InlineMath math="16 \times (75 + 1) = 1{,}216" /> parameters</li>
            </ul>
          </li>
          <li>
            <strong>Key insight</strong>: Compare this with a fully-connected layer from the flattened input (<InlineMath math="32 \times 32 \times 3 = 3{,}072" />) to 16 outputs: that would require <InlineMath math="3{,}072 \times 16 + 16 = 49{,}168" /> parameters. The conv layer achieves far more useful computation with 40x fewer parameters thanks to weight sharing and local connectivity.
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>CS231n Convolutional Networks module</strong> -- Stanford&apos;s classic deep learning course with the best intuitive explanations of convolution.</li>
          <li><strong>Dumoulin &amp; Visin (2018) &quot;A Guide to Convolution Arithmetic for Deep Learning&quot;</strong> -- Comprehensive visual guide to all conv/transposed-conv output size calculations.</li>
          <li><strong>Howard et al. (2017) &quot;MobileNets&quot;</strong> -- Introduces depthwise separable convolutions for efficient inference on mobile devices.</li>
          <li><strong>Yu &amp; Koltun (2016) &quot;Multi-Scale Context Aggregation by Dilated Convolutions&quot;</strong> -- Dilated (atrous) convolutions for expanding receptive fields without pooling.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
