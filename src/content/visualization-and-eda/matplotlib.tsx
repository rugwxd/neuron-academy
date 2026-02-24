"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function MatplotlibDeepDive() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Matplotlib is the <strong>foundational plotting library</strong> in Python. Almost every other
          visualization library (Seaborn, Plotly, pandas plots) is built on top of it or inspired by it.
          Understanding Matplotlib deeply means you can customize any visualization to publication quality.
        </p>
        <p>
          The key mental model is the <strong>Figure/Axes hierarchy</strong>. A <em>Figure</em> is the
          entire canvas — think of it as the piece of paper. An <em>Axes</em> is a single plot area
          within that figure — each Axes has its own x-axis, y-axis, title, and data. A figure can
          contain one or many Axes objects arranged in a grid or arbitrary layout.
        </p>
        <p>
          Most beginners use the <code>plt.plot()</code> shortcut (the &quot;pyplot&quot; interface), which
          implicitly creates a figure and axes behind the scenes. This works for quick plots, but for
          anything serious — multi-panel figures, shared axes, inset plots — you need the
          <strong> object-oriented interface</strong>: create a figure, add axes, and call methods
          directly on those axes objects.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>The Rendering Pipeline</h3>
        <p>
          Matplotlib&apos;s rendering follows a transformation chain. Data coordinates are mapped to display
          coordinates through a series of affine transforms:
        </p>
        <BlockMath math="\mathbf{p}_{\text{display}} = T_{\text{axes}} \circ T_{\text{data}}(\mathbf{p}_{\text{data}})" />
        <p>
          Where <InlineMath math="T_{\text{data}}" /> maps data units to normalized axes coordinates
          <InlineMath math="[0, 1] \times [0, 1]" />, and <InlineMath math="T_{\text{axes}}" /> maps
          those to pixel positions on the figure canvas.
        </p>

        <h3>Figure Size and DPI</h3>
        <p>
          The actual pixel dimensions of a figure are determined by:
        </p>
        <BlockMath math="\text{width}_{\text{px}} = \text{figsize}[0] \times \text{dpi}, \quad \text{height}_{\text{px}} = \text{figsize}[1] \times \text{dpi}" />
        <p>
          So <code>figsize=(10, 6)</code> at 100 DPI produces a 1000 x 600 pixel image. For print-quality
          figures, use <InlineMath math="\text{dpi} \geq 300" />.
        </p>

        <h3>Color Mapping</h3>
        <p>
          Colormaps map scalar values to colors via a function <InlineMath math="C: [0, 1] \to \text{RGBA}" />.
          A <em>Normalize</em> object first maps your data range to <InlineMath math="[0, 1]" />:
        </p>
        <BlockMath math="v_{\text{norm}} = \frac{v - v_{\min}}{v_{\max} - v_{\min}}" />
        <p>
          For log-scaled colormaps, <InlineMath math="v_{\text{norm}} = \frac{\log v - \log v_{\min}}{\log v_{\max} - \log v_{\min}}" />.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <CodeBlock
          language="python"
          title="figure_axes_basics.py"
          code={`import matplotlib.pyplot as plt
import numpy as np

# ── The OO interface: explicit Figure + Axes ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

x = np.linspace(0, 4 * np.pi, 200)

# Panel 1: line plot with dual y-axis
ax1 = axes[0]
ax1.plot(x, np.sin(x), color="steelblue", linewidth=2, label="sin(x)")
ax1_twin = ax1.twinx()
ax1_twin.plot(x, np.cumsum(np.sin(x)) * (x[1] - x[0]),
              color="coral", linewidth=2, linestyle="--", label="integral")
ax1.set_title("Dual Y-Axis")
ax1.set_xlabel("x")
ax1.set_ylabel("sin(x)", color="steelblue")
ax1_twin.set_ylabel("cumulative integral", color="coral")

# Panel 2: scatter with colormap
np.random.seed(42)
n = 200
xs = np.random.randn(n)
ys = xs + np.random.randn(n) * 0.5
colors = np.sqrt(xs**2 + ys**2)
sc = axes[1].scatter(xs, ys, c=colors, cmap="viridis", alpha=0.7, edgecolors="white", s=50)
fig.colorbar(sc, ax=axes[1], label="distance from origin")
axes[1].set_title("Scatter + Colorbar")

# Panel 3: filled area + annotation
axes[2].fill_between(x, np.sin(x), alpha=0.3, color="steelblue")
axes[2].plot(x, np.sin(x), color="steelblue", linewidth=2)
peak_idx = np.argmax(np.sin(x))
axes[2].annotate("peak", xy=(x[peak_idx], 1), xytext=(x[peak_idx]+1, 0.7),
                 arrowprops=dict(arrowstyle="->", color="black"),
                 fontsize=12, fontweight="bold")
axes[2].set_title("Fill Between + Annotation")

fig.suptitle("Three-Panel Figure (OO Interface)", fontsize=14, fontweight="bold")
fig.tight_layout()
plt.savefig("three_panel.png", dpi=150, bbox_inches="tight")
plt.show()`}
        />

        <CodeBlock
          language="python"
          title="advanced_subplots.py"
          code={`import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── GridSpec for non-uniform layouts ──
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

# Large plot spanning left 2 columns, top 2 rows
ax_main = fig.add_subplot(gs[0:2, 0:2])
np.random.seed(0)
data = np.random.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], 500)
ax_main.scatter(data[:, 0], data[:, 1], alpha=0.4, s=20)
ax_main.set_title("Joint Distribution")

# Marginal histogram — top right
ax_top = fig.add_subplot(gs[0, 2])
ax_top.hist(data[:, 1], bins=30, orientation="horizontal", color="steelblue", alpha=0.7)
ax_top.set_title("Marginal Y")

# Marginal histogram — bottom left
ax_bottom = fig.add_subplot(gs[2, 0:2])
ax_bottom.hist(data[:, 0], bins=30, color="coral", alpha=0.7)
ax_bottom.set_title("Marginal X")

# Correlation heatmap — bottom right
ax_heat = fig.add_subplot(gs[1:3, 2])
corr = np.corrcoef(data.T)
im = ax_heat.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax_heat.set_xticks([0, 1])
ax_heat.set_yticks([0, 1])
ax_heat.set_xticklabels(["X", "Y"])
ax_heat.set_yticklabels(["X", "Y"])
for i in range(2):
    for j in range(2):
        ax_heat.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=14)
fig.colorbar(im, ax=ax_heat, shrink=0.6)
ax_heat.set_title("Correlation")

plt.savefig("gridspec_layout.png", dpi=150, bbox_inches="tight")
plt.show()`}
        />

        <CodeBlock
          language="python"
          title="style_and_customization.py"
          code={`import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── Global style customization ──
plt.style.use("seaborn-v0_8-whitegrid")  # built-in style

# Override specific rcParams
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Per-plot fine-tuning ──
fig, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(0, 10, 100)

for i, (alpha_val, label) in enumerate([(0.5, "Low"), (1.0, "Medium"), (2.0, "High")]):
    y = np.exp(-alpha_val * x) * np.sin(2 * np.pi * x)
    ax.plot(x, y, linewidth=2, label=f"Decay = {alpha_val}")

ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Damped Oscillations")
ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)

# Fine-tune tick formatting
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))

plt.tight_layout()
plt.show()`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>Always use the OO interface for production code</strong>: <code>fig, ax = plt.subplots()</code> gives
            you explicit references. The <code>plt.plot()</code> shortcut is fine for REPL exploration but creates
            implicit state that leads to bugs in scripts.
          </li>
          <li>
            <strong>Use <code>fig.tight_layout()</code> or <code>constrained_layout=True</code></strong>: These
            prevent labels and titles from overlapping. Pass <code>bbox_inches=&quot;tight&quot;</code> to
            <code>savefig()</code> to avoid cropping.
          </li>
          <li>
            <strong>Export to vector formats for publications</strong>: Use <code>.pdf</code> or <code>.svg</code> for
            papers and presentations. Reserve <code>.png</code> for web or when you have millions of points.
          </li>
          <li>
            <strong>Colormaps matter</strong>: Use perceptually uniform colormaps (<code>viridis</code>,
            <code>plasma</code>, <code>inferno</code>) instead of <code>jet</code> or <code>rainbow</code>, which
            distort perception and are inaccessible to colorblind viewers.
          </li>
          <li>
            <strong>Use GridSpec for complex layouts</strong>: When you need panels of different sizes or
            non-rectangular arrangements, <code>GridSpec</code> is far more flexible than <code>plt.subplots()</code>.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Mixing pyplot and OO interfaces</strong>: Calling <code>plt.xlabel()</code> when you
            have an <code>ax</code> object leads to confusion — it modifies whatever the &quot;current&quot;
            axes is, not necessarily the one you intend. Always use <code>ax.set_xlabel()</code>.
          </li>
          <li>
            <strong>Forgetting to close figures</strong>: In loops that create many plots, call
            <code>plt.close(fig)</code> after saving. Otherwise Matplotlib holds all figures in memory,
            eventually causing an <code>OutOfMemoryError</code>.
          </li>
          <li>
            <strong>Using <code>jet</code> colormap</strong>: The <code>jet</code> colormap creates artificial
            boundaries at yellow and cyan that don&apos;t correspond to data features. Use <code>viridis</code> as
            your default.
          </li>
          <li>
            <strong>Not setting figure size before plotting</strong>: If you call <code>fig.set_size_inches()</code>
            after plotting, text and markers won&apos;t scale properly. Set <code>figsize</code> at creation time.
          </li>
          <li>
            <strong>Overplotting with large datasets</strong>: Plotting 1M scatter points produces an
            opaque blob. Use <code>alpha=0.01</code>, hexbin plots, or 2D histograms instead.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> You have two datasets with different scales (revenue in millions, user count
          in thousands). How would you plot them on the same figure so both are readable, and what are the
          trade-offs of each approach?
        </p>
        <p><strong>Answer:</strong></p>
        <p>There are three main approaches, each with trade-offs:</p>
        <ol>
          <li>
            <strong>Dual y-axes (<code>ax.twinx()</code>)</strong>: Creates a second y-axis on the right side
            sharing the same x-axis. Pros: compact, easy to compare trends. Cons: can mislead viewers into
            thinking the two y-scales are comparable; crossing points are arbitrary.
          </li>
          <li>
            <strong>Two subplots with shared x-axis</strong>: Use <code>fig, (ax1, ax2) = plt.subplots(2, 1,
            sharex=True)</code>. Pros: honest, no scale confusion. Cons: takes more space, harder to visually
            align peaks.
          </li>
          <li>
            <strong>Normalize both to z-scores</strong>: Compute <InlineMath math="z = (x - \mu) / \sigma" /> for
            each series and plot on the same axis. Pros: directly comparable, single y-axis. Cons: loses original
            units, requires audience to understand standardization.
          </li>
        </ol>
        <p>
          For a stakeholder presentation, dual axes with clear color coding and explicit labels is usually the best
          balance. For a technical paper, stacked subplots are more honest.
        </p>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Matplotlib Official Tutorials</strong> — The &quot;Usage Guide&quot; and &quot;Axes API&quot; sections are the definitive references.</li>
          <li><strong>Nicolas Rougier, &quot;Scientific Visualization: Python + Matplotlib&quot; (2021)</strong> — Free online book with stunning examples and source code.</li>
          <li><strong>Edward Tufte, &quot;The Visual Display of Quantitative Information&quot;</strong> — The classic on data visualization principles (not Python-specific but essential).</li>
          <li><strong>Matplotlib cheatsheets</strong> — The official cheatsheets at matplotlib.org/cheatsheets/ are excellent quick references.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
