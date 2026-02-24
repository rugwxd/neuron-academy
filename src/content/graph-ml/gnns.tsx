"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function GraphNeuralNetworks() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Graph Neural Networks (GNNs) extend deep learning to graph-structured data. The core idea
          is <strong>message passing</strong>: each node collects information from its neighbors,
          aggregates it, and updates its own representation. After several rounds of message passing,
          each node&apos;s representation encodes information about its local neighborhood — and by
          stacking layers, information from further-away nodes propagates inward.
        </p>
        <p>
          Think of it like a game of telephone, but structured. In round 1, each node asks its
          immediate neighbors &quot;what are you?&quot; and summarizes the answers. In round 2, each
          node asks its neighbors again — but now its neighbors already contain information about
          <em>their</em> neighbors. So after 2 rounds, each node has information about nodes up to
          2 hops away. After <InlineMath math="k" /> layers, each node&apos;s representation captures
          its <InlineMath math="k" />-hop neighborhood.
        </p>
        <p>
          The three landmark architectures are <strong>GCN</strong> (Graph Convolutional Network) which
          averages neighbor features with normalization, <strong>GraphSAGE</strong> which samples
          neighbors and uses learnable aggregators (making it scalable to huge graphs), and <strong>GAT</strong>
          (Graph Attention Network) which learns attention weights to decide how much each neighbor
          matters — the graph equivalent of the attention mechanism in Transformers.
        </p>
        <p>
          GNNs power production systems at Pinterest (PinSage for recommendations, processing 3B+
          nodes), Google Maps (ETA prediction on road graphs), Twitter/X (user and content
          recommendations), drug discovery at Deepmind, and fraud detection at PayPal and Stripe.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Message Passing Framework</h3>
        <p>
          Every GNN follows the same pattern. For each node <InlineMath math="v" /> at
          layer <InlineMath math="l" />:
        </p>
        <BlockMath math="\mathbf{m}_v^{(l)} = \text{AGGREGATE}\left(\left\{ \mathbf{h}_u^{(l-1)} : u \in \mathcal{N}(v) \right\}\right)" />
        <BlockMath math="\mathbf{h}_v^{(l)} = \text{UPDATE}\left(\mathbf{h}_v^{(l-1)}, \mathbf{m}_v^{(l)}\right)" />
        <p>
          where <InlineMath math="\mathbf{h}_v^{(0)} = \mathbf{x}_v" /> (input node features)
          and <InlineMath math="\mathcal{N}(v)" /> is the set of neighbors of <InlineMath math="v" />.
          Different GNN variants choose different AGGREGATE and UPDATE functions.
        </p>

        <h3>GCN (Kipf &amp; Welling, 2017)</h3>
        <p>Spectral-inspired convolution with symmetric normalization:</p>
        <BlockMath math="H^{(l)} = \sigma\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l-1)} W^{(l)}\right)" />
        <p>where:</p>
        <ul>
          <li><InlineMath math="\tilde{A} = A + I" /> — adjacency matrix with self-loops (a node also &quot;messages&quot; itself)</li>
          <li><InlineMath math="\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}" /> — degree matrix of <InlineMath math="\tilde{A}" /></li>
          <li><InlineMath math="W^{(l)}" /> — learnable weight matrix at layer <InlineMath math="l" /></li>
          <li><InlineMath math="\sigma" /> — nonlinearity (typically ReLU)</li>
        </ul>
        <p>Per-node, this is equivalent to:</p>
        <BlockMath math="\mathbf{h}_v^{(l)} = \sigma\left(W^{(l)} \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{\mathbf{h}_u^{(l-1)}}{\sqrt{|\mathcal{N}(u)| \cdot |\mathcal{N}(v)|}}\right)" />

        <h3>GraphSAGE (Hamilton et al., 2017)</h3>
        <p>Sample a fixed-size neighborhood and use a learnable aggregator:</p>
        <BlockMath math="\mathbf{h}_{\mathcal{N}(v)}^{(l)} = \text{AGG}\left(\left\{ \mathbf{h}_u^{(l-1)} : u \in \text{SAMPLE}(\mathcal{N}(v), k) \right\}\right)" />
        <BlockMath math="\mathbf{h}_v^{(l)} = \sigma\left(W^{(l)} \cdot \text{CONCAT}\left(\mathbf{h}_v^{(l-1)}, \mathbf{h}_{\mathcal{N}(v)}^{(l)}\right)\right)" />
        <p>
          AGG can be mean, max-pool, or LSTM. Sampling <InlineMath math="k" /> neighbors (typically 10-25)
          per layer makes computation independent of the full graph size — this is why GraphSAGE scales
          to billions of nodes at Pinterest.
        </p>

        <h3>GAT (Velickovic et al., 2018)</h3>
        <p>Learn attention weights for each edge:</p>
        <BlockMath math="e_{vu} = \text{LeakyReLU}\left(\mathbf{a}^T \left[ W\mathbf{h}_v \| W\mathbf{h}_u \right]\right)" />
        <BlockMath math="\alpha_{vu} = \frac{\exp(e_{vu})}{\sum_{k \in \mathcal{N}(v)} \exp(e_{vk})}" />
        <BlockMath math="\mathbf{h}_v^{(l)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu} W \mathbf{h}_u^{(l-1)}\right)" />
        <p>
          where <InlineMath math="\|" /> denotes concatenation
          and <InlineMath math="\mathbf{a}" /> is a learnable attention vector. Multi-head attention
          is used exactly as in Transformers — concatenate or average <InlineMath math="K" /> heads.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>GCN from Scratch</h3>
        <CodeBlock
          language="python"
          title="gcn_scratch.py"
          code={`import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)

    def forward(self, X, A_hat):
        """
        X: (n, in_features) node feature matrix
        A_hat: (n, n) normalized adjacency (D^{-1/2} A_tilde D^{-1/2})
        """
        # Message passing: aggregate neighbor features
        AX = A_hat @ X           # (n, in_features)
        # Transform
        return self.W(AX)        # (n, out_features)

class GCN(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes):
        super().__init__()
        self.layer1 = GCNLayer(n_features, n_hidden)
        self.layer2 = GCNLayer(n_hidden, n_classes)

    def forward(self, X, A_hat):
        H = F.relu(self.layer1(X, A_hat))
        H = F.dropout(H, p=0.5, training=self.training)
        return self.layer2(H, A_hat)

def normalize_adjacency(A):
    """Compute D^{-1/2} A_tilde D^{-1/2}."""
    A_tilde = A + torch.eye(A.shape[0])  # Add self-loops
    D = A_tilde.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D))
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt

# Example: Cora-like node classification
n_nodes = 100
n_features = 16
n_classes = 3

# Random graph and features (replace with real data)
A = (torch.rand(n_nodes, n_nodes) > 0.95).float()
A = (A + A.T).clamp(max=1)  # Make symmetric
X = torch.randn(n_nodes, n_features)
y = torch.randint(0, n_classes, (n_nodes,))
A_hat = normalize_adjacency(A)

# Train
model = GCN(n_features, 32, n_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(200):
    model.train()
    logits = model(X, A_hat)
    loss = F.cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        acc = (logits.argmax(dim=1) == y).float().mean()
        print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.2%}")`}
        />

        <h3>GNN with PyTorch Geometric</h3>
        <CodeBlock
          language="python"
          title="gnn_pyg.py"
          code={`import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.datasets import Planetoid

# Load Cora dataset (2,708 papers, 7 classes, 5,429 edges)
dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]
print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
print(f"Features: {data.num_node_features}, Classes: {dataset.num_classes}")

# --- GCN ---
class GCNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 64)
        self.conv2 = GCNConv(64, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# --- GraphSAGE ---
class SAGEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(dataset.num_node_features, 64)
        self.conv2 = SAGEConv(64, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

# --- GAT ---
class GATModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(dataset.num_node_features, 8, heads=8)
        self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return self.conv2(x, edge_index)

# Train loop (same for all models)
model = GCNModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
print(f"Test accuracy: {test_acc:.2%}")

# Typical results on Cora:
# GCN:       ~81% test accuracy
# GraphSAGE: ~79% test accuracy
# GAT:       ~83% test accuracy`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>2-3 layers is usually optimal</strong>: With too many layers, GNNs suffer from <strong>over-smoothing</strong> — all node representations converge to the same vector. In practice, 2-3 layers work best for most tasks.</li>
          <li><strong>GraphSAGE for scale</strong>: GCN requires the full graph in memory (full-batch). GraphSAGE uses mini-batch training with neighbor sampling, making it practical for graphs with billions of nodes. PinSage (Pinterest) processes 3B+ pins using this approach.</li>
          <li><strong>GAT for heterogeneous importance</strong>: When neighbors contribute unequally (e.g., some friends influence you more than others), GAT&apos;s learned attention weights automatically handle this.</li>
          <li><strong>Node features matter enormously</strong>: A GNN on a graph with rich node features (text embeddings, user profiles) will vastly outperform one with only graph structure. If you have no node features, use one-hot degree encoding or random features.</li>
          <li><strong>GNN + downstream task</strong>: Node classification (label each node), link prediction (predict new edges), and graph classification (classify entire graphs — used in molecular property prediction) are the three main tasks.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Over-smoothing with too many layers</strong>: After 5+ GCN layers, all node embeddings become nearly identical. Use skip connections (ResGCN), jumping knowledge, or graph-level readouts to mitigate this.</li>
          <li><strong>Ignoring edge direction</strong>: In social networks, &quot;A follows B&quot; does not mean &quot;B follows A.&quot; Using an undirected GCN on a directed graph loses important asymmetric information.</li>
          <li><strong>Data leakage in link prediction</strong>: If you&apos;re predicting whether an edge exists, you must remove that edge from the message-passing graph. Otherwise the model trivially detects its own input.</li>
          <li><strong>Not using self-loops</strong>: Without self-loops (<InlineMath math="\tilde{A} = A + I" />), a node&apos;s own features get lost during aggregation. GCN adds them by default, but custom implementations often forget.</li>
          <li><strong>Full-batch training on large graphs</strong>: GCN requires the full adjacency matrix per forward pass, which doesn&apos;t fit in GPU memory for large graphs. Use GraphSAGE-style mini-batching or cluster-GCN.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Compare GCN, GraphSAGE, and GAT. Which would you choose for a recommendation system with 100 million users and 10 billion interactions? Why?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>GCN</strong>: Averages neighbor features with symmetric normalization. Elegant but requires full-graph training — impossible for 100M nodes. Best for small to medium graphs (up to ~1M nodes) where all data fits in GPU memory.</li>
          <li><strong>GraphSAGE</strong>: Samples a fixed number of neighbors per layer and uses mini-batch training. This makes it <strong>inductive</strong> (can generalize to unseen nodes without retraining) and scalable. This is the right choice here because:
            <ul>
              <li>Mini-batch training: sample 2 hops of 25 neighbors each = at most 625 nodes per training example, regardless of graph size.</li>
              <li>Inductive: new users can get embeddings immediately without retraining.</li>
              <li>Pinterest&apos;s PinSage handles 3B+ nodes using this approach.</li>
            </ul>
          </li>
          <li><strong>GAT</strong>: Learns per-edge attention weights. More expressive than GCN but standard GAT is also full-batch. For 100M nodes, you&apos;d need to combine GAT layers with GraphSAGE-style sampling (which PyTorch Geometric supports).</li>
          <li><strong>Recommendation</strong>: Use a <strong>two-tower GraphSAGE</strong> — one tower for users, one for items. Train on the user-item interaction graph with neighbor sampling. Generate embeddings offline, serve via approximate nearest neighbor search (FAISS/ScaNN). This is the architecture behind PinSage and similar production systems.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Kipf &amp; Welling (2017) &quot;Semi-Supervised Classification with Graph Convolutional Networks&quot;</strong> — The GCN paper that ignited the GNN revolution.</li>
          <li><strong>Hamilton et al. (2017) &quot;Inductive Representation Learning on Large Graphs&quot;</strong> — GraphSAGE paper with the sampling framework.</li>
          <li><strong>Velickovic et al. (2018) &quot;Graph Attention Networks&quot;</strong> — GAT: attention for graphs.</li>
          <li><strong>Ying et al. (2018) &quot;Graph Convolutional Neural Networks for Web-Scale Recommender Systems&quot;</strong> — PinSage at Pinterest: GNNs at billion-node scale.</li>
          <li><strong>Stanford CS224W: Machine Learning with Graphs</strong> — Jure Leskovec&apos;s excellent course with lecture videos on YouTube.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
