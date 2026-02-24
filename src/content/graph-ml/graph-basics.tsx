"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function GraphBasics() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Graphs are the natural data structure for <strong>relationships</strong>. A social network is a
          graph (users are nodes, friendships are edges). The web is a graph (pages are nodes, hyperlinks
          are edges). A knowledge base is a graph (entities are nodes, relationships are edges). Whenever
          your data has entities connected by relationships, you&apos;re looking at a graph problem.
        </p>
        <p>
          A graph <InlineMath math="G = (V, E)" /> consists of <strong>vertices</strong> (nodes)
          and <strong>edges</strong> (connections). Edges can be directed (Twitter follows) or undirected
          (Facebook friendships), weighted (distance between cities) or unweighted, and nodes and edges
          can carry features (user profiles, transaction amounts). The way you represent and traverse
          this structure determines what insights you can extract.
        </p>
        <p>
          Three fundamental algorithms capture different aspects of graph structure.
          <strong> PageRank</strong> measures node importance through the recursive insight that a page
          is important if important pages link to it. <strong>Shortest path</strong> algorithms (Dijkstra,
          BFS) find the most efficient route between nodes — critical for maps, network routing, and
          computing graph distance features. <strong>Community detection</strong> finds clusters of
          densely connected nodes — useful for identifying friend groups, market segments, or fraud rings.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Graph Representations</h3>
        <p>
          <strong>Adjacency matrix</strong> <InlineMath math="A \in \{0,1\}^{n \times n}" />
          where <InlineMath math="A_{ij} = 1" /> if edge <InlineMath math="(i, j) \in E" />:
        </p>
        <BlockMath math="A = \begin{pmatrix} 0 & 1 & 1 & 0 \\ 1 & 0 & 1 & 0 \\ 1 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}" />
        <p>
          For undirected graphs, <InlineMath math="A" /> is symmetric. Space
          complexity: <InlineMath math="O(n^2)" />. For sparse graphs (most real-world networks),
          use an <strong>adjacency list</strong>: <InlineMath math="O(n + m)" /> where <InlineMath math="m = |E|" />.
        </p>

        <p>
          <strong>Degree matrix</strong> <InlineMath math="D" /> is diagonal
          with <InlineMath math="D_{ii} = \sum_j A_{ij}" /> (number of neighbors).
          The <strong>graph Laplacian</strong>:
        </p>
        <BlockMath math="L = D - A" />
        <p>
          The eigenvalues of <InlineMath math="L" /> encode connectivity: the number of zero eigenvalues equals
          the number of connected components. The second-smallest eigenvalue (Fiedler value) measures how
          well-connected the graph is.
        </p>

        <h3>PageRank</h3>
        <p>
          Model a &quot;random surfer&quot; who follows links with probability <InlineMath math="d" /> (damping
          factor, typically 0.85) and teleports to a random page with
          probability <InlineMath math="1 - d" />. The PageRank vector <InlineMath math="\mathbf{r}" /> satisfies:
        </p>
        <BlockMath math="\mathbf{r} = d \cdot M \mathbf{r} + \frac{1-d}{n} \mathbf{1}" />
        <p>
          where <InlineMath math="M" /> is the column-normalized adjacency matrix
          (<InlineMath math="M_{ij} = A_{ij} / \text{out-degree}(j)" />).
          This is equivalent to the dominant eigenvector of the modified transition matrix. Solved iteratively
          via <strong>power iteration</strong>: start with
          uniform <InlineMath math="\mathbf{r}^{(0)} = \frac{1}{n}\mathbf{1}" /> and
          repeat until convergence.
        </p>

        <h3>Shortest Path</h3>
        <p><strong>Dijkstra&apos;s algorithm</strong> for weighted graphs with non-negative edges:</p>
        <BlockMath math="\text{dist}(v) = \min_{u \in \text{neighbors}(v)} \left[ \text{dist}(u) + w(u, v) \right]" />
        <p>
          Time complexity: <InlineMath math="O((n + m) \log n)" /> with a min-heap.
          For unweighted graphs, <strong>BFS</strong> achieves <InlineMath math="O(n + m)" />.
        </p>

        <h3>Modularity (Community Detection)</h3>
        <p>
          The Louvain algorithm maximizes <strong>modularity</strong> <InlineMath math="Q" />, which measures
          how much denser the connections within communities are compared to a random graph:
        </p>
        <BlockMath math="Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)" />
        <p>
          where <InlineMath math="k_i" /> is the degree of node <InlineMath math="i" />,
          <InlineMath math="m" /> is the total number of edges,
          and <InlineMath math="\delta(c_i, c_j) = 1" /> if nodes <InlineMath math="i" /> and <InlineMath math="j" /> are
          in the same community. <InlineMath math="Q \in [-0.5, 1]" />, with higher values indicating stronger
          community structure.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Graph Representation and PageRank from Scratch</h3>
        <CodeBlock
          language="python"
          title="pagerank_scratch.py"
          code={`import numpy as np

def pagerank(adj_matrix, damping=0.85, max_iter=100, tol=1e-6):
    """
    Compute PageRank via power iteration.

    Args:
        adj_matrix: (n, n) adjacency matrix
        damping: probability of following a link (vs random teleport)
    Returns:
        PageRank scores (sum to 1)
    """
    n = adj_matrix.shape[0]

    # Column-normalize: M[i][j] = prob of going from j to i
    out_degree = adj_matrix.sum(axis=0)
    # Handle dangling nodes (no outgoing links)
    out_degree[out_degree == 0] = 1
    M = adj_matrix / out_degree  # column-normalized

    r = np.ones(n) / n  # uniform initialization

    for _ in range(max_iter):
        r_new = damping * M @ r + (1 - damping) / n
        r_new /= r_new.sum()  # normalize

        if np.abs(r_new - r).sum() < tol:
            break
        r = r_new

    return r

# Example: small web graph
# 0 -> 1, 0 -> 2, 1 -> 2, 2 -> 0, 3 -> 2
adj = np.array([
    [0, 0, 1, 0],   # edges INTO node 0
    [1, 0, 0, 0],   # edges INTO node 1
    [1, 1, 0, 1],   # edges INTO node 2 (most linked-to)
    [0, 0, 0, 0],   # edges INTO node 3
])

pr = pagerank(adj)
for i, score in enumerate(pr):
    print(f"  Node {i}: PageRank = {score:.4f}")

# Node 2 has highest PageRank (most incoming links)`}
        />

        <h3>Graph Algorithms with NetworkX</h3>
        <CodeBlock
          language="python"
          title="graph_algorithms_networkx.py"
          code={`import networkx as nx
import numpy as np

# --- Build a graph ---
G = nx.karate_club_graph()  # Famous social network (34 nodes)
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# --- PageRank ---
pr = nx.pagerank(G, alpha=0.85)
top_nodes = sorted(pr.items(), key=lambda x: -x[1])[:5]
print("\\nTop 5 by PageRank:")
for node, score in top_nodes:
    print(f"  Node {node}: {score:.4f}")

# --- Shortest Paths ---
path = nx.shortest_path(G, source=0, target=33)
print(f"\\nShortest path 0->33: {path} (length {len(path)-1})")

# All-pairs shortest path lengths
avg_path_length = nx.average_shortest_path_length(G)
print(f"Average shortest path length: {avg_path_length:.2f}")

# --- Community Detection (Louvain) ---
from networkx.algorithms.community import louvain_communities

communities = louvain_communities(G, seed=42)
print(f"\\nDetected {len(communities)} communities:")
for i, comm in enumerate(communities):
    print(f"  Community {i}: {sorted(comm)}")

modularity = nx.community.modularity(G, communities)
print(f"Modularity: {modularity:.4f}")

# --- Centrality Measures ---
degree_cent = nx.degree_centrality(G)
between_cent = nx.betweenness_centrality(G)
close_cent = nx.closeness_centrality(G)

print("\\nNode 0 centralities:")
print(f"  Degree: {degree_cent[0]:.4f}")
print(f"  Betweenness: {between_cent[0]:.4f}")
print(f"  Closeness: {close_cent[0]:.4f}")`}
        />

        <h3>Dijkstra&apos;s Algorithm from Scratch</h3>
        <CodeBlock
          language="python"
          title="dijkstra_scratch.py"
          code={`import heapq
from collections import defaultdict

def dijkstra(graph, source):
    """
    Dijkstra's shortest path using a min-heap.

    Args:
        graph: dict of {node: [(neighbor, weight), ...]}
        source: starting node
    Returns:
        dist: dict of shortest distances from source
        prev: dict for reconstructing paths
    """
    dist = {source: 0}
    prev = {}
    heap = [(0, source)]  # (distance, node)
    visited = set()

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        for v, weight in graph.get(u, []):
            new_dist = d + weight
            if v not in dist or new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    return dist, prev

def reconstruct_path(prev, target):
    path = []
    node = target
    while node in prev:
        path.append(node)
        node = prev[node]
    path.append(node)
    return path[::-1]

# Example: weighted graph
graph = {
    "A": [("B", 4), ("C", 1)],
    "B": [("D", 1)],
    "C": [("B", 2), ("D", 5)],
    "D": [("E", 3)],
    "E": [],
}

dist, prev = dijkstra(graph, "A")
print("Shortest distances from A:")
for node, d in sorted(dist.items()):
    path = reconstruct_path(prev, node)
    print(f"  A -> {node}: distance={d}, path={' -> '.join(path)}")

# A -> A: distance=0, path=A
# A -> B: distance=3, path=A -> C -> B
# A -> C: distance=1, path=A -> C
# A -> D: distance=4, path=A -> C -> B -> D
# A -> E: distance=7, path=A -> C -> B -> D -> E`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Graphs are everywhere in industry</strong>: Social networks (Facebook has 3B+ node graph), fraud detection (transaction graphs at Stripe/PayPal), knowledge graphs (Google, Amazon), supply chains, and biological networks.</li>
          <li><strong>Graph features are powerful ML features</strong>: PageRank, degree centrality, clustering coefficient, and community membership are often among the strongest features in models for social/transactional data.</li>
          <li><strong>Use NetworkX for prototyping, not production</strong>: NetworkX is pure Python and single-threaded. For graphs with millions of nodes, use graph-tool (C++), iGraph, Neo4j (graph database), or Apache Spark GraphX.</li>
          <li><strong>Sparse representation is critical</strong>: Real graphs are sparse — a social network with 1B users has ~1B * 500 edges (average friends), not 1B^2. Always use adjacency lists or sparse matrices (CSR format), never dense adjacency matrices.</li>
          <li><strong>Community detection has no single best algorithm</strong>: Louvain is fast and popular, but Label Propagation is faster for very large graphs. Infomap is better for directed/weighted graphs. Always validate communities with domain experts.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Using dense adjacency matrices for large graphs</strong>: A 100K-node graph needs a 100K x 100K matrix (10B entries, ~80GB). The actual graph might have only 500K edges — use sparse representation instead.</li>
          <li><strong>Forgetting dangling nodes in PageRank</strong>: Nodes with no outgoing edges (e.g., a page with no links) cause the transition matrix to have zero columns. Without handling this, PageRank mass leaks away. Add teleportation for dangling nodes.</li>
          <li><strong>Applying Dijkstra to negative-weight edges</strong>: Dijkstra assumes all edge weights are non-negative. With negative weights, use Bellman-Ford (<InlineMath math="O(nm)" />) instead.</li>
          <li><strong>Treating graph structure as static</strong>: Real graphs evolve — new edges form, nodes join and leave. Models trained on a snapshot may not generalize. Use temporal graph methods or retrain regularly.</li>
          <li><strong>Conflating high PageRank with importance for your task</strong>: PageRank measures a specific notion of importance (link-based authority). For other tasks (influence maximization, recommendation), different centrality measures may be more appropriate.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Explain PageRank intuitively. How does it handle the &quot;dead end&quot; (dangling node) problem and the &quot;spider trap&quot; problem? What is the time complexity?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Intuition</strong>: PageRank models a random surfer who starts on a random page and repeatedly follows a random outgoing link. The PageRank of a page is the fraction of time the surfer spends on that page in the long run. Pages with many incoming links from other important pages accumulate more &quot;surfer time.&quot;</li>
          <li><strong>Dead ends (dangling nodes)</strong>: Pages with no outgoing links trap the surfer. Solution: when the surfer hits a dead end, they teleport to a random page uniformly. Mathematically, we redistribute the &quot;leaked&quot; probability mass uniformly across all nodes.</li>
          <li><strong>Spider traps</strong>: A group of pages that link only to each other absorb all the probability mass over time. Solution: with probability <InlineMath math="1 - d" /> (typically 0.15), the surfer teleports to any random page, ensuring no group can monopolize the score.</li>
          <li><strong>Complexity</strong>: Each power iteration is <InlineMath math="O(m)" /> (one sparse matrix-vector multiplication, where <InlineMath math="m = |E|" />). PageRank typically converges in 50-100 iterations, so total cost is <InlineMath math="O(km)" /> where <InlineMath math="k" /> is the number of iterations. For the web (billions of nodes, hundreds of billions of edges), this takes minutes on a distributed system like MapReduce.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Page et al. (1999) &quot;The PageRank Citation Ranking&quot;</strong> — The original Stanford paper by Larry Page and Sergey Brin.</li>
          <li><strong>CLRS &quot;Introduction to Algorithms&quot; Chapter 22-24</strong> — BFS, DFS, Dijkstra, Bellman-Ford, strongly connected components.</li>
          <li><strong>Blondel et al. (2008) &quot;Fast unfolding of communities in large networks&quot;</strong> — The Louvain algorithm paper.</li>
          <li><strong>NetworkX documentation</strong> — Comprehensive graph algorithm library for Python prototyping.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
