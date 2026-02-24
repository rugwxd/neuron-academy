# Neuron Academy

A comprehensive, free, open-source learning platform covering **data science, machine learning, and AI** — from Python basics to RLHF, from SQL window functions to distributed training, from descriptive stats to transformers.

**Live site:** [rugwxd.github.io/neuron-academy](https://rugwxd.github.io/neuron-academy)

## What This Is

Neuron Academy is a single resource that takes you from zero to applied scientist. It covers everything across 33 modules and 96 in-depth topic pages — each one structured the same way:

1. **Plain English** — The concept explained clearly, assuming you're smart but new to it
2. **The Math** — Formal definitions, derivations, and notation (rendered with KaTeX)
3. **The Code** — Working Python examples, from scratch and with libraries
4. **In Practice** — When to use it, when not to, real-world gotchas
5. **Common Mistakes** — What people get wrong and why
6. **Interview Question** — A real question with a detailed solution
7. **Go Deeper** — Papers and further reading

## What It Covers

| Part | Modules |
|------|---------|
| **Foundations** | Python, Linear Algebra, Calculus, Probability, NumPy/Pandas, SQL, Visualization |
| **Statistics** | Descriptive Stats, Distributions, CLT, Hypothesis Testing, Confidence Intervals, Bayesian Stats, A/B Testing, Causal Inference |
| **Machine Learning** | Linear/Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, SVMs, KNN, K-Means, PCA, t-SNE/UMAP, Cross-Validation, Metrics, Hyperparameter Tuning |
| **Deep Learning** | Perceptrons, Backpropagation, CNNs, RNNs, Seq2Seq, PyTorch, Optimizers, Activation Functions |
| **Transformers & LLMs** | Self-Attention, Positional Encoding, Full Architecture, Scaling Laws, Pre-training, Fine-tuning, RLHF, RAG, Prompt Engineering |
| **Specialized** | GANs, VAEs, Diffusion Models, Reinforcement Learning, Recommendation Systems, Time Series, Search & Ranking, Graph Neural Networks |
| **Production** | Feature Stores, ML System Design, MLOps, Model Serving, Monitoring, Distributed Training, Spark |
| **Interview Prep** | SHAP/LIME, Fairness, Probability Puzzles, SQL Challenges, System Design Case Studies, Self-Supervised Learning, Knowledge Distillation, Multi-Modal Learning |

## Tech Stack

- **Next.js 16** with App Router and static export
- **TypeScript** + **Tailwind CSS**
- **KaTeX** for math rendering
- **GitHub Pages** for hosting

## Development

```bash
npm install
npm run dev     # http://localhost:3000
npm run build   # static export to out/
```

## License

MIT
