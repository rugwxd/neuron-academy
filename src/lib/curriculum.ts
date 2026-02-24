export interface Topic {
  slug: string;
  title: string;
  description: string;
}

export interface Module {
  id: number;
  slug: string;
  title: string;
  part: string;
  partNumber: number;
  topics: Topic[];
}

export const curriculum: Module[] = [
  // PART I: FOUNDATIONS
  {
    id: 1,
    slug: "python-for-data-science",
    title: "Python for Data Science",
    part: "Foundations",
    partNumber: 1,
    topics: [
      { slug: "variables-and-types", title: "Variables, Data Types & Operators", description: "The building blocks of Python" },
      { slug: "control-flow", title: "Control Flow", description: "if/else, loops, comprehensions, generators" },
      { slug: "functions", title: "Functions & Closures", description: "Functions, *args, **kwargs, lambda, closures, decorators" },
      { slug: "oop", title: "Classes & OOP", description: "Inheritance, dunder methods, and when to use classes in DS" },
    ],
  },
  {
    id: 2,
    slug: "mathematics-for-ml",
    title: "Mathematics for ML",
    part: "Foundations",
    partNumber: 1,
    topics: [
      { slug: "linear-algebra", title: "Linear Algebra", description: "Vectors, matrices, eigenvalues, SVD, and the geometry of data" },
      { slug: "calculus-and-optimization", title: "Calculus & Optimization", description: "Derivatives, gradients, and gradient descent on 3D surfaces" },
      { slug: "probability", title: "Probability Theory", description: "From sample spaces to Bayes' theorem and common distributions" },
      { slug: "information-theory", title: "Information Theory", description: "Entropy, cross-entropy, KL divergence — why they're everywhere in ML" },
      { slug: "optimization-theory", title: "Optimization Theory", description: "Convexity, SGD, Adam, and convergence" },
    ],
  },
  {
    id: 3,
    slug: "data-manipulation",
    title: "Data Manipulation",
    part: "Foundations",
    partNumber: 1,
    topics: [
      { slug: "numpy", title: "NumPy", description: "Arrays, broadcasting, vectorization" },
      { slug: "pandas", title: "Pandas", description: "DataFrames, groupby, merge, window functions" },
      { slug: "data-cleaning", title: "Data Cleaning", description: "Missing values, outliers, type casting" },
    ],
  },
  {
    id: 4,
    slug: "sql-mastery",
    title: "SQL Mastery",
    part: "Foundations",
    partNumber: 1,
    topics: [
      { slug: "fundamentals", title: "SQL Fundamentals", description: "SELECT, WHERE, GROUP BY, HAVING, ORDER BY" },
      { slug: "joins", title: "JOINs", description: "INNER, LEFT, RIGHT, FULL, CROSS, SELF — with visual diagrams" },
      { slug: "window-functions", title: "Window Functions", description: "ROW_NUMBER, RANK, LAG, LEAD, running totals" },
      { slug: "advanced-patterns", title: "Advanced SQL Patterns", description: "CTEs, gaps-and-islands, sessionization, funnel analysis" },
    ],
  },
  {
    id: 5,
    slug: "visualization-and-eda",
    title: "Visualization & EDA",
    part: "Foundations",
    partNumber: 1,
    topics: [
      { slug: "matplotlib", title: "Matplotlib Deep Dive", description: "Figures, axes, subplots, customization" },
      { slug: "eda-methodology", title: "EDA Methodology", description: "What to look for, in what order" },
    ],
  },
  // PART II: STATISTICS & EXPERIMENTATION
  {
    id: 6,
    slug: "statistical-foundations",
    title: "Statistical Foundations",
    part: "Statistics & Experimentation",
    partNumber: 2,
    topics: [
      { slug: "descriptive-stats", title: "Descriptive Statistics", description: "Mean, median, variance, skewness, kurtosis" },
      { slug: "distributions", title: "Distributions Deep Dive", description: "Normal, t, chi-square, binomial, Poisson — interactive PDF/CDF" },
      { slug: "central-limit-theorem", title: "Central Limit Theorem", description: "Sample from any distribution, watch means become normal" },
      { slug: "maximum-likelihood", title: "Maximum Likelihood Estimation", description: "Derive MLE for Gaussian, Bernoulli, Poisson" },
    ],
  },
  {
    id: 7,
    slug: "statistical-inference",
    title: "Statistical Inference",
    part: "Statistics & Experimentation",
    partNumber: 2,
    topics: [
      { slug: "confidence-intervals", title: "Confidence Intervals", description: "Construction, interpretation, bootstrap CIs" },
      { slug: "hypothesis-testing", title: "Hypothesis Testing", description: "Framework, Type I/II errors, power" },
      { slug: "t-tests", title: "t-Tests", description: "One-sample, two-sample, paired" },
      { slug: "multiple-testing", title: "Multiple Testing Correction", description: "Bonferroni, Holm, Benjamini-Hochberg FDR" },
    ],
  },
  {
    id: 8,
    slug: "bayesian-statistics",
    title: "Bayesian Statistics",
    part: "Statistics & Experimentation",
    partNumber: 2,
    topics: [
      { slug: "bayes-theorem", title: "Bayes' Theorem", description: "From coin flips to real problems" },
      { slug: "mcmc", title: "MCMC", description: "Metropolis-Hastings, Gibbs sampling — with animation" },
    ],
  },
  {
    id: 9,
    slug: "experimentation",
    title: "Experimentation & Causal Inference",
    part: "Statistics & Experimentation",
    partNumber: 2,
    topics: [
      { slug: "ab-testing", title: "A/B Testing", description: "Sample size, power analysis, sequential testing" },
      { slug: "causal-inference", title: "Causal Inference", description: "DAGs, propensity scores, diff-in-diff" },
    ],
  },
  // PART III: MACHINE LEARNING
  {
    id: 10,
    slug: "supervised-learning",
    title: "Supervised Learning",
    part: "Machine Learning",
    partNumber: 3,
    topics: [
      { slug: "linear-regression", title: "Linear Regression", description: "From scratch with gradient descent — interactive regression line" },
      { slug: "logistic-regression", title: "Logistic Regression", description: "Decision boundaries and maximum likelihood" },
      { slug: "decision-trees", title: "Decision Trees", description: "Watch splits animate, information gain vs Gini" },
      { slug: "random-forests", title: "Random Forests", description: "Bagging, feature importance, why ensembles reduce variance" },
      { slug: "gradient-boosting", title: "Gradient Boosting", description: "Residual fitting, XGBoost, LightGBM, CatBoost" },
      { slug: "svm", title: "Support Vector Machines", description: "Hard/soft margin, kernel trick — interactive visualization" },
      { slug: "bias-variance", title: "Bias-Variance Tradeoff", description: "The fundamental ML concept — interactive complexity slider" },
    ],
  },
  {
    id: 11,
    slug: "evaluation-and-tuning",
    title: "Evaluation & Tuning",
    part: "Machine Learning",
    partNumber: 3,
    topics: [
      { slug: "cross-validation", title: "Cross-Validation", description: "k-fold, stratified, time series, nested CV" },
      { slug: "metrics", title: "Evaluation Metrics", description: "Accuracy, precision, recall, F1, ROC-AUC, PR-AUC" },
      { slug: "hyperparameter-tuning", title: "Hyperparameter Tuning", description: "Grid search, random search, Bayesian optimization" },
    ],
  },
  {
    id: 12,
    slug: "unsupervised-learning",
    title: "Unsupervised Learning",
    part: "Machine Learning",
    partNumber: 3,
    topics: [
      { slug: "k-means", title: "K-Means Clustering", description: "Interactive centroid convergence, elbow method" },
      { slug: "pca", title: "PCA", description: "Dimensionality reduction, variance explained, relationship to SVD" },
      { slug: "t-sne-umap", title: "t-SNE & UMAP", description: "Nonlinear dimensionality reduction for visualization" },
    ],
  },
  {
    id: 13,
    slug: "ensemble-methods",
    title: "Ensemble Methods & Advanced ML",
    part: "Machine Learning",
    partNumber: 3,
    topics: [
      { slug: "bagging-vs-boosting", title: "Bagging vs Boosting", description: "Visual comparison of ensemble strategies" },
      { slug: "xgboost-internals", title: "XGBoost Internals", description: "Split finding, regularization, system design" },
    ],
  },
  // PART IV: DEEP LEARNING
  {
    id: 14,
    slug: "neural-networks",
    title: "Neural Networks from Scratch",
    part: "Deep Learning",
    partNumber: 4,
    topics: [
      { slug: "perceptron", title: "Perceptron & XOR Problem", description: "Where it all started" },
      { slug: "backpropagation", title: "Backpropagation", description: "Step-by-step with computational graph animation" },
      { slug: "activation-functions", title: "Activation Functions", description: "Sigmoid, ReLU, GELU, Swish — why each exists" },
      { slug: "optimizers", title: "Optimizers Deep Dive", description: "SGD, momentum, Adam — loss surface animation" },
    ],
  },
  {
    id: 15,
    slug: "pytorch",
    title: "PyTorch Mastery",
    part: "Deep Learning",
    partNumber: 4,
    topics: [
      { slug: "tensors", title: "Tensors & Autograd", description: "Computational graphs and automatic differentiation" },
      { slug: "training-loop", title: "Training Loop", description: "From scratch, then with Lightning" },
    ],
  },
  {
    id: 16,
    slug: "cnns",
    title: "Convolutional Neural Networks",
    part: "Deep Learning",
    partNumber: 4,
    topics: [
      { slug: "convolution", title: "Convolution Operation", description: "Watch filters slide, see feature maps" },
      { slug: "architectures", title: "Classic Architectures", description: "LeNet to EfficientNet — evolution" },
    ],
  },
  {
    id: 17,
    slug: "sequence-models",
    title: "Sequence Models",
    part: "Deep Learning",
    partNumber: 4,
    topics: [
      { slug: "rnns", title: "RNNs & LSTMs", description: "Gate mechanisms, vanishing gradients" },
      { slug: "seq2seq", title: "Seq2Seq with Attention", description: "The model that started it all" },
    ],
  },
  {
    id: 18,
    slug: "transformers",
    title: "Transformers",
    part: "Deep Learning",
    partNumber: 4,
    topics: [
      { slug: "self-attention", title: "Self-Attention from Scratch", description: "Query/key/value — see attention weights interactively" },
      { slug: "full-architecture", title: "The Full Transformer", description: "Encoder-decoder with data flow animation" },
      { slug: "positional-encoding", title: "Positional Encoding", description: "Sinusoidal, learned, RoPE, ALiBi" },
      { slug: "scaling-laws", title: "Scaling Laws", description: "Chinchilla, compute-optimal training" },
    ],
  },
  // PART V: NLP & LANGUAGE MODELS
  {
    id: 19,
    slug: "nlp-foundations",
    title: "NLP Foundations",
    part: "NLP & Language Models",
    partNumber: 5,
    topics: [
      { slug: "tokenization", title: "Tokenization", description: "Word, subword BPE, WordPiece, SentencePiece" },
      { slug: "embeddings", title: "Word Embeddings", description: "Word2Vec, GloVe — explore nearest neighbors" },
    ],
  },
  {
    id: 20,
    slug: "large-language-models",
    title: "Large Language Models",
    part: "NLP & Language Models",
    partNumber: 5,
    topics: [
      { slug: "pretraining", title: "Pre-training", description: "Masked LM, causal LM, next-token prediction" },
      { slug: "fine-tuning", title: "Fine-tuning & LoRA", description: "Full fine-tuning, adapters, LoRA, QLoRA" },
      { slug: "rlhf", title: "RLHF & Alignment", description: "Reward modeling, PPO, DPO — the alignment pipeline" },
      { slug: "rag", title: "RAG", description: "Retrieval-Augmented Generation architecture" },
      { slug: "prompt-engineering", title: "Prompt Engineering", description: "Zero-shot, few-shot, chain-of-thought" },
    ],
  },
  // PART VI: SPECIALIZED DOMAINS
  {
    id: 21,
    slug: "generative-models",
    title: "Generative Models",
    part: "Specialized Domains",
    partNumber: 6,
    topics: [
      { slug: "vaes", title: "Variational Autoencoders", description: "ELBO, reparameterization, latent space" },
      { slug: "gans", title: "GANs", description: "Generator/discriminator training, mode collapse" },
      { slug: "diffusion", title: "Diffusion Models", description: "Forward/reverse process, DDPM, Stable Diffusion" },
    ],
  },
  {
    id: 22,
    slug: "reinforcement-learning",
    title: "Reinforcement Learning",
    part: "Specialized Domains",
    partNumber: 6,
    topics: [
      { slug: "mdps", title: "Markov Decision Processes", description: "States, actions, rewards, Bellman equations" },
      { slug: "q-learning", title: "Q-Learning", description: "Tabular Q-learning with interactive grid world" },
      { slug: "policy-gradients", title: "Policy Gradients & PPO", description: "REINFORCE, actor-critic, PPO" },
    ],
  },
  {
    id: 23,
    slug: "recommendation-systems",
    title: "Recommendation Systems",
    part: "Specialized Domains",
    partNumber: 6,
    topics: [
      { slug: "collaborative-filtering", title: "Collaborative Filtering", description: "User-user, item-item, matrix factorization" },
      { slug: "deep-recsys", title: "Deep Learning for RecSys", description: "NCF, Wide & Deep, two-tower models" },
    ],
  },
  {
    id: 24,
    slug: "time-series",
    title: "Time Series & Forecasting",
    part: "Specialized Domains",
    partNumber: 6,
    topics: [
      { slug: "components", title: "Time Series Components", description: "Trend, seasonality, decomposition" },
      { slug: "arima", title: "ARIMA / SARIMA", description: "Box-Jenkins methodology, model selection" },
    ],
  },
  {
    id: 25,
    slug: "search-and-ranking",
    title: "Search & Ranking",
    part: "Specialized Domains",
    partNumber: 6,
    topics: [
      { slug: "information-retrieval", title: "Information Retrieval", description: "Inverted index, TF-IDF, BM25" },
      { slug: "learning-to-rank", title: "Learning to Rank", description: "Pointwise, pairwise, listwise approaches" },
    ],
  },
  {
    id: 26,
    slug: "graph-ml",
    title: "Graph Machine Learning",
    part: "Specialized Domains",
    partNumber: 6,
    topics: [
      { slug: "graph-basics", title: "Graph Basics & Algorithms", description: "PageRank, shortest path, community detection" },
      { slug: "gnns", title: "Graph Neural Networks", description: "Message passing, GCN, GraphSAGE, GAT" },
    ],
  },
  // PART VII: PRODUCTION & SYSTEMS
  {
    id: 27,
    slug: "feature-engineering",
    title: "Feature Engineering at Scale",
    part: "Production & Systems",
    partNumber: 7,
    topics: [
      { slug: "feature-stores", title: "Feature Stores", description: "Feast, Tecton, online/offline features" },
      { slug: "encoding-strategies", title: "Encoding Strategies", description: "One-hot, target, frequency, embeddings" },
    ],
  },
  {
    id: 28,
    slug: "ml-system-design",
    title: "ML System Design",
    part: "Production & Systems",
    partNumber: 7,
    topics: [
      { slug: "framework", title: "System Design Framework", description: "The ML system design interview framework" },
      { slug: "case-studies", title: "Case Studies", description: "Netflix, Stripe, Google, Instagram, Uber, Meta" },
    ],
  },
  {
    id: 29,
    slug: "mlops",
    title: "MLOps & Production ML",
    part: "Production & Systems",
    partNumber: 7,
    topics: [
      { slug: "model-serving", title: "Model Serving", description: "FastAPI, TorchServe, Triton" },
      { slug: "monitoring", title: "Model Monitoring", description: "Data drift, concept drift, performance degradation" },
    ],
  },
  {
    id: 30,
    slug: "distributed-computing",
    title: "Distributed Computing for ML",
    part: "Production & Systems",
    partNumber: 7,
    topics: [
      { slug: "spark", title: "Apache Spark", description: "RDDs, DataFrames, PySpark" },
      { slug: "distributed-training", title: "Distributed Training", description: "Data/model/pipeline parallelism, FSDP, DeepSpeed" },
    ],
  },
  // PART VIII: INTERVIEW PREP
  {
    id: 31,
    slug: "interview-questions",
    title: "Interview Questions Bank",
    part: "Interview Prep & Mastery",
    partNumber: 8,
    topics: [
      { slug: "probability-puzzles", title: "Probability Puzzles", description: "20+ problems with solutions" },
      { slug: "sql-challenges", title: "SQL Challenges", description: "30+ real interview queries" },
      { slug: "ml-system-design-questions", title: "ML System Design Questions", description: "8+ full case studies" },
    ],
  },
  {
    id: 32,
    slug: "interpretability",
    title: "Model Interpretability",
    part: "Interview Prep & Mastery",
    partNumber: 8,
    topics: [
      { slug: "shap", title: "SHAP", description: "Shapley values from game theory to ML" },
      { slug: "fairness", title: "Fairness & Responsible AI", description: "Demographic parity, equalized odds" },
    ],
  },
  {
    id: 33,
    slug: "advanced-topics",
    title: "Advanced Topics",
    part: "Interview Prep & Mastery",
    partNumber: 8,
    topics: [
      { slug: "self-supervised", title: "Self-Supervised Learning", description: "Contrastive learning, SimCLR, MAE" },
      { slug: "multi-modal", title: "Multi-Modal Learning", description: "CLIP, vision-language models" },
      { slug: "knowledge-distillation", title: "Knowledge Distillation", description: "Teacher-student networks" },
    ],
  },
];

export function getModule(slug: string): Module | undefined {
  return curriculum.find((m) => m.slug === slug);
}

export function getTopic(moduleSlug: string, topicSlug: string): { module: Module; topic: Topic } | undefined {
  const mod = getModule(moduleSlug);
  if (!mod) return undefined;
  const topic = mod.topics.find((t) => t.slug === topicSlug);
  if (!topic) return undefined;
  return { module: mod, topic };
}

export function getPartModules(partNumber: number): Module[] {
  return curriculum.filter((m) => m.partNumber === partNumber);
}

export const parts = [
  { number: 1, title: "Foundations", color: "blue" },
  { number: 2, title: "Statistics & Experimentation", color: "green" },
  { number: 3, title: "Machine Learning", color: "purple" },
  { number: 4, title: "Deep Learning", color: "orange" },
  { number: 5, title: "NLP & Language Models", color: "red" },
  { number: 6, title: "Specialized Domains", color: "teal" },
  { number: 7, title: "Production & Systems", color: "yellow" },
  { number: 8, title: "Interview Prep & Mastery", color: "pink" },
];
