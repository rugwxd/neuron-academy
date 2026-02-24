"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function MultiModal() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Humans effortlessly connect what they see with what they read — you can look at a photo of a dog
          catching a frisbee and match it to the sentence &quot;a dog leaps to catch a frisbee.&quot;
          <strong> Multi-modal learning</strong> teaches machines to do the same: learn a shared representation
          space where images, text, audio, and other modalities can be compared and combined.
        </p>
        <p>
          The breakthrough came with <strong>CLIP (Contrastive Language-Image Pre-training)</strong> from OpenAI.
          CLIP trains an image encoder and a text encoder jointly so that matching image-text pairs have similar
          embeddings and non-matching pairs are pushed apart. Once trained on 400 million image-text pairs from
          the internet, CLIP can classify images into <em>any</em> set of categories described in natural
          language — without ever being explicitly trained on those categories. This is called
          <strong> zero-shot classification</strong>.
        </p>
        <p>
          Beyond CLIP, models like <strong>Flamingo</strong> and <strong>LLaVA</strong> take it further by
          feeding visual features directly into a language model, enabling open-ended visual question answering.
          The key insight across all these models is the same: learn to <strong>align representations across
          modalities</strong> so that semantically similar content — regardless of whether it is an image, a
          sentence, or a sound clip — lives nearby in embedding space.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>CLIP: Contrastive Loss Across Modalities</h3>
        <p>
          Given a batch of <InlineMath math="N" /> image-text pairs, CLIP encodes each image
          <InlineMath math="x_i^{\text{img}}" /> and each text <InlineMath math="x_j^{\text{txt}}" /> into
          embeddings <InlineMath math="z_i^{\text{img}}" /> and <InlineMath math="z_j^{\text{txt}}" />. The
          similarity matrix is:
        </p>
        <BlockMath math="s_{ij} = \frac{(z_i^{\text{img}})^\top z_j^{\text{txt}}}{\|z_i^{\text{img}}\| \cdot \|z_j^{\text{txt}}\|} \cdot e^t" />
        <p>
          where <InlineMath math="t" /> is a learned temperature parameter (log-parameterized). The loss is
          a symmetric cross-entropy over the <InlineMath math="N \times N" /> similarity matrix:
        </p>
        <BlockMath math="\mathcal{L}_{\text{CLIP}} = \frac{1}{2N} \sum_{i=1}^{N} \left[ -\log \frac{e^{s_{ii}}}{\sum_{j=1}^{N} e^{s_{ij}}} - \log \frac{e^{s_{ii}}}{\sum_{j=1}^{N} e^{s_{ji}}} \right]" />
        <p>
          The first term treats each image as a query and finds the matching text (image-to-text retrieval).
          The second term treats each text as a query and finds the matching image (text-to-image retrieval).
          The diagonal entries <InlineMath math="s_{ii}" /> are the positive pairs.
        </p>

        <h3>Zero-Shot Classification with CLIP</h3>
        <p>
          To classify an image <InlineMath math="x" /> into one of <InlineMath math="K" /> categories, create
          text prompts like &quot;a photo of a [class]&quot; for each class, encode them, and pick the most
          similar text:
        </p>
        <BlockMath math="p(y = k \mid x) = \frac{\exp(\text{sim}(z^{\text{img}}, z_k^{\text{txt}}) / \tau)}{\sum_{j=1}^{K} \exp(\text{sim}(z^{\text{img}}, z_j^{\text{txt}}) / \tau)}" />
        <p>
          No labeled training data for these specific classes is needed — the model&apos;s understanding of
          language connects visual features to class descriptions.
        </p>

        <h3>Vision-Language Models (Flamingo Architecture)</h3>
        <p>
          Flamingo interleaves frozen visual features with a frozen language model using <strong>Perceiver
          Resampler</strong> cross-attention layers. Given image features <InlineMath math="V \in \mathbb{R}^{M \times d_v}" /> and
          a set of learned latent queries <InlineMath math="Q \in \mathbb{R}^{L \times d}" /> (where <InlineMath math="L \ll M" />):
        </p>
        <BlockMath math="\text{Perceiver}(Q, V) = \text{softmax}\left(\frac{QW_Q (VW_K)^\top}{\sqrt{d}}\right) VW_V" />
        <p>
          This compresses <InlineMath math="M" /> visual tokens into <InlineMath math="L" /> fixed-length tokens
          that are then inserted between text tokens in the language model via gated cross-attention layers:
        </p>
        <BlockMath math="h_l = h_l + \tanh(\alpha_l) \cdot \text{CrossAttn}(h_l, \text{Perceiver}(Q, V))" />
        <p>
          where <InlineMath math="\alpha_l" /> is a learned gating parameter initialized to zero so that the
          model starts as a pure language model and gradually learns to incorporate visual information.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Zero-Shot Image Classification with CLIP</h3>
        <CodeBlock
          language="python"
          title="clip_zero_shot.py"
          code={`import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load pre-trained CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# Load an image
image = Image.open("example.jpg")

# Define candidate classes (zero-shot: no training needed!)
candidate_labels = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird",
    "a photo of a car",
    "a photo of a person playing guitar",
]

# Process inputs
inputs = processor(
    text=candidate_labels,
    images=image,
    return_tensors="pt",
    padding=True,
)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    # outputs.logits_per_image: (1, num_labels)
    logits = outputs.logits_per_image
    probs = logits.softmax(dim=-1)

# Results
print("Zero-Shot Classification Results:")
for label, prob in zip(candidate_labels, probs[0]):
    print(f"  {label}: {prob.item():.3f}")
predicted = candidate_labels[probs[0].argmax()]
print(f"\\nPredicted: {predicted}")`}
        />

        <h3>Computing Image-Text Similarity</h3>
        <CodeBlock
          language="python"
          title="clip_similarity.py"
          code={`import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

def get_image_embedding(image_path):
    """Encode a single image into CLIP embedding space."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    # L2 normalize for cosine similarity
    return emb / emb.norm(dim=-1, keepdim=True)

def get_text_embedding(text):
    """Encode text into CLIP embedding space."""
    inputs = processor(text=text, return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    return emb / emb.norm(dim=-1, keepdim=True)

def image_text_similarity(image_path, texts):
    """Compute cosine similarity between one image and multiple texts."""
    img_emb = get_image_embedding(image_path)   # (1, 512)
    txt_emb = get_text_embedding(texts)          # (N, 512)
    # Cosine similarity
    similarities = (img_emb @ txt_emb.T).squeeze(0)  # (N,)
    return similarities.numpy()

# --- Example: rank descriptions by relevance to an image ---
descriptions = [
    "a sunset over the ocean",
    "a cat sitting on a windowsill",
    "a busy city street at night",
    "mountains covered in snow",
    "a bowl of fresh fruit on a table",
]

scores = image_text_similarity("sunset_photo.jpg", descriptions)
ranked = sorted(zip(descriptions, scores), key=lambda x: -x[1])
print("Descriptions ranked by relevance:")
for desc, score in ranked:
    print(f"  {score:.3f}  {desc}")`}
        />

        <h3>Building a Simple Image Search System</h3>
        <CodeBlock
          language="python"
          title="clip_image_search.py"
          code={`import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPImageSearch:
    """
    Text-to-image search using CLIP embeddings.
    Index a folder of images, then search with natural language.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.image_paths = []
        self.image_embeddings = None

    def index_folder(self, folder_path, extensions=(".jpg", ".png", ".jpeg")):
        """Compute and cache embeddings for all images in a folder."""
        folder = Path(folder_path)
        self.image_paths = sorted([
            p for p in folder.iterdir()
            if p.suffix.lower() in extensions
        ])
        print(f"Indexing {len(self.image_paths)} images...")

        embeddings = []
        batch_size = 32
        for i in range(0, len(self.image_paths), batch_size):
            batch_paths = self.image_paths[i:i + batch_size]
            images = [
                Image.open(p).convert("RGB") for p in batch_paths
            ]
            inputs = self.processor(
                images=images, return_tensors="pt", padding=True
            )
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu())

        self.image_embeddings = torch.cat(embeddings, dim=0)  # (N, 512)
        print(f"Indexed {self.image_embeddings.shape[0]} images.")

    def search(self, query, top_k=5):
        """Search images using a natural language query."""
        inputs = self.processor(
            text=query, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            text_emb = self.model.get_text_features(**inputs)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # Cosine similarity against all indexed images
        similarities = (text_emb @ self.image_embeddings.T).squeeze(0)
        top_indices = similarities.argsort(descending=True)[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "path": str(self.image_paths[idx]),
                "score": similarities[idx].item(),
            })
        return results

    def save_index(self, path):
        """Save embeddings to disk for fast reloading."""
        torch.save({
            "paths": [str(p) for p in self.image_paths],
            "embeddings": self.image_embeddings,
        }, path)

    def load_index(self, path):
        """Load pre-computed embeddings."""
        data = torch.load(path)
        self.image_paths = [Path(p) for p in data["paths"]]
        self.image_embeddings = data["embeddings"]
        print(f"Loaded index with {len(self.image_paths)} images.")

# --- Usage ---
searcher = CLIPImageSearch()
searcher.index_folder("./my_photos")

# Search with natural language
results = searcher.search("a dog playing in the snow", top_k=5)
print("\\nSearch results for: a dog playing in the snow")
for r in results:
    print(f"  Score: {r['score']:.3f}  Path: {r['path']}")

# Try different queries on the same index
for query in ["sunset at the beach", "people eating dinner", "a red car"]:
    results = searcher.search(query, top_k=3)
    print(f"\\nQuery: {query}")
    for r in results:
        print(f"  {r['score']:.3f}  {r['path']}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>CLIP excels at zero-shot but not at fine-grained tasks</strong>: CLIP matches or beats supervised models on many benchmarks without any task-specific training. However, it struggles with fine-grained classification (distinguishing bird species, car models) and spatial reasoning (&quot;the cat is to the left of the dog&quot;). Fine-tune or use specialized models for these cases.</li>
          <li><strong>Prompt engineering matters for CLIP</strong>: The text template significantly affects zero-shot accuracy. &quot;A photo of a [class]&quot; works better than just &quot;[class]&quot;. OpenAI found that ensembling 80+ prompt templates (e.g., &quot;a blurry photo of a [class]&quot;, &quot;a sculpture of a [class]&quot;) improves accuracy by 3-5%.</li>
          <li><strong>CLIP embeddings are excellent for retrieval</strong>: The shared embedding space enables cross-modal search (text query, image results) with simple cosine similarity. This powers image search, content moderation, and recommendation systems at scale.</li>
          <li><strong>Vision-language models need careful decoding</strong>: Models like LLaVA and Flamingo can hallucinate objects not present in images. Use constrained decoding, temperature scaling, and structured output formats to improve reliability in production.</li>
          <li><strong>Multi-modal models are data hungry</strong>: CLIP was trained on 400M image-text pairs. For domain-specific applications (medical imaging, satellite imagery), you often need to fine-tune on domain data because internet-scraped pairs do not cover specialized vocabulary.</li>
          <li><strong>Use SigLIP over CLIP for new projects</strong>: SigLIP replaces the softmax-based contrastive loss with a sigmoid loss that operates on individual pairs rather than the full batch, allowing more efficient training with smaller batch sizes while achieving comparable or better performance.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Not normalizing embeddings before computing similarity</strong>: CLIP&apos;s contrastive loss operates on cosine similarity. If you compute dot products without L2-normalizing the embeddings first, your similarity scores will be dominated by embedding magnitude rather than semantic direction.</li>
          <li><strong>Using CLIP for spatial reasoning</strong>: CLIP encodes images holistically and does not capture spatial relationships well. &quot;A dog chasing a cat&quot; and &quot;a cat chasing a dog&quot; may get very similar embeddings. Use object detection models or specialized VLMs for spatial understanding.</li>
          <li><strong>Ignoring CLIP&apos;s biases</strong>: CLIP inherits biases from internet-scraped training data. It can associate certain demographics with specific occupations or attributes. Always audit CLIP-based systems for fairness before deployment.</li>
          <li><strong>Fine-tuning CLIP aggressively</strong>: Full fine-tuning with a high learning rate destroys the pre-trained alignment. Use very small learning rates (<InlineMath math="10^{-6}" /> to <InlineMath math="10^{-5}" />) or freeze one encoder and only fine-tune the other. Consider LoRA or linear probing before full fine-tuning.</li>
          <li><strong>Treating multi-modal as always better</strong>: Adding modalities only helps when they provide complementary information. If the text already contains all the signal, adding images may just increase compute cost and noise without improving accuracy.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Design a product image search system where users type natural language queries and get matching product images. How would you architect this?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Offline indexing pipeline</strong>:
            <ul>
              <li>Use a CLIP image encoder (or a fine-tuned variant on product data) to embed all product images into 512-dimensional vectors.</li>
              <li>Store embeddings in a vector database (Pinecone, Weaviate, FAISS) with product metadata.</li>
              <li>For products with multiple images, embed each separately or create an average embedding.</li>
            </ul>
          </li>
          <li><strong>Online query pipeline</strong>:
            <ul>
              <li>Encode the user&apos;s text query with CLIP&apos;s text encoder.</li>
              <li>Perform approximate nearest neighbor search (HNSW or IVF-PQ) against the image embedding index.</li>
              <li>Return top-K results ranked by cosine similarity.</li>
            </ul>
          </li>
          <li><strong>Improvements for production</strong>:
            <ul>
              <li>Fine-tune CLIP on your product catalog using product title as the text and product image as the image (in-domain contrastive training).</li>
              <li>Use a two-stage retrieval: CLIP for coarse retrieval (top-100), then a cross-encoder for re-ranking (top-10).</li>
              <li>Add metadata filters (category, price range, availability) as pre-filters before vector search.</li>
            </ul>
          </li>
          <li><strong>Evaluation</strong>: Measure recall@K, MRR, and NDCG on a human-annotated query-image relevance dataset. A/B test against keyword-based search to measure engagement uplift.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Radford et al. (2021) &quot;Learning Transferable Visual Models From Natural Language Supervision&quot;</strong> — The CLIP paper. Demonstrates that contrastive learning on 400M image-text pairs produces strong zero-shot visual classifiers.</li>
          <li><strong>Alayrac et al. (2022) &quot;Flamingo: a Visual Language Model for Few-Shot Learning&quot;</strong> — Introduces gated cross-attention to inject visual features into frozen language models for few-shot visual reasoning.</li>
          <li><strong>Li et al. (2022) &quot;BLIP: Bootstrapping Language-Image Pre-training&quot;</strong> — Combines contrastive, matching, and generation objectives with a caption filtering mechanism for cleaner training data.</li>
          <li><strong>Liu et al. (2023) &quot;Visual Instruction Tuning&quot; (LLaVA)</strong> — Shows that connecting a CLIP visual encoder to an LLM with a simple projection layer and instruction-tuning yields strong visual conversation abilities.</li>
          <li><strong>Zhai et al. (2023) &quot;Sigmoid Loss for Language Image Pre-Training&quot; (SigLIP)</strong> — Replaces CLIP&apos;s softmax contrastive loss with a pairwise sigmoid loss, removing the need for large batch sizes while maintaining performance.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
