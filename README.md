# Multimodal Retail Search Engine

A production-grade, bi-directional e-commerce discovery pipeline that bridges text queries and raw images into a single vector space. Powered by **CLIP (`clip-ViT-B-32`)** and optimized using **FAISS** for fast, dense vector similarity retrieval.

## 📌 Architecture Overview

Traditional e-commerce platforms rely heavily on exact string matching or siloed image-to-image pipelines. This project implements a **Text-Output Multimodal Architecture** that solves the modality bias problem inherent in dual-index tracking systems.
Query (Text OR Image) ──> CLIP Encoder ──> 512D Vector Space ──> FAISS Index (Text Only)
│
▼
Structured Catalog Match
(Category, SKU, Specs)

By indexing dense text vectors aligned via contrastive pre-training, the engine adaptively routes both text queries and visual signatures to a single, unified database index. The system guarantees a structured product detail output regardless of the user's input type.

---

## 🛠️ Tech Stack & Dependencies

* **Core Framework:** PyTorch
* **Vector Indexing:** FAISS (`faiss-cpu`)
* **Embedding Model:** SentenceTransformers (`clip-ViT-B-32`)
* **Data Layer:** HuggingFace `datasets` (Fashion MNIST catalog tracking)
* **Visualization:** Matplotlib, Seaborn, Scikit-learn

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/multimodal-retail-search.git](https://github.com/YOUR_USERNAME/multimodal-retail-search.git)
cd multimodal-retail-search

2. Install Requirements
Bash
pip install sentence-transformers faiss-cpu datasets pillow matplotlib seaborn scikit-learn

3. Execution FlowThe code is modularized into sequential production milestones:Dataset Initialization: Fetches the retail catalog and constructs descriptive textual metadata profiles.Embedding Extraction: Passes data tokens through the CLIP encoder to extract 512-dimensional vector representations.Index Rebuilding: Normalizes embeddings ($L_2$) and binds them into a searchable FAISS IndexFlatIP pool.Search Interface: Natively processes both visual image uploads and natural text query formats.📊 Evaluation MetricsThe pipeline is quantitatively validated using information retrieval benchmarks to monitor system performance under catalog constraints:Mean Reciprocal Rank (MRR): Evaluates the average reciprocal rank of the first relevant item, measuring how effectively target classes surface in the top deck.Confusion Matrix Heatmap: Tracks cross-modal visual misclassifications (e.g., verifying vector separation between morphologically similar items like Pullovers and Coats).Precision@K & Recall@K Curves: Profiles retrieval quality across dynamic search depths ($K = 1$ to $K = 10$).
