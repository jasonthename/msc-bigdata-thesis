# MSc Big Data Thesis — RAG Math Solver

Retrieval-Augmented Generation (RAG) pipeline for solving primary and middle-school math problems using instructional PDF corpus retrieval and an LLM solver.

All notebooks were developed and run in **Google Colab**. No local hardware requirements apply.

---

## Notebooks

### `data_harvest.ipynb`
Crawls [mathwords.com](https://www.mathwords.com) across a set of math topic categories and saves each term's instructional page as a PDF to Google Drive. Uses Playwright with print-media emulation to render pages faithfully. Includes resume-safe skip logic and randomised rate-limiting to avoid bot detection.

### `ingest_corpus.ipynb`
Ingests the harvested PDFs into an Elasticsearch index (`math-corpus-v1`). For each PDF page it extracts selectable text via PyMuPDF, renders the page as a PNG image, and produces a 2048-dimensional multimodal embedding using **NVIDIA `llama-nemotron-embed-vl-1b-v2`** (image + text). Documents are bulk-indexed with cosine-similarity dense vector search enabled.

### `main_experiments.ipynb`
Runs the core thesis experiments across three solver conditions:

| Condition | Description |
|-----------|-------------|
| **A** | LLM-only baseline — no retrieval |
| **B** | Retrieval attempted; guidance used regardless of score |
| **C** | Full RAG — retrieval guidance applied only when the Elasticsearch score exceeds a confidence threshold |

Each condition is evaluated against a fixed 15-problem thesis test set. The solver uses **Google Gemini 2.5 Flash** via OpenRouter and returns structured LaTeX or visual outputs (Venn diagrams, number lines, fraction bars). Also includes an image-based problem extraction path using a vision model.

> **GPU requirement:** Run `main_experiments.ipynb` on a minimum **NVIDIA T4** runtime in Google Colab (Runtime → Change runtime type → T4 GPU). This is required for loading the Nemotron embedding model at usable speed.

---

## Setup

1. Mount Google Drive and place harvested PDFs under `MyDrive/MSc AI and Big Data/thesis/data/`.
2. Add the following secrets to Colab (or a `.env` file):
   - `ELASTIC_HOST`, `ELASTIC_API_KEY` (or `ELASTIC_USER` + `ELASTIC_PASSWORD`)
   - `OPENROUTER_API_KEY`
3. Run the notebooks in order: `data_harvest` → `ingest_corpus` → `main_experiments`.
