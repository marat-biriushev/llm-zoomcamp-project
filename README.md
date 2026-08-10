# PCI DSS Assistant

A question-answering assistant over the **PCI DSS v4.0.1** standard — the security
standard every company that stores, processes, or transmits payment card data has
to comply with.

> Status: work in progress. This README is updated as each step is completed.

## Problem

PCI DSS v4.0.1 is a 397-page document containing 300+ individual requirements.
Engineers, auditors, and compliance officers constantly need answers to concrete
questions:

- *How long do we have to retain audit logs?*
- *Do we need MFA for administrative access, or only for remote access?*
- *Can we store the CVV after the transaction is authorized?*
- *What exactly does requirement 8.3.6 ask for?*

Finding the answer means opening a PDF, guessing the right keyword, and reading
around a table that spans three columns. The answer is almost always there — the
problem is retrieval, not the absence of information. That is exactly the shape of
problem RAG solves.

This project ingests the standard, retrieves the relevant pages for a question, and
has an LLM answer using only those pages, always citing the requirement number and
page so the answer can be verified against the source.

## Data

- **Source:** PCI DSS v4.0.1 (June 2024), published by the PCI Security Standards Council.
- **Size:** 397 pages, of which 261 belong to the "Requirements and Testing Procedures" section.
- **Chunking:** one document per page. The standard is laid out so that a page
  almost always holds a single requirement together with its testing procedures and
  guidance (1.4 requirements per page on average), which makes the page a natural
  unit of meaning. Documents are 800–3600 characters, ~2100 on average.

The PDF is **not** committed to this repository. The PCI SSC publishes it for free
but restricts redistribution, so `ingest.download_pdf()` fetches it at setup time.

## Plan

The project is built step by step. Steps 1–7 mirror the structure of the
LLM Zoomcamp modules; steps 8–11 cover what the course did not.

| # | Step | What it delivers | Criterion |
|---|------|------------------|-----------|
| 1 | Data | PDF parsed into a list of documents | — |
| 2 | Text search | minsearch index, retrieve pages for a question | Retrieval flow |
| 3 | First RAG | context → prompt → OpenAI → answer with citations | Retrieval flow |
| 4 | Ground truth | LLM-generated questions for each page | — |
| 5 | Search evaluation | hit rate and MRR for text search | Retrieval evaluation |
| 6 | Vector & hybrid search | embeddings in Qdrant, re-measured against the baseline | Retrieval evaluation, hybrid search |
| 7 | Answer evaluation | several prompts compared with LLM-as-a-judge | LLM evaluation |
| 8 | Interface | Streamlit chat UI | Interface |
| 9 | Feedback & monitoring | Postgres + Grafana dashboard | Monitoring |
| 10 | Ingestion pipeline | automated ingestion | Ingestion pipeline |
| 11 | Docker & docs | one-command startup, full README | Containerization, reproducibility |

### Progress

- [x] 1 — Data
- [x] 2 — Text search
- [x] 3 — First RAG
- [x] 4 — Ground truth
- [ ] 5 — Search evaluation
- [ ] 6 — Vector & hybrid search
- [ ] 7 — Answer evaluation
- [ ] 8 — Interface
- [ ] 9 — Feedback & monitoring
- [ ] 10 — Ingestion pipeline
- [ ] 11 — Docker & docs

## Technologies

| Area | Choice | Why |
|------|--------|-----|
| PDF parsing | PyMuPDF | reads page text directly, no OCR needed |
| Text search | minsearch | in-memory TF-IDF, no infrastructure; serves as the retrieval baseline |
| Vector search | Qdrant *(step 6)* | stores dense and sparse vectors in one collection, so hybrid search needs no extra service |
| LLM | OpenAI | — |
| Interface | Streamlit *(step 8)* | — |
| Monitoring | Postgres + Grafana *(step 9)* | — |

## Running it so far

Requires [uv](https://docs.astral.sh/uv/) and an OpenAI API key.

```bash
git clone <this-repo>
cd llm-zoomcamp-project

cp .env.example .env      # then put your OPENAI_API_KEY in it
uv sync

uv run jupyter notebook rag.ipynb
```

Run the cells top to bottom. Step 1 downloads the standard and turns it into 261
documents.

## Repository layout

```
ingest.py      loading the PDF and building the search index
rag.ipynb      the working notebook — one section per step
data/          the downloaded PDF (gitignored)
```

## License and attribution

PCI DSS v4.0.1 is © 2006–2024 PCI Security Standards Council, LLC. This project
does not redistribute the standard; it downloads it and indexes it locally for
question answering. It is an educational project and is not affiliated with, nor
endorsed by, the PCI Security Standards Council.
