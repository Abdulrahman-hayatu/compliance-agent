# Nigerian Fintech Regulatory Compliance Agent

A multi-agent LLM application that checks fintech policy documents against Nigerian regulatory frameworks. Upload a PDF or TXT policy document and the pipeline extracts discrete policy claims, retrieves relevant regulatory clauses from a pre-built vector index, assesses each claim for compliance, and generates a structured markdown report with a remediation checklist. The tool currently covers two frameworks: the **CBN Circular and Guidelines for the Operations of Agent Banking in Nigeria (October 6, 2025)** and the **Nigeria Data Protection Act 2023 (NDPC)**.


---

## ⚠️ Scope Notice

> This tool covers **CBN Agent Banking Guidelines (October 2025)** and the **Nigeria Data Protection Act 2023** only. It does not cover all CBN regulations.

---

## Tech Stack

- **Language:** Python 3.10+
- **Agent Framework:** LangGraph
- **LLM API:** Groq (`llama-3.3-70b-versatile`)
- **Embeddings:** `sentence-transformers` (`BAAI/bge-small-en-v1.5`)
- **Vector Store:** FAISS (`IndexFlatL2`)
- **Document Parsing:** `pdfplumber`
- **UI:** Gradio
- **Deployment:** Hugging Face Spaces

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Abdulrahman-Hayatu/compliance-agent.git
cd compliance-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Copy the example env file and add your Groq API key:

```bash
cp .env.example .env
```

Open `.env` and set:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 4. Build the FAISS index

Run the indexer **once** before starting the app. This reads both regulatory PDFs from `data/`, chunks and embeds them, and saves the index to `index/`.

```bash
python -m rag.indexer
```

Expected output:
```
CBN chunks : 41
NDPC chunks: 13
Total      : 54
✓ Indexing complete.
```

### 5. Run the app

```bash
python app.py
```

Open the local URL printed in the terminal (e.g. `http://127.0.0.1:7860`).

---

## How It Works

The pipeline is built with **LangGraph** and runs four agents in sequence:

1. **Parser Agent** — Sends the uploaded document text to the Groq LLM, which extracts discrete, testable policy claims as a JSON array. Each claim is a single sentence describing what the document asserts about operations, data handling, agent relationships, customer treatment, or security.

2. **Retrieval Agent** — For each extracted claim, queries the pre-built FAISS vector index using `BAAI/bge-small-en-v1.5` embeddings to retrieve the four most semantically relevant regulatory chunks from the CBN and NDPC documents.

3. **Compliance Checker Agent** — Sends each claim alongside its retrieved regulatory context to the Groq LLM, which returns a structured JSON verdict: `COMPLIANT`, `NON_COMPLIANT`, or `UNCLEAR`, with a specific regulation reference, a one-sentence explanation, and a remediation action where applicable.

4. **Report Generator Agent** — Passes all compliance results to the Groq LLM to produce a structured markdown report containing an executive summary, a compliance scorecard, detailed per-claim findings, a prioritised remediation checklist, and a scope limitations note.

---

## Project Structure

```
compliance-agent/
├── app.py                  # Gradio UI entry point
├── agents/
│   ├── parser_agent.py     # Agent 1: Document Parser
│   ├── retrieval_agent.py  # Agent 2: Retrieval Agent
│   ├── checker_agent.py    # Agent 3: Compliance Checker
│   └── report_agent.py     # Agent 4: Report Generator
├── graph/
│   ├── state.py            # ComplianceState TypedDict
│   └── workflow.py         # LangGraph state graph definition
├── rag/
│   ├── indexer.py          # Script to build and save FAISS index
│   └── retriever.py        # FAISS query interface
├── data/
│   ├── CIRCULAR AND GUIDELINES FOR THE OPERATIONS OF AGENT BANKING IN NIGERIA OCTOBER 6 2025.pdf
│   └── Nigeria_Data_Protection_Act_2023.pdf
├── index/
│   ├── compliance_index.faiss
│   └── chunks.pkl
├── requirements.txt
├── .env.example
└── README.md
```

---

> Do **not** run `rag/indexer.py` at app startup — the index is pre-built and committed to the repo.