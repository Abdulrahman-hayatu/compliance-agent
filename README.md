---
title: Nigerian Fintech Compliance Agent
emoji: 🔥
colorFrom: red
colorTo: red
sdk: gradio
sdk_version: 6.14.0
python_version: '3.13'
app_file: app.py
pinned: false
license: apache-2.0
---
# 🇳🇬 Nigerian Fintech Regulatory Compliance Agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-1C3C3C?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-gpt--oss--120b-F55036?logo=groq&logoColor=white)](https://console.groq.com)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?logo=gradio&logoColor=white)](https://gradio.app)
[![HuggingFace](https://img.shields.io/badge/🤗%20Spaces-Live%20Demo-FFD21E)](https://huggingface.co/spaces/Abdulrahman-Hayatu/Nigerian-fintech-compliance-agent)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

A multi-agent LLM application that checks fintech policy documents against Nigerian regulatory frameworks. Upload a PDF or TXT policy document and the pipeline extracts discrete policy claims, retrieves relevant regulatory clauses from a pre-built FAISS vector index, assesses each claim for compliance, and produces a structured markdown report with a prioritised remediation checklist. The tool covers two frameworks: the **CBN Circular and Guidelines for the Operations of Agent Banking in Nigeria (October 6, 2025)** and the **Nigeria Data Protection Act 2023 (NDPC)**.

**[🚀 Try the Live Demo](https://huggingface.co/spaces/Abdulrahman-Hayatu/Nigerian-fintech-compliance-agent)**

---

## ⚠️ Scope Notice

> This tool covers the **CBN Agent Banking Guidelines (October 2025)** and the **Nigeria Data Protection Act 2023** only. It does not cover all CBN regulations. Outputs do not constitute legal advice — always consult a qualified compliance officer.

---

## How It Works

The pipeline is built with **LangGraph** and runs four specialised agents in sequence:

| Step | Agent | Role |
|------|-------|------|
| 1 | **Parser Agent** | Extracts discrete, testable policy claims from the uploaded document as a JSON array |
| 2 | **Retrieval Agent** | Queries the FAISS index to fetch the top-3 most relevant regulatory chunks per claim |
| 3 | **Compliance Checker Agent** | Assesses each claim against its regulatory context — returns `COMPLIANT`, `NON_COMPLIANT`, or `UNCLEAR` with a clause reference and explanation |
| 4 | **Report Generator Agent** | Produces a structured markdown report with executive summary, scorecard, findings, and remediation checklist |

Every retrieved chunk carries a stable ID and a citable clause/section label (e.g. `[CBN 9.1]`, `[NDPA PART VI]`), and the Checker Agent is instructed to cite only from what was actually retrieved — if nothing relevant surfaces, it returns `UNCLEAR` with `N/A` rather than guessing a reference.

---

## Evaluation

The pipeline is scored against a hand-reviewed **60-example golden dataset** (`eval/golden_dataset.jsonl`) spanning both source documents and four difficulty tiers:

| Tier | Description |
|------|-------------|
| `easy` | Single-clause, direct lookup |
| `multi_hop` | Requires connecting 2+ clauses |
| `adversarial` | Worded to sound compliant while violating a clause, or vice versa |
| `out_of_scope` | Genuinely outside both documents — correct answer is always `UNCLEAR` |

Each example carries an `expected_verdict` and `expected_clause_ids` (referencing the real, stable chunk IDs), reviewed across multiple passes to verify every citation actually appears in the indexed text it claims to.

### Running the eval harness

```bash
# Free — no Groq API calls, just checks retrieval quality
python -m eval.run_eval --retrieval-only

# Costs Groq tokens — runs the real checker_agent and scores verdicts
python -m eval.run_eval --verdict --output eval/results.json

# Smoke-test on a subset first
python -m eval.run_eval --verdict --limit 10
```

`eval/metrics.py` computes per-label precision/recall/F1, a confusion matrix, context-recall (with partial credit), and accuracy broken down by difficulty tier — all as pure, unit-testable functions with no API dependency.

### Latest result

60/60 claims, full verdict pass:

| Metric | Value |
|---|---|
| Accuracy | 91.7% |
| Macro F1 | 0.918 |
| Easy tier accuracy | 100% |
| Multi-hop tier accuracy | 86.7% |
| Adversarial tier accuracy | 86.7% |
| Out-of-scope tier accuracy | 93.3% |

The difficulty tiers discriminate as intended — accuracy drops on harder tiers rather than staying flat, which is evidence the dataset is actually testing different failure modes rather than one easy pattern repeated 60 times.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Agent Framework | LangGraph |
| LLM API | Groq (`openai/gpt-oss-120b`) |
| Embeddings | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) |
| Vector Store | FAISS (`IndexFlatL2`) |
| Document Parsing | `pdfplumber` |
| UI | Gradio |
| Deployment | Hugging Face Spaces |

---

## Local Setup

### Prerequisites

- Python 3.10+
- A free Groq API key from [console.groq.com](https://console.groq.com)

### 1. Clone the repository

```bash
git clone https://github.com/Abdulrahman-Hayatu/compliance-agent.git
cd compliance-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Build the FAISS index

Run the indexer **once** before starting the app. It reads both regulatory PDFs from `data/`, chunks and embeds them, and saves the artifacts to `index/`.

```bash
python -m rag.indexer
```

Expected output:

```
CBN chunks : 44
NDPC chunks: 32
Total      : 76
✓ Indexing complete.
```

>  **Do not run the indexer at app startup.** The index is pre-built and committed to the repo.

### 5. Run the app

```bash
python app.py
```

Open the local URL printed in your terminal (e.g. `http://127.0.0.1:7860`).

---

## Deployment to Hugging Face Spaces

HF Spaces does not serve large binary files (PDFs, `.faiss`, `.pkl`) directly from the repo. The solution is to host the pre-built index artifacts in a **HuggingFace Dataset repository** and have the app download them automatically at startup.

### Prerequisites

```bash
pip install huggingface_hub
```

---

### Step 1 — Authenticate

```bash
huggingface-cli login
```

Generate a **Write** access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and paste it when prompted. Also add it to your `.env`:

```env
HF_TOKEN=your_hf_write_token_here
```

---

### Step 2 — Upload the index to a HF Dataset repo

Run this once locally after building the FAISS index:

```bash
python -m rag.upload_index --repo Abdulrahman-Hayatu/compliance-index
```

This creates a public dataset repo and uploads `compliance_index.faiss` and `chunks.pkl` to it. Verify at:
`https://huggingface.co/datasets/Abdulrahman_hayatu/compliance-index`

---

### Step 3 — Create the Space

```bash
huggingface-cli repo create nigerian-fintech-compliance-agent \
  --type space \
  --space_sdk gradio
```

---

### Step 4 — Add the Space as a git remote

```bash
git remote add space https://huggingface.co/spaces/Abdulrahman-Hayatu/nigerian-fintech-compliance-agent
```

---

### Step 5 — Set Space Secrets

The app needs two secrets at runtime — never commit these to git:

```bash
# Your Groq API key
huggingface-cli secret set GROQ_API_KEY \
  --repo-type space \
  --repo Abdulrahman-Hayatu/nigerian-fintech-compliance-agent

# The HF dataset repo holding the index artifacts
huggingface-cli secret set HF_INDEX_REPO \
  --repo-type space \
  --repo Abdulrahman-Hayatu/nigerian-fintech-compliance-agent
```

When prompted for `HF_INDEX_REPO`, enter: `Abdulrahman-Hayatu/compliance-index`

---

### Step 6 — Push and deploy

```bash
git push space main
```

> If HF created an initial commit, use `git push space main --force`

At startup, the app will detect that the index is missing and automatically download it from your dataset repo before serving requests.

---

### Step 7 — Verify

Once the Space shows **Running** (3–5 min on first build), open:

```
https://huggingface.co/spaces/Abdulrahman-Hayatu/nigerian-fintech-compliance-agent
```

Upload a test policy document and confirm all 4 agent stages complete and a report is generated.

---

### Updating the deployment

```bash
git add .
git commit -m "your change description"
git push origin main   # GitHub
git push space main    # Hugging Face Spaces
```

---

## Reliability Notes

A few production-hardening details worth calling out:

- **Stable, metadata-rich chunk IDs.** Each indexed chunk carries a persistent ID (`cbn_014`, `ndpa_021`) plus its source document and clause/section number, so retrieved context can be cited and verified rather than treated as an anonymous blob of text.
- **Grounded citations.** The Checker Agent is explicitly instructed to cite only the labels it was actually shown, and to return `N/A` rather than guessing when nothing relevant was retrieved.
- **JSON-mode + bounded retries.** The Checker Agent's Groq calls use `response_format={"type": "json_object"}` and retry transient failures with backoff — except rate-limit (429) errors, which fail fast instead of burning retries and backoff delay on a call that can't succeed within the retry window.
- **No silent failures.** If an earlier pipeline step errors out, the final report says so explicitly instead of rendering an indistinguishable "0 findings" result.

---

## Project Structure

```
compliance-agent/
├── app.py                    # Gradio UI entry point
├── agents/
│   ├── __init__.py
│   ├── parser_agent.py       # Agent 1: Document Parser
│   ├── retrieval_agent.py    # Agent 2: Retrieval Agent
│   ├── checker_agent.py      # Agent 3: Compliance Checker
│   └── report_agent.py       # Agent 4: Report Generator
├── graph/
│   ├── __init__.py
│   ├── state.py              # ComplianceState TypedDict
│   └── workflow.py           # LangGraph state graph definition
├── rag/
│   ├── __init__.py
│   ├── indexer.py            # Script to build and save FAISS index (run once)
│   └── retriever.py          # FAISS query interface (singleton)
├── eval/
│   ├── golden_dataset.jsonl        # 60-example hand-reviewed eval set
│   ├── golden_dataset_review.md    # Human-readable version, grouped by tier
│   ├── metrics.py                  # Pure metric functions (P/R/F1, context recall)
│   ├── run_eval.py                 # CLI harness — --retrieval-only / --verdict
│   └── first_results.json          # Latest full verdict-pass run
├── data/
│   ├── CIRCULAR AND GUIDELINES FOR THE OPERATIONS OF AGENT BANKING IN NIGERIA OCTOBER 6 2025.pdf
│   └── Nigeria_Data_Protection_Act_2023.pdf
├── index/
│   ├── compliance_index.faiss   # Pre-built FAISS index (committed to repo)
│   └── chunks.pkl               # Serialized text chunks (committed to repo)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Author

**Abdulrahman Hayatu Usman**  
BSc Computer Science — Ahmadu Bello University, Zaria

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/abdulrahman-hayatu)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?logo=github&logoColor=white)](https://github.com/Abdulrahman-Hayatu)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?logo=gmail&logoColor=white)](mailto:hayatuusmanabdulrahman@gmail.com)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Profile-FFD21E)](https://huggingface.co/Abdulrahman-Hayatu)