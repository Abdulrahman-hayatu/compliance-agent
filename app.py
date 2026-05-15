"""
app.py

Gradio UI entry point for the Nigerian Fintech Regulatory Compliance Agent.
"""

from __future__ import annotations

import os
import tempfile
import logging

import gradio as gr
import pdfplumber
from dotenv import load_dotenv

from rag.hub_loader import ensure_index_available

# Download index artifacts from HF Hub if running on Spaces
ensure_index_available()

from graph.workflow import compliance_graph
from agents.parser_agent import parser_agent
from agents.retrieval_agent import retrieval_agent
from agents.checker_agent import checker_agent
from agents.report_agent import report_agent

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Text extraction function for PDF and TXT files, with error handling for unsupported formats and empty extractions.
def extract_text(file_path: str) -> str:
    """
    Extract text from a PDF or TXT file.
    Returns stripped text or raises ValueError if extraction yields nothing.
    """
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        text = "\n".join(pages).strip()

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()

    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Please upload a PDF or TXT file."
        )

    if not text:
        raise ValueError(
            "No text could be extracted from the uploaded file. "
            "If it is a scanned PDF, OCR is required before uploading."
        )

    return text


# Pipeline runner function that executes the four agents in sequence, yielding status updates and the final report for the Gradio UI.
def run_compliance_check(file):
    """
    Generator function that runs the four-agent pipeline step by step,
    yielding (status, report) tuples so Gradio can update the UI in real time.
    """

    # Guard: no file uploaded — yield a message and exit early
    if file is None:
        yield "Please upload a document before running.", ""
        return

    # Extract text from the uploaded file, yielding an error message if extraction fails
    try:
        extracted_text = extract_text(file.name)
    except Exception as exc:
        yield f"File extraction error: {exc}", ""
        return

    # Initialise state with the extracted text and empty placeholders for all subsequent data
    state = {
        "uploaded_text":     extracted_text,
        "policy_claims":     [],
        "retrieved_chunks":  {},
        "compliance_results": [],
        "final_report":      "",
        "status":            "Starting compliance check...",
    }

    try:
        # Agent 1: Parser 
        yield "Parsing document claims...", ""
        state = parser_agent(state)
        yield state["status"], ""

        # Agent 2: Retrieval 
        yield "Retrieving relevant regulations...", ""
        state = retrieval_agent(state)
        yield state["status"], ""

        # Agent 3: Checker 
        yield "Checking compliance against CBN and NDPC regulations...", ""
        state = checker_agent(state)
        yield state["status"], ""

        # Agent 4: Reporter 
        yield "Generating compliance report...", ""
        state = report_agent(state)

        yield "Done.", state["final_report"]

    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        yield f"Pipeline error: {exc}", ""


# Gradio UI 

with gr.Blocks(title="Nigerian Fintech Compliance Agent") as demo:

    gr.Markdown(
        """
        # 🇳🇬 Nigerian Fintech Regulatory Compliance Agent
        Upload a fintech policy document (PDF or TXT) and the agent pipeline will
        assess it against the **CBN Agent Banking Guidelines (Oct 2025)** and the
        **Nigeria Data Protection Act 2023**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload your fintech policy document",
                file_types=[".pdf", ".txt"],
            )
            run_btn = gr.Button("Run Compliance Check", variant="primary")

    with gr.Row():
        status_box = gr.Textbox(
            label="Agent Status",
            interactive=False,
            placeholder="Status updates will appear here...",
        )

    with gr.Row():
        report_output = gr.Markdown(
            label="Compliance Report",
            value="*Your compliance report will appear here after the check completes.*",
        )

    run_btn.click(
        fn=run_compliance_check,
        inputs=[file_input],
        outputs=[status_box, report_output],
    )

if __name__ == "__main__":
    demo.launch()