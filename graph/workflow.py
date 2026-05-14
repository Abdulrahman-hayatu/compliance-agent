"""
graph/workflow.py
Defines the LangGraph workflow for the compliance pipeline, connecting the four agents in sequence:
1. parser_agent: extracts policy claims from the uploaded document
2. retrieval_agent: fetches relevant regulatory chunks for each claim
3. checker_agent: assesses the compliance of each claim against the retrieved chunks
4. report_agent: generates a formatted markdown compliance report based on the checker results.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from graph.state import ComplianceState
from agents.parser_agent import parser_agent
from agents.retrieval_agent import retrieval_agent
from agents.checker_agent import checker_agent
from agents.report_agent import report_agent


def build_graph():
    graph = StateGraph(ComplianceState)
    graph.add_node("parser",    parser_agent)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("checker",   checker_agent)
    graph.add_node("reporter",  report_agent)
    graph.set_entry_point("parser")
    graph.add_edge("parser",    "retrieval")
    graph.add_edge("retrieval", "checker")
    graph.add_edge("checker",   "reporter")
    graph.add_edge("reporter",  END)
    return graph.compile()


compliance_graph = build_graph()