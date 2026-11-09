"""LangGraph declares control flow; stages own their implementations."""
from .state import InvestigationState


def build_graph(nodes, checkpointer=None):
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(InvestigationState)
    for name in ["route", "retrieve", "answer", "extract", "propose", "validate",
                 "plan_retrieval", "repair", "simulate", "analyze", "inspect", "report"]:
        graph.add_node(name, getattr(nodes, name))
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", lambda s: "inspect" if s["action"] == "inspect"
                                else "retrieve", {"inspect": "inspect", "retrieve": "retrieve"})
    graph.add_conditional_edges("retrieve", lambda s: s["action"], {
        "question": "answer", "simulate": "extract"
    })
    graph.add_conditional_edges("answer", lambda s: s["decision"], {
        "done": "report", "missing": "plan_retrieval", "retry": "answer", "unverified": "report"
    })
    graph.add_conditional_edges("extract", lambda s: s["decision"], {
        "verified": "propose", "retry": "extract", "unverified": "report"
    })
    graph.add_edge("propose", "validate")

    def after_validation(s):
        if s["decision"] == "ready":
            return "simulate"
        if s["decision"] == "missing":
            return "plan_retrieval"
        if s.get("blueprint") is not None:
            # A supplied deterministic plan cannot improve by recompiling it.
            return "report"
        return "repair" if s["repairs"] < nodes.settings.max_repairs else "report"

    graph.add_conditional_edges("validate", after_validation, {
        n: n for n in ["simulate", "plan_retrieval", "repair", "report"]
    })
    graph.add_conditional_edges("plan_retrieval", lambda s: s["decision"], {
        "search": "retrieve", "stop": "report"
    })
    graph.add_edge("repair", "propose")
    graph.add_conditional_edges("simulate", lambda s: s["decision"], {
        "done": "analyze", "stop": "report"
    })
    graph.add_edge("analyze", "report")
    graph.add_conditional_edges("inspect", lambda s: s["decision"], {
        "done": "report", "retry": "inspect", "unverified": "report"
    })
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)
