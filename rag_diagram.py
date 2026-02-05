"""
RAG pipeline diagram: visual flow that highlights the active step (indexing or query phase).
Also provides layout wrappers so the UI can be structured like the diagram.
"""


def ui_node_header(label: str, sublabel: str = "") -> str:
    """HTML for a diagram-style section header (use above Streamlit widgets in each node)."""
    sub = f'<div class="ui-node-sublabel">{sublabel}</div>' if sublabel else ""
    return f'<div class="ui-diagram-node"><span class="ui-node-label">{label}</span>{sub}</div>'


def ui_arrow() -> str:
    """HTML for a vertical arrow between pipeline nodes (Documents → Chunking → ...)."""
    return '<div class="ui-diagram-arrow" aria-hidden="true">→</div>'


def rag_diagram_html(phase: str, active_index: int) -> str:
    """
    phase: "indexing" | "query"
    active_index: 0-based step index to highlight (current step).
    """
    if phase == "indexing":
        nodes = ["Documents", "Chunking", "Chunks", "Embedding", "Vector Store"]
    else:
        nodes = ["Query", "Embed", "Search", "Retrieve", "Augment", "LLM", "Response"]

    n = len(nodes)
    # -1 means no step highlighted (e.g. static diagram on Chat page)
    if active_index < 0:
        active_index = -1
    else:
        active_index = max(0, min(active_index, n - 1))
    parts = []
    for i, label in enumerate(nodes):
        active = "rag-node-active" if (active_index >= 0 and i == active_index) else ""
        parts.append(f'<span class="rag-node {active}" data-step="{i}">{label}</span>')
        if i < n - 1:
            parts.append('<span class="rag-arrow">→</span>')
    return (
        '<div class="rag-diagram-wrap">'
        f'<div class="rag-diagram rag-diagram-{phase}">'
        + "".join(parts) +
        "</div></div>"
    )


def _lh_node(label: str, active: bool, node_id: str = "") -> str:
    """LeewayHertz-style light blue node."""
    cls = "advanced-rag-node" + (" advanced-rag-node-active" if active else "")
    return f'<span class="{cls}" data-node="{node_id}">{label}</span>'


def _lh_arrow(label: str) -> str:
    """Arrow with optional label above it (white)."""
    return (
        '<span class="advanced-rag-arrow-cell">'
        f'<span class="advanced-rag-arrow-label">{label}</span>'
        '<span class="advanced-rag-arrow">→</span>'
        '</span>'
    )


def advanced_rag_diagram_html(phase: str = "both", active_index: int = -1) -> str:
    """
    LeewayHertz-style Advanced RAG diagram: two panels (Indexing + Query).
    phase: "indexing" | "query" | "both"
    active_index: same as rag_diagram_html (indexing 0-5, query 0-6); -1 = none highlighted.
    """
    # Indexing nodes: 0 Documents, 1 Chunks, 2 Embedding Model, 3 Vector Store, 4 Vector Store
    # Map caller steps (0,2,3,5) -> 0,1,2,3,4
    if phase == "indexing" and active_index >= 0:
        idx_active = 0 if active_index <= 1 else 1 if active_index == 2 else 2 if active_index == 3 else 4 if active_index >= 4 else 3
    else:
        idx_active = -1
    if phase == "query" and active_index >= 0:
        q_active = min(active_index + 1, 7)  # 0->1 Query, 1->2 Embed, ..., 6->7 Response
    else:
        q_active = -1
    if phase == "both":
        idx_active = -1
        q_active = -1

    def index_node(i: int, label: str) -> str:
        return _lh_node(label, idx_active == i, f"idx-{i}")

    def query_node(i: int, label: str) -> str:
        return _lh_node(label, q_active == i, f"q-{i}")

    indexing_row = (
        index_node(0, "Documents")
        + _lh_arrow("Chunking")
        + index_node(1, "Chunks")
        + _lh_arrow("")
        + index_node(2, "Embedding Model")
        + _lh_arrow("Vectorize")
        + index_node(3, "Vector Store")
        + _lh_arrow("Indexing")
        + index_node(4, "Vector Store")
    )
    # Query row: User → Query → Embedding Model → Vectorize → Search → Vector Store → Retrieve → Relevant Contexts → Augment → Prompt → LLM → Generate → Response
    query_row = (
        query_node(0, "User")
        + _lh_arrow("Query")
        + query_node(1, "Query")
        + _lh_arrow("")
        + query_node(2, "Embedding Model")
        + _lh_arrow("Vectorize")
        + _lh_arrow("Search")
        + query_node(3, "Vector Store")
        + _lh_arrow("Retrieve")
        + query_node(4, "Relevant Contexts")
        + _lh_arrow("Augment")
        + query_node(5, "Prompt")
        + _lh_arrow("")
        + query_node(6, "LLM")
        + _lh_arrow("Generate")
        + query_node(7, "Response")
    )

    return (
        '<div class="advanced-rag-wrap">'
        '<div class="advanced-rag-section">'
        '<div class="advanced-rag-section-title">Indexing</div>'
        f'<div class="advanced-rag-flow">{indexing_row}</div>'
        '</div>'
        '<div class="advanced-rag-section">'
        '<div class="advanced-rag-section-title">User Query</div>'
        f'<div class="advanced-rag-flow">{query_row}</div>'
        '</div>'
        '</div>'
    )
