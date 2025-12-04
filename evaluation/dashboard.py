"""
Evaluation dashboard for DocuLens.

This module provides a Streamlit-based dashboard for visualizing
RAGAS evaluation metrics, retrieval latency, and per-query breakdown.

Excellence Track Feature: Evaluation dashboard showing retrieval metrics and latency.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

import streamlit as st

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query: str
    response: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    num_sources: int = 0
    content_types: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryMetrics":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EvaluationSession:
    """A collection of query metrics from an evaluation session."""
    session_id: str
    name: str
    metrics: List[QueryMetrics] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_metrics(self, m: QueryMetrics) -> None:
        self.metrics.append(m)
    
    def _avg(self, attr: str) -> Optional[float]:
        vals = [getattr(m, attr) for m in self.metrics if getattr(m, attr) is not None]
        return sum(vals) / len(vals) if vals else None
    
    @property
    def avg_faithfulness(self) -> Optional[float]:
        return self._avg("faithfulness")
    
    @property
    def avg_answer_relevancy(self) -> Optional[float]:
        return self._avg("answer_relevancy")
    
    @property
    def avg_context_precision(self) -> Optional[float]:
        return self._avg("context_precision")
    
    @property
    def avg_retrieval_latency(self) -> float:
        return sum(m.retrieval_latency_ms for m in self.metrics) / len(self.metrics) if self.metrics else 0.0
    
    @property
    def avg_total_latency(self) -> float:
        return sum(m.total_latency_ms for m in self.metrics) / len(self.metrics) if self.metrics else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {"session_id": self.session_id, "name": self.name,
                "metrics": [m.to_dict() for m in self.metrics],
                "created_at": self.created_at, "metadata": self.metadata}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationSession":
        session = cls(session_id=data.get("session_id", ""), name=data.get("name", ""),
                      created_at=data.get("created_at", datetime.now().isoformat()),
                      metadata=data.get("metadata", {}))
        for m in data.get("metrics", []):
            session.add_metrics(QueryMetrics.from_dict(m))
        return session


class MetricsStore:
    """Persistent storage for evaluation metrics."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/evaluation_metrics.json")
        self.sessions: Dict[str, EvaluationSession] = {}
        logger.info(f"MetricsStore initialized: {self.storage_path}")
        self._load()
    
    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                for s in data.get("sessions", []):
                    session = EvaluationSession.from_dict(s)
                    self.sessions[session.session_id] = session
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load metrics: {e}")
    
    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({"sessions": [s.to_dict() for s in self.sessions.values()]}, f, indent=2)
    
    def create_session(self, name: str, metadata: Optional[Dict] = None) -> EvaluationSession:
        session_id = f"session_{int(time.time())}_{len(self.sessions)}"
        session = EvaluationSession(session_id=session_id, name=name, metadata=metadata or {})
        self.sessions[session_id] = session
        self._save()
        return session
    
    def add_metrics(self, session_id: str, metrics: QueryMetrics) -> None:
        if session_id in self.sessions:
            self.sessions[session_id].add_metrics(metrics)
            self._save()
    
    def get_session(self, session_id: str) -> Optional[EvaluationSession]:
        return self.sessions.get(session_id)
    
    def get_all_sessions(self) -> List[EvaluationSession]:
        return list(self.sessions.values())
    
    def delete_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save()


def render_metric_card(label: str, value: Optional[float], suffix: str = "") -> None:
    st.metric(label=label, value=f"{value:.3f}{suffix}" if value is not None else "N/A")


def render_aggregate_metrics(session: EvaluationSession) -> None:
    st.subheader("📊 Aggregate Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_metric_card("Faithfulness", session.avg_faithfulness)
    with c2: render_metric_card("Answer Relevancy", session.avg_answer_relevancy)
    with c3: render_metric_card("Context Precision", session.avg_context_precision)
    with c4: st.metric(label="Total Queries", value=len(session.metrics))


def render_latency_metrics(session: EvaluationSession) -> None:
    st.subheader("⏱️ Latency Metrics")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric(label="Avg Retrieval", value=f"{session.avg_retrieval_latency:.1f} ms")
    avg_gen = sum(m.generation_latency_ms for m in session.metrics) / len(session.metrics) if session.metrics else 0
    with c2: st.metric(label="Avg Generation", value=f"{avg_gen:.1f} ms")
    with c3: st.metric(label="Avg Total", value=f"{session.avg_total_latency:.1f} ms")


def render_metrics_chart(session: EvaluationSession) -> None:
    st.subheader("📈 Metrics Over Queries")
    if not session.metrics:
        st.info("No metrics data available.")
        return
    import pandas as pd
    df = pd.DataFrame({
        "Query #": list(range(1, len(session.metrics) + 1)),
        "Faithfulness": [m.faithfulness or 0 for m in session.metrics],
        "Answer Relevancy": [m.answer_relevancy or 0 for m in session.metrics],
        "Context Precision": [m.context_precision or 0 for m in session.metrics],
    })
    st.line_chart(df.set_index("Query #")[["Faithfulness", "Answer Relevancy", "Context Precision"]])


def render_latency_chart(session: EvaluationSession) -> None:
    st.subheader("⏱️ Latency Over Queries")
    if not session.metrics:
        st.info("No latency data available.")
        return
    import pandas as pd
    df = pd.DataFrame({
        "Query #": list(range(1, len(session.metrics) + 1)),
        "Retrieval (ms)": [m.retrieval_latency_ms for m in session.metrics],
        "Reranking (ms)": [m.rerank_latency_ms for m in session.metrics],
        "Generation (ms)": [m.generation_latency_ms for m in session.metrics],
    })
    st.bar_chart(df.set_index("Query #"))


def render_query_breakdown(session: EvaluationSession) -> None:
    st.subheader("🔍 Per-Query Breakdown")
    if not session.metrics:
        st.info("No query data available.")
        return
    import pandas as pd
    rows = [{
        "#": i, "Query": m.query[:50] + "..." if len(m.query) > 50 else m.query,
        "Faith.": f"{m.faithfulness:.3f}" if m.faithfulness else "N/A",
        "Relev.": f"{m.answer_relevancy:.3f}" if m.answer_relevancy else "N/A",
        "Prec.": f"{m.context_precision:.3f}" if m.context_precision else "N/A",
        "Latency": f"{m.total_latency_ms:.0f}ms", "Sources": m.num_sources,
    } for i, m in enumerate(session.metrics, 1)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_content_type_distribution(session: EvaluationSession) -> None:
    st.subheader("📁 Content Type Distribution")
    if not session.metrics:
        st.info("No content type data available.")
        return
    type_counts = {"text": 0, "table": 0, "image_summary": 0}
    for m in session.metrics:
        for ct in m.content_types:
            if ct in type_counts:
                type_counts[ct] += 1
    if sum(type_counts.values()) == 0:
        st.info("No content type data available.")
        return
    import pandas as pd
    st.bar_chart(pd.DataFrame({"Content Type": list(type_counts.keys()), "Count": list(type_counts.values())}).set_index("Content Type"))


def render_session_selector(store: MetricsStore) -> Optional[EvaluationSession]:
    sessions = store.get_all_sessions()
    if not sessions:
        st.info("No evaluation sessions found. Run an evaluation to see metrics.")
        return None
    options = {s.name: s.session_id for s in sessions}
    selected = st.selectbox("Select Evaluation Session", options=list(options.keys()))
    return store.get_session(options[selected]) if selected else None


def render_dashboard_page():
    """Main dashboard page renderer."""
    st.set_page_config(
        page_title="DocuLens Evaluation Dashboard",
        page_icon="📊",
        layout="wide",
    )
    
    st.title("📊 DocuLens Evaluation Dashboard")
    st.markdown("Monitor RAGAS metrics, retrieval latency, and per-query performance.")
    
    # Initialize metrics store
    store = MetricsStore()
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Sessions")
        
        # Session management
        if st.button("🔄 Refresh"):
            st.rerun()
        
        # Import/Export
        st.markdown("---")
        st.subheader("Import/Export")
        
        uploaded_file = st.file_uploader("Import Session (JSON)", type=["json"])
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                session = EvaluationSession.from_dict(data)
                store.sessions[session.session_id] = session
                store._save()
                st.success(f"Imported session: {session.name}")
            except Exception as e:
                st.error(f"Failed to import: {e}")
    
    # Main content
    session = render_session_selector(store)
    
    if session:
        st.markdown("---")
        
        # Top-level metrics
        render_aggregate_metrics(session)
        
        st.markdown("---")
        
        # Latency metrics
        render_latency_metrics(session)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            render_metrics_chart(session)
        
        with col2:
            render_latency_chart(session)
        
        st.markdown("---")
        
        # Content type distribution
        render_content_type_distribution(session)
        
        st.markdown("---")
        
        # Query breakdown
        render_query_breakdown(session)
        
        # Export option
        st.markdown("---")
        st.subheader("📥 Export Session")
        
        export_data = json.dumps(session.to_dict(), indent=2)
        st.download_button(
            label="Download Session JSON",
            data=export_data,
            file_name=f"{session.name.replace(' ', '_')}_metrics.json",
            mime="application/json",
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "DocuLens Evaluation Dashboard | Excellence Track Feature"
        "</div>",
        unsafe_allow_html=True,
    )


def create_demo_session(store: MetricsStore) -> EvaluationSession:
    """Create a demo session with sample data for testing."""
    import random
    
    session = store.create_session(
        name=f"Demo Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        metadata={"type": "demo"},
    )
    
    queries = [
        "What is the main topic of this document?",
        "What are the key financial metrics?",
        "Describe the trends shown in the charts.",
        "What are the main conclusions?",
        "How do the tables support the findings?",
    ]
    
    for i, query in enumerate(queries):
        metrics = QueryMetrics(
            query=query,
            response=f"Sample response for query {i+1}...",
            faithfulness=random.uniform(0.7, 1.0),
            answer_relevancy=random.uniform(0.6, 1.0),
            context_precision=random.uniform(0.5, 0.9),
            retrieval_latency_ms=random.uniform(50, 200),
            generation_latency_ms=random.uniform(500, 2000),
            rerank_latency_ms=random.uniform(10, 50),
            total_latency_ms=random.uniform(600, 2500),
            num_sources=random.randint(3, 7),
            content_types=random.choices(["text", "table", "image_summary"], k=random.randint(2, 5)),
        )
        store.add_metrics(session.session_id, metrics)
    
    return session


# Main entry point
if __name__ == "__main__":
    render_dashboard_page()
