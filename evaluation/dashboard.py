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
from dataclasses import dataclass, field

import streamlit as st


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
        return {
            "query": self.query,
            "response": self.response,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "rerank_latency_ms": self.rerank_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "num_sources": self.num_sources,
            "content_types": self.content_types,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryMetrics":
        return cls(
            query=data.get("query", ""),
            response=data.get("response", ""),
            faithfulness=data.get("faithfulness"),
            answer_relevancy=data.get("answer_relevancy"),
            context_precision=data.get("context_precision"),
            context_recall=data.get("context_recall"),
            retrieval_latency_ms=data.get("retrieval_latency_ms", 0.0),
            generation_latency_ms=data.get("generation_latency_ms", 0.0),
            rerank_latency_ms=data.get("rerank_latency_ms", 0.0),
            total_latency_ms=data.get("total_latency_ms", 0.0),
            num_sources=data.get("num_sources", 0),
            content_types=data.get("content_types", []),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )


@dataclass
class EvaluationSession:
    """A collection of query metrics from an evaluation session."""
    session_id: str
    name: str
    metrics: List[QueryMetrics] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_metrics(self, query_metrics: QueryMetrics) -> None:
        self.metrics.append(query_metrics)
    
    @property
    def avg_faithfulness(self) -> Optional[float]:
        values = [m.faithfulness for m in self.metrics if m.faithfulness is not None]
        return sum(values) / len(values) if values else None
    
    @property
    def avg_answer_relevancy(self) -> Optional[float]:
        values = [m.answer_relevancy for m in self.metrics if m.answer_relevancy is not None]
        return sum(values) / len(values) if values else None
    
    @property
    def avg_context_precision(self) -> Optional[float]:
        values = [m.context_precision for m in self.metrics if m.context_precision is not None]
        return sum(values) / len(values) if values else None
    
    @property
    def avg_retrieval_latency(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.retrieval_latency_ms for m in self.metrics) / len(self.metrics)
    
    @property
    def avg_total_latency(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.total_latency_ms for m in self.metrics) / len(self.metrics)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "metrics": [m.to_dict() for m in self.metrics],
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationSession":
        session = cls(
            session_id=data.get("session_id", ""),
            name=data.get("name", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
        )
        for m_data in data.get("metrics", []):
            session.add_metrics(QueryMetrics.from_dict(m_data))
        return session


class MetricsStore:
    """Persistent storage for evaluation metrics."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/evaluation_metrics.json")
        self.sessions: Dict[str, EvaluationSession] = {}
        self._load()
    
    def _load(self) -> None:
        """Load metrics from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    for session_data in data.get("sessions", []):
                        session = EvaluationSession.from_dict(session_data)
                        self.sessions[session.session_id] = session
            except (json.JSONDecodeError, KeyError):
                self.sessions = {}
    
    def _save(self) -> None:
        """Save metrics to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            data = {
                "sessions": [s.to_dict() for s in self.sessions.values()]
            }
            json.dump(data, f, indent=2)
    
    def create_session(self, name: str, metadata: Optional[Dict] = None) -> EvaluationSession:
        """Create a new evaluation session."""
        session_id = f"session_{int(time.time())}_{len(self.sessions)}"
        session = EvaluationSession(
            session_id=session_id,
            name=name,
            metadata=metadata or {},
        )
        self.sessions[session_id] = session
        self._save()
        return session
    
    def add_metrics(self, session_id: str, metrics: QueryMetrics) -> None:
        """Add metrics to a session."""
        if session_id in self.sessions:
            self.sessions[session_id].add_metrics(metrics)
            self._save()
    
    def get_session(self, session_id: str) -> Optional[EvaluationSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)
    
    def get_all_sessions(self) -> List[EvaluationSession]:
        """Get all sessions."""
        return list(self.sessions.values())
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save()


def render_metric_card(label: str, value: Optional[float], suffix: str = "") -> None:
    """Render a metric card with label and value."""
    if value is not None:
        st.metric(label=label, value=f"{value:.3f}{suffix}")
    else:
        st.metric(label=label, value="N/A")


def render_aggregate_metrics(session: EvaluationSession) -> None:
    """Render aggregate metrics for a session."""
    st.subheader("📊 Aggregate Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_card("Faithfulness", session.avg_faithfulness)
    
    with col2:
        render_metric_card("Answer Relevancy", session.avg_answer_relevancy)
    
    with col3:
        render_metric_card("Context Precision", session.avg_context_precision)
    
    with col4:
        st.metric(
            label="Total Queries",
            value=len(session.metrics),
        )


def render_latency_metrics(session: EvaluationSession) -> None:
    """Render latency metrics for a session."""
    st.subheader("⏱️ Latency Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Avg Retrieval Latency",
            value=f"{session.avg_retrieval_latency:.1f} ms",
        )
    
    with col2:
        avg_gen = sum(m.generation_latency_ms for m in session.metrics) / len(session.metrics) if session.metrics else 0
        st.metric(
            label="Avg Generation Latency",
            value=f"{avg_gen:.1f} ms",
        )
    
    with col3:
        st.metric(
            label="Avg Total Latency",
            value=f"{session.avg_total_latency:.1f} ms",
        )


def render_metrics_chart(session: EvaluationSession) -> None:
    """Render metrics chart over queries."""
    st.subheader("📈 Metrics Over Queries")
    
    if not session.metrics:
        st.info("No metrics data available.")
        return
    
    # Prepare data for chart
    chart_data = {
        "Query #": list(range(1, len(session.metrics) + 1)),
        "Faithfulness": [m.faithfulness or 0 for m in session.metrics],
        "Answer Relevancy": [m.answer_relevancy or 0 for m in session.metrics],
        "Context Precision": [m.context_precision or 0 for m in session.metrics],
    }
    
    import pandas as pd
    df = pd.DataFrame(chart_data)
    df_melted = df.melt(id_vars=["Query #"], var_name="Metric", value_name="Score")
    
    st.line_chart(df.set_index("Query #")[["Faithfulness", "Answer Relevancy", "Context Precision"]])


def render_latency_chart(session: EvaluationSession) -> None:
    """Render latency chart over queries."""
    st.subheader("⏱️ Latency Over Queries")
    
    if not session.metrics:
        st.info("No latency data available.")
        return
    
    import pandas as pd
    
    chart_data = {
        "Query #": list(range(1, len(session.metrics) + 1)),
        "Retrieval (ms)": [m.retrieval_latency_ms for m in session.metrics],
        "Reranking (ms)": [m.rerank_latency_ms for m in session.metrics],
        "Generation (ms)": [m.generation_latency_ms for m in session.metrics],
    }
    
    df = pd.DataFrame(chart_data)
    st.bar_chart(df.set_index("Query #"))


def render_query_breakdown(session: EvaluationSession) -> None:
    """Render per-query breakdown table."""
    st.subheader("🔍 Per-Query Breakdown")
    
    if not session.metrics:
        st.info("No query data available.")
        return
    
    import pandas as pd
    
    rows = []
    for i, m in enumerate(session.metrics, 1):
        rows.append({
            "#": i,
            "Query": m.query[:50] + "..." if len(m.query) > 50 else m.query,
            "Faith.": f"{m.faithfulness:.3f}" if m.faithfulness else "N/A",
            "Relev.": f"{m.answer_relevancy:.3f}" if m.answer_relevancy else "N/A",
            "Prec.": f"{m.context_precision:.3f}" if m.context_precision else "N/A",
            "Latency": f"{m.total_latency_ms:.0f}ms",
            "Sources": m.num_sources,
        })
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_content_type_distribution(session: EvaluationSession) -> None:
    """Render content type distribution chart."""
    st.subheader("📁 Content Type Distribution")
    
    if not session.metrics:
        st.info("No content type data available.")
        return
    
    # Count content types
    type_counts = {"text": 0, "table": 0, "image_summary": 0}
    for m in session.metrics:
        for ct in m.content_types:
            if ct in type_counts:
                type_counts[ct] += 1
    
    if sum(type_counts.values()) == 0:
        st.info("No content type data available.")
        return
    
    import pandas as pd
    
    df = pd.DataFrame({
        "Content Type": list(type_counts.keys()),
        "Count": list(type_counts.values()),
    })
    
    st.bar_chart(df.set_index("Content Type"))


def render_session_selector(store: MetricsStore) -> Optional[EvaluationSession]:
    """Render session selector and return selected session."""
    sessions = store.get_all_sessions()
    
    if not sessions:
        st.info("No evaluation sessions found. Run an evaluation to see metrics.")
        return None
    
    session_options = {s.name: s.session_id for s in sessions}
    selected_name = st.selectbox(
        "Select Evaluation Session",
        options=list(session_options.keys()),
    )
    
    if selected_name:
        return store.get_session(session_options[selected_name])
    return None


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
