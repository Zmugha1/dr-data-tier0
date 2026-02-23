"""Steps 6-8: Entity extraction and graph construction."""

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    import networkx as nx
except ImportError:
    nx = None
try:
    import ollama
except ImportError:
    ollama = None


class KnowledgeGraphBuilder:
    """Steps 6-8: Entity extraction and graph construction."""

    def __init__(self, model: str = "phi4:latest"):
        if nx is None:
            raise ImportError("networkx required. Run: pip install networkx")
        self.model = model
        self.graph_path = Path("data/graph_store/knowledge_graph.pkl")
        self.graph_dir = Path("data/graph_store")
        self.graph_dir.mkdir(parents=True, exist_ok=True)

        if self.graph_path.exists():
            with open(self.graph_path, "rb") as f:
                self.G = pickle.load(f)
        else:
            self.G = nx.DiGraph()

    def add_document(self, doc_data: Dict[str, Any]) -> None:
        """Step 6: Entity extraction using local LLM."""
        doc_hash = doc_data["hash"]
        filename = doc_data["filename"]

        self.G.add_node(
            f"DOC:{doc_hash}",
            type="Document",
            name=filename,
            hash=doc_hash,
        )

        sample_chunks = doc_data["chunks"][:3]
        sample_text = "\n".join([c["text"] for c in sample_chunks])
        if len(sample_text) > 1000:
            sample_text = sample_text[:1000] + "..."

        entities = self._extract_entities(sample_text, filename)

        for ent in entities:
            ent_id = f"ENT:{ent['type']}:{ent['name'].replace(' ', '_')}"

            if ent_id not in self.G:
                self.G.add_node(
                    ent_id,
                    type=ent["type"],
                    name=ent["name"],
                    sources=[],
                )

            self.G.add_edge(
                f"DOC:{doc_hash}",
                ent_id,
                relation="CONTAINS",
                chunk_index=0,
            )

            for rel in ent.get("relations", []):
                target_id = f"ENT:{rel['target_type']}:{rel['target'].replace(' ', '_')}"
                if target_id not in self.G:
                    self.G.add_node(target_id, type=rel["target_type"], name=rel["target"])
                self.G.add_edge(
                    ent_id,
                    target_id,
                    relation=rel["type"],
                    weight=rel.get("weight", 1.0),
                )

    def _extract_entities(self, text: str, context: str) -> List[Dict]:
        """Step 6: Structured entity extraction with Phi-4."""
        prompt = f"""Analyze this business document excerpt and extract entities.
Document: {context}
Text: {text[:800]}

Return ONLY a JSON array of entities:
[
  {{
    "name": "Entity Name",
    "type": "Vendor|Product|Amount|Date|Location|Person|Organization",
    "relations": [
      {{
        "target": "Related Entity",
        "target_type": "Type",
        "type": "purchased_from|located_in|amount_of|dated|works_for",
        "weight": 1.0
      }}
    ]
  }}
]

JSON:"""

        try:
            if ollama:
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options={"temperature": 0, "seed": 42},
                )
                content = response.get("response", "")
                match = re.search(r"\[.*\]", content, re.DOTALL)
                if match:
                    return json.loads(match.group())
        except Exception:
            pass

        return self._fallback_extraction(text)

    def _fallback_extraction(self, text: str) -> List[Dict]:
        """Deterministic fallback using regex patterns."""
        entities = []
        amounts = re.findall(r"\$\d[\d,]*\.?\d*", text)
        for amt in amounts[:3]:
            entities.append({"name": amt, "type": "Amount", "relations": []})
        dates = re.findall(r"\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}", text)
        for date in dates[:3]:
            entities.append({"name": date, "type": "Date", "relations": []})
        return entities

    def query_graph(self, entity_name: str, depth: int = 2) -> Dict[str, Any]:
        """Retrieve subgraph for GraphRAG."""
        matches = [
            n
            for n in self.G.nodes()
            if entity_name.lower()
            in str(self.G.nodes[n].get("name", "")).lower()
        ]
        if not matches:
            return {"nodes": [], "edges": []}

        subgraph_nodes = set()
        for start in matches[:1]:
            subgraph_nodes.add(start)
            current = {start}
            for _ in range(depth):
                next_level = set()
                for node in current:
                    neighbors = set(self.G.neighbors(node)) | set(
                        self.G.predecessors(node)
                    )
                    next_level.update(neighbors)
                subgraph_nodes.update(next_level)
                current = next_level

        subG = self.G.subgraph(subgraph_nodes)
        return {
            "nodes": [{"id": n, **subG.nodes[n]} for n in subG.nodes()],
            "edges": [{"source": u, "target": v, **subG.edges[u, v]} for u, v in subG.edges()],
            "centrality": nx.degree_centrality(subG),
        }

    def persist(self) -> Dict[str, Any]:
        """Step 8: Validation and export."""
        orphans = [n for n in self.G.nodes() if self.G.degree(n) == 0]
        with open(self.graph_path, "wb") as f:
            pickle.dump(self.G, f)
        try:
            nx.write_gexf(self.G, self.graph_dir / "knowledge_graph.gexf")
        except Exception:
            pass
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "orphans": len(orphans),
        }
