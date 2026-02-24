"""Knowledge Graph construction"""
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import networkx as nx
import ollama


class GraphBuilder:
    def __init__(self):
        self.graph_dir = Path("data") / "graphs"
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.graph_file = self.graph_dir / "knowledge_graph.json"
        self.G = nx.DiGraph()
        self._load_graph()

    def _load_graph(self):
        if self.graph_file.exists():
            try:
                with open(self.graph_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.G = nx.node_link_graph(data)
            except Exception:
                self.G = nx.DiGraph()

    def _save_graph(self):
        data = nx.node_link_data(self.G)
        with open(self.graph_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def extract_entities_phi4(self, text: str, doc_name: str) -> Dict:
        """Extract entities using Phi-4"""
        prompt = f"""Extract entities and relationships from this text:
        
Text: {text[:1500]}

Extract:
1. People/Organizations
2. Key concepts/traits
3. Numerical scores/metrics

Return JSON:
{{
  "entities": [
    {{"name": "Entity Name", "type": "Person|Organization|Concept|Metric"}}
  ],
  "relationships": [
    {{"source": "Entity1", "target": "Entity2", "relation": "description"}}
  ]
}}

JSON:"""

        try:
            response = ollama.generate(model="phi4:latest", prompt=prompt, options={"temperature": 0.1})
            content = response['response']
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except Exception as e:
            print(f"Error: {e}")

        return {"entities": [], "relationships": []}

    def add_document_to_graph(self, doc_name: str, chunks: List[Dict]):
        """Add document chunks to graph"""
        doc_id = f"doc_{hashlib.md5(doc_name.encode()).hexdigest()[:8]}"
        self.G.add_node(doc_id, type="Document", name=doc_name)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_text = chunk.get("text", "")[:200]
            self.G.add_node(chunk_id, type="Chunk", text=chunk_text)
            self.G.add_edge(doc_id, chunk_id, relation="contains")

            extraction = self.extract_entities_phi4(chunk.get("text", ""), doc_name)

            entity_map = {}
            for ent in extraction.get("entities", []):
                ent_name = ent.get("name", "")
                if not ent_name:
                    continue
                ent_id = f"{ent.get('type', 'Entity')}:{ent_name}".replace(" ", "_")
                if ent_id not in self.G:
                    self.G.add_node(ent_id, type=ent.get("type", "Entity"), name=ent_name)
                entity_map[ent_name] = ent_id
                self.G.add_edge(chunk_id, ent_id, relation="mentions")

            for rel in extraction.get("relationships", []):
                src = entity_map.get(rel.get("source", ""))
                tgt = entity_map.get(rel.get("target", ""))
                if src and tgt:
                    self.G.add_edge(src, tgt, relation=rel.get("relation", "related"))

        self._save_graph()

    def query_graph(self, query_entities: List[str], depth: int = 2) -> Dict:
        """Query the graph"""
        found = []
        for node in self.G.nodes():
            name = str(self.G.nodes[node].get("name", "")).lower()
            for q in query_entities:
                if q.lower() in name:
                    found.append(node)
                    break

        if not found:
            return {"nodes": [], "edges": [], "relationships": [], "entity_count": 0}

        subgraph_nodes = set(found)
        current = set(found)

        for _ in range(depth - 1):
            next_level = set()
            for node in current:
                neighbors = set(self.G.successors(node)) | set(self.G.predecessors(node))
                next_level.update(neighbors)
            subgraph_nodes.update(next_level)
            current = next_level

        subG = self.G.subgraph(subgraph_nodes)

        relationships = []
        for u, v, data in subG.edges(data=True):
            u_name = subG.nodes[u].get("name", u)
            v_name = subG.nodes[v].get("name", v)
            rel = data.get("relation", "related")
            relationships.append(f"{u_name} --[{rel}]--> {v_name}")

        return {
            "nodes": list(subgraph_nodes),
            "relationships": relationships,
            "entity_count": len([n for n in subgraph_nodes if not str(n).startswith("doc_")])
        }

    def extract_query_entities(self, query: str) -> List[str]:
        """Extract entities from query"""
        words = query.replace("?", "").replace(".", "").split()
        return [w for w in words if len(w) > 3]

    def visualize_graph(self) -> Dict:
        """Get graph data for visualization"""
        nodes = [
            {"id": n, "label": self.G.nodes[n].get("name", self.G.nodes[n].get("text", n)),
             "type": self.G.nodes[n].get("type", "Unknown")}
            for n in self.G.nodes()
        ]
        edges = [
            {"source": u, "target": v, "label": d.get("relation", "")}
            for u, v, d in self.G.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}

    def get_stats(self) -> Dict:
        return {"nodes": self.G.number_of_nodes(), "edges": self.G.number_of_edges()}

    def clear(self):
        self.G = nx.DiGraph()
        if self.graph_file.exists():
            self.graph_file.unlink()
        self._save_graph()
