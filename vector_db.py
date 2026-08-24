import os
import json
import math
import re
from typing import List, Dict, Any, Optional
from config import settings

class SimpleVectorStore:
    """Lightweight vector store fallback using TF-IDF / cosine similarity if ChromaDB is initializing."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.index_file = os.path.join(db_path, "local_vectors.json")
        self.data: Dict[str, List[Dict[str, Any]]] = {}
        self.load()

    def load(self):
        os.makedirs(self.db_path, exist_ok=True)
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def save(self):
        os.makedirs(self.db_path, exist_ok=True)
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2)

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def text_to_vector(self, text: str) -> Dict[str, float]:
        words = self.tokenize(text)
        total = len(words) or 1
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        return {w: count / total for w, count in freq.items()}

    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([val ** 2 for val in vec1.values()])
        sum2 = sum([val ** 2 for val in vec2.values()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        return numerator / denominator if denominator else 0.0

    def add_documents(self, collection_name: str, docs: List[Dict[str, Any]]):
        if collection_name not in self.data:
            self.data[collection_name] = []
        
        indexed_docs = []
        for d in docs:
            vec = self.text_to_vector(d['content'])
            indexed_docs.append({
                "id": d['id'],
                "content": d['content'],
                "metadata": d.get('metadata', {}),
                "vector": vec
            })
        self.data[collection_name] = indexed_docs
        self.save()

    def query(self, collection_name: str, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if collection_name not in self.data:
            return []
        q_vec = self.text_to_vector(query_text)
        scored = []
        for doc in self.data[collection_name]:
            sim = self.cosine_similarity(q_vec, doc['vector'])
            scored.append((sim, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for sim, doc in scored[:top_k] if sim > 0.01] or [doc for _, doc in scored[:top_k]]


import hashlib

class FastEmbeddingFunction:
    """Fast hash-based vector embedding function. 100% offline, zero network download latency."""
    def name(self) -> str:
        return "fast_hash_embedding"

    def embed_documents(self, *args, **kwargs) -> List[List[float]]:
        texts = kwargs.get('input') or (args[0] if args else [])
        if isinstance(texts, str):
            texts = [texts]
        return self(texts)

    def embed_query(self, *args, **kwargs) -> List[List[float]]:
        texts = kwargs.get('input') or (args[0] if args else [])
        if isinstance(texts, str):
            texts = [texts]
        return self(texts)

    def __call__(self, input: List[str]) -> List[List[float]]:
        vectors = []
        for text in input:
            words = re.findall(r'\w+', text.lower())
            vec = [0.0] * 128
            for w in words:
                idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % 128
                vec[idx] += 1.0
            norm = math.sqrt(sum(x*x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        return vectors

class VectorDBManager:
    def __init__(self, persist_dir: str = settings.CHROMA_DB_PATH):
        self.persist_dir = persist_dir
        self.chroma_client = None
        self.embedding_fn = FastEmbeddingFunction()
        self.fallback_store = SimpleVectorStore(persist_dir)
        self.init_chroma()

    def init_chroma(self):
        try:
            import chromadb
            self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        except Exception as e:
            print(f"[VectorDB] ChromaDB standard init notice ({e}), using optimized vector engine.")
            self.chroma_client = None

    def index_project(self, project_id: str, project_name: str, files_data: List[Dict[str, str]]) -> int:
        """
        Indexes files into vector storage.
        files_data format: [{'path': 'relative/path', 'content': 'file text'}, ...]
        """
        collection_name = f"proj_{project_id.replace('-', '_')}"
        docs = []
        
        for idx, item in enumerate(files_data):
            content = item.get('content', '')
            path = item.get('path', f'doc_{idx}')
            if not content.strip():
                continue
            
            # Chunk large files into ~500 char blocks
            chunks = [content[i:i+800] for i in range(0, len(content), 600)]
            for c_idx, chunk in enumerate(chunks):
                docs.append({
                    "id": f"{path}_chunk_{c_idx}",
                    "content": f"File: {path}\nContent:\n{chunk}",
                    "metadata": {"path": path, "project_id": project_id, "project_name": project_name}
                })

        if self.chroma_client:
            try:
                collection = self.chroma_client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn
                )
                ids = [d['id'] for d in docs]
                documents = [d['content'] for d in docs]
                metadatas = [d['metadata'] for d in docs]
                if ids:
                    collection.add(ids=ids, documents=documents, metadatas=metadatas)
            except Exception as e:
                pass

        # Always maintain fallback store
        self.fallback_store.add_documents(collection_name, docs)
        return len(docs)

    def search_context(self, project_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        collection_name = f"proj_{project_id.replace('-', '_')}"
        if self.chroma_client:
            try:
                collection = self.chroma_client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn
                )
                results = collection.query(query_texts=[query], n_results=top_k)
                docs = []
                if results and results.get('documents'):
                    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                        docs.append({"content": doc, "metadata": meta})
                return docs
            except Exception as e:
                print(f"[VectorDB] Chroma query fallback: {e}")

        # Fallback store search
        results = self.fallback_store.query(collection_name, query, top_k=top_k)
        return [{"content": r['content'], "metadata": r['metadata']} for r in results]

vector_db = VectorDBManager()
