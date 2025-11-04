import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding_creater import CreateEmbeddings

class VectorStore:
    def __init__(self, dir: str="../vector_store", embedding_model: str="all-MiniLM-L6-v2", chunk_size: int=1000, overlap: int=200):
        self.persist_dir = dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(self.embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = overlap

    def build_vectorstore(self, documents: List[Any]):
        embd = CreateEmbeddings(self.embedding_model, self.chunk_size, self.chunk_overlap)
        chunks = embd.create_chunks(documents)
        embeddings = embd.create_embeddings(chunks)
        metadatas = [{"text":chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any]=None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)

    def save(self):
        path = os.path.join(self.persist_dir, "faiss_store.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss_store.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)


if __name__ == "__main__":
    from src.loader import loader
    docs = loader("data")
    store = VectorStore("faiss_store")
    store.build_vectorstore(docs)
    store.load()
    print(store.query("Tell me about Armodrillo", top_k=3))