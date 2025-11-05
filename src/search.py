import os
from dotenv import load_dotenv
from src.vectorstore import VectorStore
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
load_dotenv()

class Search:
    def __init__(self, dir: str="../faiss_store", embedding_model: str="all-MiniLM-L6-v2", llm: str="gemini-2.0-flash"):
        self.vectorstore = VectorStore(dir=dir, embedding_model=embedding_model)
        faiss_path = os.path.join(dir, "faiss_store.index")
        metadata_path = os.path.join(dir, "metadata.pkl")
        if not os.path.exists(faiss_path) or not os.path.exists(metadata_path):
            from src.loader import loader
            docs = loader("data")
            self.vectorstore.build_vectorstore(docs)
        else:
            self.vectorstore.load()
        google_api_key = ""
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)       

    def search_and_summarize(self, query: str, top_k: int=5):
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text","") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant information found"
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content