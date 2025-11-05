from src.loader import loader
from src.vectorstore import VectorStore
from src.search import Search
import os

# Example usage
if __name__ == "__main__":
    if not os.path.exists("faiss_store/faiss_store.index"):
        print("Loading Documents...")
        docs = loader("data")
        print(f"Loaded {len(docs)} documents.")
        print("Building Vector Store...")
        store = VectorStore("faiss_store")
    else:
    #store.build_from_documents(docs)
        store = VectorStore("faiss_store")
        store.load()
        print("Vector Store loaded!")
    #print(store.query("What is attention mechanism?", top_k=3))
    rag_search = Search()
    os.system('cls')
    while True:
        query = input("Enter your Query: ")
        summary = rag_search.search_and_summarize(query, top_k=3)
        print("Answer: \n", summary)
        print("\n\n")