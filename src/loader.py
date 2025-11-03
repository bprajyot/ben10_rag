from pathlib import Path
from typing import Any, List
from langchain_community.document_loaders import PyPDFLoader

def loader(data_path: str) -> List[Any]:
    path = Path(data_path)
    documents = []

    pdf_list = list(path.rglob("**/*.pdf"))
    try:
        for pdf in pdf_list:
            loader = PyPDFLoader(str(pdf))
            loaded = loader.load()
            documents.extend(loaded)
    except Exception as e:
        print(f"Error loading PDFs: {e}")

    return documents