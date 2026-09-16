from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_documents(docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    split_docs = []

    for doc in docs:
        texts = splitter.split_text(doc.page_content)
        for i, t in enumerate(texts):
            meta = dict(doc.metadata) if doc.metadata else {}
            meta.update({"chunk_index": i})
            split_docs.append(Document(page_content = t, metadata = meta))
    return split_docs