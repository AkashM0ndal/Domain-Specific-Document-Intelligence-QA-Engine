from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents




class EmbeddingPipeline:

    def __init__(self, model_name: str= 'all-MiniLM-L6-v2', chunk_size: int= 1000, chunk_overlap: int= 200):
        
        self.chunk_size= chunk_size
        self.chunk_overlap= chunk_overlap
        self.model= SentenceTransformer(model_name)

        print(f'[INFO] Loaded embedding model: {model_name}')



    def chunk_documents(self, documents: List[Any]) -> List[Any]:

        # Remove any documents where page_content is None or not a valid string
        valid_docs= [
            doc for doc in documents
            if getattr(doc, 'page_content', None) and isinstance(doc.page_content, str) and doc.page_content.strip()

        ]

        splitter= RecursiveCharacterTextSplitter(
            chunk_size= self.chunk_size,
            chunk_overlap= self.chunk_overlap,
            length_function= len,
            separators= ['\n\n', '\n', ' ', '']
        )

        chunks= splitter.split_documents(valid_docs)

        # Ensure every resulting chunk also has valid string content
        chunks= [c for c in chunks if isinstance(c.page_content, str) and c.page_content.strip()]
        print(f'[INFO] Split {len(valid_docs)} documents into {len(chunks)} chunks. ')

        return chunks


    # Here we just load our chunks into embeddings
    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        
        texts = [str(chunk.page_content) for chunk in chunks]
        print(f'[INFO] Generating embeddings for {len(texts)} chunks...')
        texts= [chunk.page_content for chunk in chunks]
        print(f'[INFO] Generating embeddings for {len(texts)} chunks...')
        embeddings= self.model.encode(texts, show_progress_bar= True)
        print(f'[INFO] Embedding shape: {embeddings.shape}')

        return embeddings