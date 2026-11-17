from src.data_loader import load_all_documnets
from src.embedding import EmbeddingPipeline


## Example usage

if __name__ == '__main__':
    docs= load_all_documnets('Data')
    chunks= EmbeddingPipeline().chunk_documents(docs)
    chunkvectors= EmbeddingPipeline().embeded_chunks(chunks)

    print(chunkvectors)