from langchain.vectorstores import VectorStore


class VectorSimilarityTopicClassiffier:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def classify(self, text: str) -> str:
        return self.vector_store.similarity_search(text, k=1)[0].metadata['topic']
