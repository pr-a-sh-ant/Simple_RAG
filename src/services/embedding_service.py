from fastembed import TextEmbedding
import dotenv

dotenv.load_dotenv()


class EmbeddingService:
    def __init__(self):
        self.client = TextEmbedding(model_name='BAAI/bge-base-en')

    def embed_documents(self, texts):
        embeddings_generator = self.client.embed(texts)
        return list(embeddings_generator)
    
    def embed_query(self, text):
        result = next(self.client.embed([text]))
        return result.tolist()
      