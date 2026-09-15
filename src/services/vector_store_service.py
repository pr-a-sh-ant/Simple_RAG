import chromadb
from .embedding_service import EmbeddingService
from uuid import uuid4
from rank_bm25 import BM25Okapi
from fastembed.rerank.cross_encoder import TextCrossEncoder

class VectorStoreService:
    def __init__(self, persist_directory="./chroma_db"):
        self.embedder = EmbeddingService()
        

        self.bm25 = None
        self.reranker = TextCrossEncoder(model_name="BAAI/bge-reranker-base")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="documents")
        self._build_bm25()

    def _build_bm25(self):
        data = self.collection.get()
        self.corpus_id = data['ids']
        self.corpus_texts = data['documents']
        self.corpus_metas = data['metadatas']
        if self.corpus_texts:
            self.bm25 = BM25Okapi([text.lower().split() for text in self.corpus_texts])

    def add_documents(self, chunks):
        texts = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [str(uuid4()) for _ in chunks]
        embeddings = self.embedder.embed_documents(texts)
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        self._build_bm25()


    def search(self, query, k=5, candidate_n=20, mode="rerank", where=None):
        query_embedding = self.embedder.embed_query(query)
        vres = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(candidate_n, len(self.corpus_id)),
            where=where
        )

        v_ids = vres["ids"][0]

        # vector-only: pure similarity ordering, no BM25, no rerank
        if mode == "vector" or self.bm25 is None:
            return list(zip(vres["documents"][0], vres["metadatas"][0], vres["distances"][0]))[:k]

        # get scores for all documents in the corpus
        scores = self.bm25.get_scores(query.lower().split())
        # sort the scores and get the top k chunks
        bm25_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidate_n]
        # map indices back to the chunk ids
        bm25_ids = [self.corpus_id[i] for i in bm25_idx]

        # Reciprocal rank fusion
        rrf = {}
        for rank, doc_id in enumerate(v_ids):
            rrf[doc_id] = rrf.get(doc_id, 0) + 1 / (rank + 60)
        for rank, doc_id in enumerate(bm25_ids):
            rrf[doc_id] = rrf.get(doc_id, 0) + 1 / (rank + 60)
        # pick the top 
        top = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:candidate_n]
        idx = {id_: i for i, id_ in enumerate(self.corpus_id)}

        if mode == "hybrid":
            top = top[:k]
            return [(self.corpus_texts[idx[_id]], self.corpus_metas[idx[_id]], score) for _id, score in top]

        cand_ids   = [doc_id for doc_id, _ in top]
        cand_texts = [self.corpus_texts[idx[_id]] for _id in cand_ids]
        ce_scores= list(self.reranker.rerank(query, cand_texts))
        ordered =  sorted(zip(cand_ids, ce_scores), key=lambda x: x[1], reverse=True)
        ordered = ordered[:k]

        return [(self.corpus_texts[idx[_id]], self.corpus_metas[idx[_id]], score) for _id, score in ordered]