# Custom RAG — Retrieval-Augmented Generation over a document corpus

A from-scratch RAG system that ingests documents, retrieves relevant passages, and answers
questions grounded in those passages. Built end-to-end on a **$0 / fully local-or-free stack**
(no paid APIs) and improved through measured retrieval phases, each evaluated against a held-out
gold question set.

## Stack (all free)

| Layer | Choice | Notes |
|---|---|---|
| Embeddings | **fastembed** (`BAAI/bge-base-en`, 768-dim) | Local ONNX, no PyTorch |
| Vector store | **ChromaDB** (`PersistentClient`) | On-disk, local |
| Keyword search | **rank_bm25** (`BM25Okapi`) | Lexical retrieval |
| Reranker | **fastembed `TextCrossEncoder`** (`BAAI/bge-reranker-base`) | Local ONNX cross-encoder |
| Generation / judging | **Groq** free API (`llama-3.1-8b-instant`) | Answer generation + LLM-as-judge |
| API | **FastAPI** + static chat UI | `/upload` and `/ask` endpoints |

## Pipeline

```
ingest → recursive chunking → embed → ChromaDB
                                          │
query ─► vector search ─┐                 │
        BM25 search   ──┼─► RRF fusion ─► cross-encoder rerank ─► top-k ─► prompt ─► LLM answer
                        ┘  (candidate_n=20)        (→ k)
```

Retrieval is a **two-stage retrieve-then-rerank** design: hybrid search casts a wide net
(20 candidates via Reciprocal Rank Fusion of vector + BM25), then a cross-encoder re-scores each
`(query, chunk)` pair precisely and trims to the final `k`.

## Evaluation

All numbers below are produced by `src/eval/run_eval.py` against `gold_questions.json` — **59 questions**
(55 answerable, 4 deliberately out-of-scope) over a **91-chunk corpus**: a 9-document fictional
"TechCorp" universe **plus 3 "archived/superseded" distractor documents**.

Metrics:
- **Recall@k** — deterministic. A question is a hit if any retrieved chunk in the top-k contains all of
  its relevant keywords (whitespace-normalized substring match).
- **MRR** — Mean Reciprocal Rank; rewards getting the right chunk to rank 1, not just into the top-k.
- **Judge score (/5)** — LLM-as-judge rates answer quality against the expected answer.

### The distractor design (why ranking matters)

A clean corpus saturates: every retriever hits ~1.0 recall@5 and the ordering improvements from
reranking are invisible. To make ordering *matter*, the corpus includes **3 archived policy documents**
whose chunks are lexically near-identical to the current ones but carry **old values** (e.g. archived
401(k) match = *50% up to 3%*; current = *100% up to 4%*). A set of **12 "temporal-disambiguation"
questions** asks for the *current* value. Keywords are chosen so only the current chunk counts as a hit
— the archived chunk is pure retrieval bait. To rank correctly, the system must tell "current policy"
from "former policy," which is exactly what a cross-encoder reranker is good at and what lexical/dense
retrieval is not.

### Phase 2 results — retrieval quality (91-chunk corpus, 55 answerable questions)

**All answerable questions (n=55)**

| Metric | Vector-only | Hybrid (RRF) | Hybrid + Rerank |
|---|---|---|---|
| **Recall@1** | 0.764 | 0.727 | **0.818** |
| Recall@5 | 0.982 | 1.000 | 0.982 |
| **MRR** | 0.870 | 0.856 | **0.897** |

**Hard temporal-disambiguation subset (n=12)**

| Metric | Vector-only | Hybrid (RRF) | Hybrid + Rerank |
|---|---|---|---|
| **Recall@1** | 0.417 | 0.333 | **0.583** |
| **MRR** | 0.708 | 0.653 | **0.792** |

**What moved and why:**
- **Recall@5 is saturated and misleading.** With easy questions every method finds the right chunk
  *somewhere* in the top 5. The real differentiator is **Recall@1 / MRR** — getting it to rank 1.
- **Hybrid (BM25) *hurts* on hard questions** (Recall@1 0.333 < vector's 0.417). Keyword matching is
  fooled by the archived chunks, which share almost all the same words as the current ones, and pulls
  them up the ranking.
- **Reranking rescues it.** Swapping in the stronger `BAAI/bge-reranker-base` cross-encoder, which reads
  the "former / superseded" semantics, **nearly doubles hard-question Recall@1 (0.333 → 0.583)** and is
  the best method overall (0.818 Recall@1, 0.897 MRR).
- **Model choice matters.** The small `ms-marco-MiniLM-L-6-v2` reranker did *not* beat hybrid on this
  task (hard Recall@1 stayed 0.333); only the larger `bge-reranker-base` distinguished current from
  archived. This was found by measurement, not assumed.

### A note on the judge score

LLM-as-judge answer quality is essentially flat across the three retrievers (vector 4.16, hybrid 4.04,
rerank 4.13 out of 5). This is expected, not a null result: with `k=5`, **all three put the correct
chunk somewhere in the context window**, so the generator answers well regardless of *rank order within
those 5*. Ranking quality only converts into answer quality when `k` is small enough that the top-ranked
chunk is the one the model actually reads. The takeaway: **judge@k=5 cannot see ranking improvements —
that is exactly why Recall@1 / MRR are the right metrics for evaluating a reranker.**

## Reproduce

```bash
# install
./venv/bin/pip install -r requirements.txt

# (one-time) ingest a document
./venv/bin/python -m src.eval.run_eval --ingest path/to/file.pdf

# run the eval in any retrieval mode
./venv/bin/python -m src.eval.run_eval --retriever vector  --label vector
./venv/bin/python -m src.eval.run_eval --retriever hybrid  --label hybrid
./venv/bin/python -m src.eval.run_eval --retriever rerank  --label rerank
```

Results are written to `src/eval/results/<label>_<date>.json`.

## Run the app

```bash
./venv/bin/uvicorn src.main:app --reload
```

Open `http://localhost:8000` for the chat UI, or POST to `/upload` (file) and `/ask` (question).
