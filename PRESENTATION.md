# Custom RAG — Presentation & Interview Prep

A complete walkthrough of the project: what it is, how every piece works, every design
decision and *why*, the evaluation methodology, the results, and the questions an interviewer
will ask (with answers). Read top-to-bottom to be able to defend any part of it.

---

## 1. The 30-second pitch

> "I built a Retrieval-Augmented Generation system from scratch on a completely free stack — no paid
> APIs. It ingests documents, retrieves relevant passages with a hybrid vector + keyword search plus a
> cross-encoder reranker, and answers questions grounded in those passages. The part I'm proudest of is
> the evaluation: I built a gold question set with adversarial 'distractor' documents, and the metrics
> caught a counterintuitive failure — hybrid search actually *hurt* accuracy on near-duplicate documents
> — which I diagnosed and fixed by upgrading the reranker model. So it's not just a pipeline; it's a
> measured, debugged pipeline."

That paragraph is the whole interview. Everything below lets you defend every sentence in it.

---

## 2. What is RAG, and why?

**The problem:** An LLM only knows what was in its training data. Ask it about *your* company handbook
and it will either say "I don't know" or — worse — confidently make something up (hallucinate).

**The fix (RAG):** Instead of relying on the model's memory, you (1) **retrieve** the most relevant
passages from your own documents and (2) **stuff them into the prompt** as context, then ask the model
to answer *using only that context*. The model becomes a reasoning/wording engine over facts you supply.

**The pipeline:**

```
INGESTION (offline, once per document)
  load → chunk → embed → store in vector DB
                                  │
QUERY TIME (per question)         ▼
  question ─► retrieve relevant chunks ─► build prompt (context + question) ─► LLM ─► grounded answer
```

**Why it matters:** RAG is how you give an LLM private, current, or domain-specific knowledge **without
retraining it**. It's cheaper, updatable (just change the documents), and far less prone to hallucination
because the answer is anchored to retrieved text.

---

## 3. The architecture, component by component

| Stage | Tool | What it does | Why this choice |
|---|---|---|---|
| **Load** | `pypdf`, `python-docx`, `BeautifulSoup`, `pytesseract` | Extract raw text from PDF/DOCX/HTML/images | Cover the common document formats; OCR handles scanned files |
| **Chunk** | custom `TextSplitter` | Split text into ~500-char overlapping pieces on natural boundaries | Embeddings work on passages, not whole docs; boundaries preserve meaning |
| **Embed** | `fastembed` (`BAAI/bge-base-en`, 768-dim) | Turn text into vectors | Local ONNX, **no PyTorch**, free; bge is a strong open embedding model |
| **Store** | `ChromaDB` (`PersistentClient`) | Save vectors + text + metadata on disk | Free, local, zero-setup persistence |
| **Retrieve** | ChromaDB vector search + `rank_bm25` + RRF + cross-encoder | Find and rank the best chunks | Hybrid + rerank — the heart of the project (see §5) |
| **Generate** | `Groq` API (`llama-3.1-8b-instant`) | Write the final answer from context | Groq's free tier is fast; the model is good enough for grounded Q&A |
| **Serve** | `FastAPI` + static HTML chat UI | `/upload` and `/ask` endpoints | Lightweight, standard Python API |

**The $0 constraint drove every choice.** Embeddings and reranking run locally via ONNX (no GPU, no
torch, no per-call cost); only generation uses a hosted API, and Groq's free tier covers it. This is a
talking point: *"I deliberately engineered the whole thing to cost nothing, which forced local-first
choices like fastembed instead of OpenAI embeddings."*

---

## 4. Key concepts you must be able to explain

### 4.1 Embeddings & semantic (vector) search
An **embedding** maps text to a vector (here, 768 numbers) such that similar *meanings* land close
together in space. To search, you embed the query and find the chunks whose vectors are nearest (cosine
distance). This catches **paraphrases** — "how much vacation do I get?" matches a chunk about "paid time
off" even with no shared words.

**Weakness:** it can *miss exact tokens*. Specific strings — a product name, "Tuesdays", an error code —
get "smeared" into the general meaning, so dense search sometimes ranks the wrong passage first.

### 4.2 BM25 / keyword search
**BM25** is classic lexical search: it scores a chunk by how often the query's *exact words* appear in it
(with weighting so rare words matter more and long documents don't win just by being long). It's the
opposite tradeoff from vectors: **great at exact terms, blind to synonyms.**

### 4.3 Hybrid search + Reciprocal Rank Fusion (RRF)
Run both retrievers and **fuse their rankings**. RRF is the fusion rule:

```
score(chunk) = Σ  1 / (k + rank_in_that_list)      # k = 60, a standard constant
```

A chunk near the top of *either* list gets a high score; chunks near the top of *both* win. RRF fuses by
**rank, not raw score**, so you don't have to normalize a cosine distance against a BM25 score (different
scales) — you just use their positions. The `60` damps the difference between rank 1 and rank 2 so no
single list dominates.

### 4.4 Bi-encoder vs. cross-encoder (the reranker)
- A **bi-encoder** (the embedding model) encodes the query and each chunk *separately*, then compares
  vectors. Fast (you can pre-compute all chunk vectors) but **lossy** — the query and chunk never "see"
  each other.
- A **cross-encoder** takes `(query, chunk)` **together** as one input and outputs a single relevance
  score. Far more accurate because it can attend to the interaction between the two — but **slow**, so you
  can't run it over the whole corpus.

**Retrieve-then-rerank** uses both: cheap hybrid search casts a wide net (top ~20 candidates), then the
expensive cross-encoder precisely re-scores only those 20 and trims to the final 5. Best of both.

### 4.5 The evaluation metrics
- **Recall@k** — did *any* retrieved chunk in the top-k contain the answer? Measures "is it in there at
  all." Deterministic.
- **Recall@1 / MRR (Mean Reciprocal Rank)** — did the *right* chunk land at the *top*? MRR = average of
  `1/rank` of the first correct chunk. These measure **ordering**, which is what a reranker improves.
- **LLM-as-judge** — a second LLM call grades the final answer 1–5 against a reference answer (correctness
  + faithfulness). Measures answer quality, not retrieval.
- **Hallucination rate** — on out-of-scope questions, how often the model invents an answer instead of
  refusing. Measures grounding/safety.

---

## 5. The phases (the story of the project)

### Phase 1 — Baseline RAG
Naive character-splitting → vector-only search → generate. It worked, but I had no idea *how well*, so I
built the evaluation harness and a gold question set first. **Lesson: you can't improve what you don't
measure.**

### Phase 2 — Retrieval quality (the core work)
Three measured improvements, each compared against the previous baseline:

1. **Recursive/boundary-aware chunking** — split on paragraph → line → sentence → word, with overlap, so
   chunks don't cut mid-sentence and lose meaning.
2. **Hybrid search (vector + BM25 + RRF)** — add keyword search to catch the exact-token cases dense
   search misses.
3. **Cross-encoder reranking** — add a precise second-stage re-scorer.

### Phase 2.5 — The distractor experiment (the most important part)
**Problem I hit:** on a clean corpus, *every* method scored ~1.0 recall@5. The improvements were
invisible — the task was too easy. A flat result tells you nothing and is a weak resume story.

**What I did:** I made the task *realistically harder*. I added 3 "archived/superseded" policy documents
whose chunks are **lexically almost identical** to the current ones but carry **old values** (archived
401(k) match = 50%/3%; current = 100%/4%). Then 12 "temporal-disambiguation" questions that ask for the
*current* value. To answer correctly, the system must tell "current policy" from "former policy."

**What the metrics revealed (the payoff):**

| Hard questions (n=12) | Recall@1 | MRR |
|---|---|---|
| Vector-only | 0.417 | 0.708 |
| **Hybrid (RRF)** | **0.333** | 0.653 |
| Hybrid + Rerank (`bge-reranker-base`) | **0.583** | **0.792** |

- **Hybrid made it *worse*** — BM25 got fooled by the archived chunks (same words!) and pulled them up.
  This is counterintuitive and exactly the kind of thing only measurement catches.
- **Reranking fixed it**, but **only after a model upgrade**: the small `MiniLM-L-6` reranker also stayed
  at 0.333; the stronger `bge-reranker-base` (which actually reads "former/superseded" semantics) nearly
  **doubled hard-question Recall@1**.

This is the centerpiece. It demonstrates measurement, a real bug found, root-cause diagnosis, and a fix
validated by re-measuring.

### Phase 3 — (deliberately not done) and why
Generation-side work (citations, faithfulness). I chose to **stop and make the existing story tight**
rather than add breadth. *Talking point: depth I can defend beats a checklist of buzzwords I can't.*

---

## 6. Final results table (91-chunk corpus, 55 answerable questions)

| Metric | Vector | Hybrid | Hybrid + Rerank |
|---|---|---|---|
| Recall@1 (all) | 0.764 | 0.727 | **0.818** |
| Recall@5 (all) | 0.982 | 1.000 | 0.982 |
| MRR (all) | 0.870 | 0.856 | **0.897** |
| Recall@1 (hard) | 0.417 | 0.333 | **0.583** |
| MRR (hard) | 0.708 | 0.653 | **0.792** |
| Judge /5 | 4.16 | 4.04 | 4.13 |
| Hallucination % | 0 | 0 | 0 |

**Reading it:** Rerank wins on the metrics that measure *ranking* (Recall@1, MRR). Judge is flat (~4.1)
because at k=5 all methods get the answer chunk into the context, so the generator answers fine regardless
of internal rank order — **judge@k=5 literally cannot see a ranking improvement**, which is precisely why
Recall@1/MRR are the correct metrics for a reranker.

---

## 7. Interview Q&A (rehearse these)

**Q: What is RAG and why use it instead of fine-tuning?**
A: RAG injects retrieved facts into the prompt at query time; fine-tuning bakes knowledge into weights.
RAG is cheaper, instantly updatable (change documents, not the model), traceable to a source, and less
hallucination-prone. Fine-tuning is for changing *behavior/style*, not for adding *facts*.

**Q: Why hybrid search instead of just vector search?**
A: Vector search captures meaning/paraphrase but smears exact tokens (names, dates, codes). BM25 nails
exact tokens but is blind to synonyms. Fusing them with RRF covers both failure modes. I have a concrete
example — question q21 needed the literal tokens "Tuesdays"/"Thursdays"; vector search missed it, BM25
recovered it.

**Q: What's RRF and why rank-based fusion?**
A: Reciprocal Rank Fusion scores each chunk as Σ 1/(60+rank) across the two result lists. It fuses by
*position*, so I don't have to reconcile a cosine distance and a BM25 score that live on different scales.

**Q: Bi-encoder vs cross-encoder?**
A: Bi-encoder embeds query and document separately — fast, pre-computable, used for first-stage retrieval
over the whole corpus. Cross-encoder scores the pair jointly — slow but accurate, used to rerank a small
candidate set. Two-stage retrieve-then-rerank gets speed and accuracy.

**Q: How do you know any of this actually helped? / How did you evaluate?**
A: A gold set of 59 questions with known-relevant keywords. Recall@k and MRR are deterministic;
LLM-as-judge grades answer quality. Crucially, I added adversarial distractor documents so the metrics
could *separate* the methods — on an easy corpus everything scored ~1.0.

**Q: Tell me about a bug or surprising result.**
A: Hybrid search *lowered* accuracy on near-duplicate documents — BM25 ranked outdated "archived" chunks
highly because they shared the same words as the current ones. Recall@1 dropped from 0.42 to 0.33. I fixed
it with a stronger cross-encoder reranker that reads the "former/superseded" semantics, which nearly
doubled hard-question Recall@1 to 0.58.

**Q: Why did the judge score barely move?**
A: At k=5 every retriever puts the right chunk somewhere in the context, so the generator answers well
regardless of rank order — judge@k=5 can't see ranking. That's *why* I report Recall@1/MRR. At a small k
the ranking gain would convert to an answer-quality gain.

**Q: How do you stop it hallucinating?**
A: Three layers: (1) ground answers in retrieved context via the prompt; (2) a hallucination metric on
out-of-scope questions that checks the model refuses instead of inventing; (3) the LLM-judge penalizes
unsupported claims. The honest next step (a planned Phase 3) is inline citations so every claim is
traceable to a chunk.

**Q: Why these specific tools / why local?**
A: A hard $0 constraint. fastembed runs embeddings and reranking locally via ONNX (no torch, no GPU, no
per-call cost); ChromaDB persists locally; only generation hits a free hosted API (Groq). It forced
local-first engineering.

**Q: How would you chunk differently / what about chunk size?**
A: Recursive splitting on paragraph→line→sentence→word with overlap preserves semantic boundaries. Size is
a tradeoff: too big dilutes the embedding and wastes context; too small fragments facts across chunks
(which is exactly why cross-document synthesis questions are hard). 500 chars with 50 overlap worked for
this corpus; I'd tune it per corpus by measuring recall.

**Q: What are the limitations / what would you do next?**
A: (1) Recall metric is keyword-substring based — robust but coarse; a semantic-match metric would be more
rigorous. (2) Judge and hallucination metrics use a hosted LLM, so they wobble slightly run-to-run. (3)
Single-turn only — no conversation history. (4) No inline citations yet. Next I'd add grounded citations +
a faithfulness metric, then conversational retrieval.

---

## 8. Limitations (state these proactively — it reads as maturity)

- **Synthetic corpus.** The documents are fictional ("TechCorp"), so there's no copyright issue, but it
  isn't real-world messy data.
- **Keyword-based recall.** Deterministic and bug-resistant, but can't credit a semantically-correct chunk
  that uses different words than the keywords. (I normalize whitespace to avoid a class of false misses —
  a bug I found and fixed: PDF line-wraps split keyword phrases.)
- **Hosted-LLM nondeterminism.** Recall/MRR are exact; judge and hallucination scores can drift ±a little
  even at temperature 0.
- **Small out-of-scope set (n=4)** makes the hallucination rate noisy (each question = 25%).
- **Single-turn, no citations** — the two clearest next steps.

---

## 9. How to demo it live

```bash
# 1. Run the eval (shows the numbers)
./venv/bin/python -m src.eval.run_eval --retriever vector --label vector
./venv/bin/python -m src.eval.run_eval --retriever hybrid --label hybrid
./venv/bin/python -m src.eval.run_eval --retriever rerank --label rerank

# 2. Run the app + chat UI
./venv/bin/uvicorn src.main:app --reload   # → http://localhost:8000
```

**Demo script:** upload the handbook, ask "how many sick days do I get?" (clean hit), then ask a
temporal-disambiguation question like "what is the *current* 401(k) match?" and show that it returns
100%/4% (current) and not the archived 50%/3% — the reranker doing its job, live.

---

## 10. One-sentence summaries to memorize

- **RAG:** retrieve relevant passages, put them in the prompt, answer from them — gives an LLM private
  knowledge without retraining.
- **Hybrid search:** vectors for meaning + BM25 for exact words, fused by rank (RRF).
- **Reranking:** a slow-but-accurate cross-encoder re-scores the top ~20 candidates and reorders them.
- **The headline:** I built adversarial distractors that exposed hybrid search *backfiring*, then fixed
  ranking with a stronger reranker — found by measurement, not assumption.
- **Why it's $0:** embeddings + reranking run locally via ONNX; only generation uses a free hosted API.
