# Interview Q&A — Custom RAG Project

Questions you'll get, model answers, **and how to actually deliver each one**. The delivery
matters as much as the content — a correct answer said badly sounds memorized; a correct answer
*structured* well sounds like you understand it.

---

## Part 0 — How to answer (read this first)

### The 4-beat structure for any technical question
Answer in this order. It works for almost every question below.

1. **Direct answer first** (1 sentence). Don't warm up. State the conclusion.
2. **The "why"** (1–2 sentences). The mechanism or reason.
3. **A concrete example from *your* project** (1 sentence). This is what separates you from someone who
   read a blog post. Always pull it back to something you actually built and measured.
4. **The tradeoff / limit** (1 sentence). Showing you know when it *doesn't* apply signals seniority.

> Example applied to "Why hybrid search?":
> *(1)* "Because vector and keyword search fail in opposite ways, and fusing them covers both."
> *(2)* "Vectors capture meaning but smear exact tokens; BM25 nails exact tokens but misses synonyms."
> *(3)* "In my project, question q21 needed the literal word 'Tuesdays' — vector search missed it, BM25
> recovered it."
> *(4)* "The cost is you now maintain two indexes and a fusion step, so I'd only add it if the corpus has
> exact-token queries."

### Seven delivery rules
- **Lead with the answer, not the journey.** "It's X, because…" not "So there are a few things to
  consider…"
- **Use your own numbers.** "Recall@1 went from 0.33 to 0.58" is worth ten generic sentences.
- **Name the tradeoff yourself** before they ask. It pre-empts the follow-up and shows judgment.
- **Draw the pipeline.** If there's a whiteboard, sketch `load → chunk → embed → store → retrieve →
  rerank → generate`. Talking over a diagram is far stronger than talking into the air.
- **It's fine to pause.** Two seconds of thinking beats five seconds of "um." Silence reads as
  thoughtful.
- **If you don't know, say so — then reason.** "I haven't measured that, but here's how I'd find out…"
  Never bluff; interviewers test for it.
- **Stop talking when you've answered.** Over-explaining buries the good answer and invites a poke into
  something you're shaky on.

### Behavioral questions: use STAR
**S**ituation → **T**ask → **A**ction → **R**esult. Keep S and T to one sentence each; spend your time
on **A** (what *you* did) and **R** (the measured outcome). Always end on a number.

---

## Part 1 — RAG fundamentals

**Q1. What is RAG?**
> **Answer:** "Retrieval-Augmented Generation. Instead of relying on the model's training memory, you
> retrieve relevant passages from your own documents and put them in the prompt, then ask the model to
> answer using only that context. The model becomes a reasoning engine over facts you supply."
>
> **Delivery:** This is the warm-up — say it crisply and confidently, it sets the tone. Don't ramble.
> If there's a board, draw the pipeline as you say it.

**Q2. RAG vs. fine-tuning — when each?**
> **Answer:** "RAG adds *facts* at query time; fine-tuning changes *behavior or style* by updating
> weights. Use RAG for knowledge that's private, changes often, or needs to be traceable to a source —
> it's cheaper and instantly updatable. Use fine-tuning to change *how* the model responds, not *what* it
> knows."
>
> **Delivery:** The crisp contrast "facts vs. behavior" is the whole answer — say it as a clean
> opposition. Tradeoff to add: "Fine-tuning for facts is expensive and goes stale the moment the data
> changes."

**Q3. Walk me through your pipeline.**
> **Answer:** "Two halves. Offline ingestion: load the document, chunk it on natural boundaries, embed
> each chunk, store vectors and text in ChromaDB. Query time: embed the question, retrieve candidates
> with hybrid search, rerank them with a cross-encoder, put the top chunks in the prompt, and have the
> LLM generate a grounded answer."
>
> **Delivery:** **Draw it.** Narrate left to right. Splitting it into "offline" vs "query time" shows you
> understand what's pre-computed vs. per-request — a detail that reads as real experience.

---

## Part 2 — Retrieval

**Q4. How do embeddings / vector search work?**
> **Answer:** "An embedding maps text to a vector so that similar *meanings* sit close together. To
> search, I embed the query and find the nearest chunk vectors by cosine distance. The win is
> paraphrase-matching — 'time off' matches a chunk about 'PTO' with no shared words. The weakness is it
> smears exact tokens, so specific strings can rank wrong."
>
> **Delivery:** End on the weakness — it's the natural setup for "so I added BM25," which makes your next
> answer feel inevitable rather than bolted on.

**Q5. What's BM25 and why add it?**
> **Answer:** "BM25 is classic keyword search — it scores chunks by exact word overlap, weighting rare
> words higher. It's the opposite tradeoff from embeddings: great at exact terms, blind to synonyms. I
> added it because dense search alone missed exact-token questions."
>
> **Delivery:** Frame it as *complementary*, not better. "Opposite failure modes" is the phrase to land.

**Q6. What's RRF and why fuse by rank instead of score?**
> **Answer:** "Reciprocal Rank Fusion. Each chunk scores Σ 1/(60+rank) summed across the two result
> lists, so anything near the top of either list ranks well, and chunks near the top of both win. I fuse
> by *rank*, not raw score, because a cosine distance and a BM25 score live on totally different scales —
> ranks are directly comparable. The 60 constant damps the gap between rank 1 and 2 so neither retriever
> dominates."
>
> **Delivery:** The "different scales" point is the insight they're checking for — say it explicitly.
> If pushed on the 60: "It's the standard constant from the original RRF paper; larger = flatter
> weighting across ranks."

**Q7. Give a concrete example where hybrid beat vector-only.**
> **Answer:** "Question q21 needed the literal tokens 'Tuesdays' and 'Thursdays'. Vector search smeared
> the day-of-week tokens and missed it; BM25 matched them exactly and recovered it. Recall@5 went from
> 0.977 to 1.0 just from that one question."
>
> **Delivery:** Specific question + specific number. This is the kind of answer that makes an interviewer
> believe you actually did the work.

---

## Part 3 — Reranking (your strongest material)

**Q8. Bi-encoder vs. cross-encoder?**
> **Answer:** "A bi-encoder — my embedding model — encodes the query and each chunk *separately*, then
> compares vectors. Fast, because I can pre-compute every chunk vector, so I use it to search the whole
> corpus. A cross-encoder takes the query and chunk *together* as one input and outputs a relevance
> score — much more accurate because it sees the interaction, but too slow to run over everything. So I
> retrieve wide with the bi-encoder and rerank a small candidate set with the cross-encoder."
>
> **Delivery:** The "separately vs. together" contrast is the core — gesture two hands apart, then
> together. End on the two-stage pattern; it shows you know *how* to use each, not just what they are.

**Q9. Why retrieve 20 then trim to 5 — why not just retrieve 5?**
> **Answer:** "The cross-encoder is only useful if the right chunk is *in* the candidate set. Retrieving
> wide (20) gives it a better pool to reorder; if I retrieved only 5, a chunk ranked 8th by hybrid could
> never be rescued. 20 is a balance — wide enough to catch near-misses, small enough that the slow
> cross-encoder stays fast."
>
> **Delivery:** This tests whether you understand *why* the two-stage design exists. The phrase "rescue a
> chunk hybrid ranked low" is the one to land.

**Q10. (THE BIG ONE) Tell me about a surprising result or bug.**
> **Answer (STAR):** "On a clean corpus every retriever scored ~1.0 recall@5, so I couldn't tell them
> apart — *(S/T)*. I made the task harder by adding 'archived' documents that were lexically almost
> identical to the current ones but with outdated values, plus questions asking for the *current* value
> *(A)*. The metrics then showed something I didn't expect: hybrid search made accuracy *worse* —
> Recall@1 dropped from 0.42 to 0.33 — because BM25 ranked the archived chunks highly since they shared
> the same words. I diagnosed it as a lexical-overlap problem, and fixed it by upgrading the cross-encoder
> to a stronger model that reads the 'former/superseded' semantics, which nearly doubled hard-question
> Recall@1 to 0.58 *(R)*."
>
> **Delivery:** This is your centerpiece — **practice it until it's smooth but not robotic.** Hit the
> three beats: (1) the saturation problem, (2) the counterintuitive failure with the number, (3) the fix
> with the number. The emotional arc — "I didn't expect this" → "I diagnosed it" → "I fixed it and
> re-measured" — is what makes it memorable. **Always end on 0.58.**

**Q11. Why did the small reranker fail but the bigger one work?**
> **Answer:** "Both archived and current chunks are topically about, say, the 401(k) — so a small
> cross-encoder trained on general relevance can't tell them apart; 'current' is a weak signal to it. The
> larger `bge-reranker-base` has enough capacity to weight the 'former/superseded' cues, so it
> down-ranks the archived chunk. I found this by testing three reranker models and comparing Recall@1 —
> not by assuming bigger is better."
>
> **Delivery:** The last sentence — "found by measurement, not assumption" — is a recurring theme; use it.
> It signals scientific discipline.

---

## Part 4 — Evaluation (this separates you from the pack)

**Q12. How did you evaluate the system?**
> **Answer:** "A gold set of 59 questions with known-relevant keywords. For retrieval I use Recall@k and
> MRR, which are deterministic. For answer quality, an LLM-as-judge grades each answer 1–5 against a
> reference. And a hallucination check on out-of-scope questions verifies the model refuses instead of
> inventing. The key move was adding adversarial distractor documents so the metrics could actually
> *separate* the methods."
>
> **Delivery:** Most juniors have *no* evaluation, so this is a differentiator — say it with some pride.
> Lead with "deterministic" for the retrieval metrics; it shows you know which numbers you can trust.

**Q13. What's MRR and why use it over recall?**
> **Answer:** "Mean Reciprocal Rank — the average of 1/rank of the first correct chunk. Recall@k only asks
> 'is the right chunk *somewhere* in the top k'; MRR asks 'how *high* did it rank'. A reranker's whole job
> is ordering, so recall@5 was blind to it — everything scored ~1.0 — but MRR and Recall@1 showed the
> improvement clearly."
>
> **Delivery:** "Recall asks *whether*, MRR asks *where*" — a clean one-liner. Then tie it to why you
> *needed* it: recall was saturated.

**Q14. Why did your LLM-judge score barely change between methods?**
> **Answer:** "Because at k=5 all three retrievers get the correct chunk *somewhere* into the context, and
> the generator reads all five — so it answers well regardless of internal rank order. Judge@k=5 literally
> can't see a ranking improvement. That's exactly why I report Recall@1 and MRR for the reranker. At a
> small k the ranking gain would convert into an answer-quality gain."
>
> **Delivery:** This is a *trap question* — a flat number looks like failure. Reframing it as "the metric
> can't see what I improved, so I used the right metric" turns an apparent weakness into a display of
> understanding. High-value answer; rehearse it.

**Q15. How do you know your metric isn't lying to you?**
> **Answer:** "I actually hit that — my recall metric was under-reporting because PDF line-wraps split
> keyword phrases across a newline, so an exact-substring check failed even though the text was there. I
> found it by inspecting the 'misses,' saw the chunk *did* contain the answer, and fixed it by normalizing
> whitespace before matching. Lesson: I verify misses by hand before trusting a metric."
>
> **Delivery:** Shows you're skeptical of your own numbers — exactly the instinct interviewers want.

---

## Part 5 — Design, scaling, and curveballs

**Q16. How would this scale to millions of documents?**
> **Answer:** "The bottleneck is the linear scans — my BM25 and cross-encoder run over the candidate set
> in memory, which is fine at ~100 chunks but not millions. I'd move to an approximate-nearest-neighbor
> vector index (HNSW, which ChromaDB supports), a proper BM25 backend like Elasticsearch/OpenSearch, keep
> the cross-encoder only on a small candidate set, and batch/cache embeddings. The retrieve-then-rerank
> shape stays the same; only the first-stage indexes change."
>
> **Delivery:** Name the *specific* bottleneck first (linear scan), then solutions. Saying "the
> architecture stays the same, only the indexes change" shows you see the design, not just the code.

**Q17. How do you reduce hallucination?**
> **Answer:** "Layered. First, ground the answer — the prompt instructs the model to answer only from
> retrieved context. Second, a hallucination metric on out-of-scope questions checks it refuses instead of
> inventing. Third, the LLM-judge penalizes unsupported claims. The clearest next step, which I scoped as
> a future phase, is inline citations so every claim is traceable to a specific chunk."
>
> **Delivery:** "Layered" + three concrete layers. Naming the *next* step (citations) shows you know the
> gap and have a plan — better than pretending it's solved.

**Q18. What are the limitations of your project?** *(Answer this proactively somewhere.)*
> **Answer:** "Four. The corpus is synthetic. My recall metric is keyword-based — robust but coarse, so it
> can't credit a semantically-right chunk that uses different words. The judge and hallucination metrics
> use a hosted LLM, so they wobble slightly run-to-run while recall/MRR are exact. And the out-of-scope set
> is only four questions, so the hallucination rate is noisy. The two clearest next steps are inline
> citations and conversational multi-turn retrieval."
>
> **Delivery:** Volunteering limits *before* being asked reads as maturity and disarms the "gotcha." Keep
> it to a quick list — don't dwell or it sounds like the project is fragile.

**Q19. Why these specific tools / why build it free?**
> **Answer:** "A hard $0 constraint, which forced local-first engineering. Embeddings and reranking run
> locally through fastembed's ONNX runtime — no PyTorch, no GPU, no per-call cost. ChromaDB persists
> locally. Only generation hits a hosted API, and Groq's free tier covers it. So the only thing I depend
> on externally is the final generation call."
>
> **Delivery:** Frame the constraint as a *design driver*, not a limitation. "It forced local-first
> choices" turns "I was cheap" into "I engineered around a constraint."

**Q20. (Curveball) Something you don't know — e.g. "Why cosine and not Euclidean distance?"**
> **Answer:** "Honestly I used ChromaDB's default and didn't benchmark the alternative — but my
> understanding is cosine measures *direction*, which suits normalized embeddings where magnitude isn't
> meaningful, whereas Euclidean is sensitive to vector length. If it mattered I'd measure both on my gold
> set and compare recall."
>
> **Delivery:** **The template for any question you can't fully answer:** (1) admit the boundary honestly,
> (2) reason from what you *do* know, (3) say how you'd find out — measurement. Never bluff; the "here's
> how I'd test it" close turns a gap into a strength.

---

## Part 6 — Behavioral / wrap-up

**Q21. What was the hardest part?**
> **Answer:** "Making the evaluation *discriminating*. The pipeline was straightforward; the hard part was
> that a clean corpus made every method look identical at ~1.0 recall. Designing adversarial distractors
> that exposed the real differences — and being honest when the result surprised me — was the real work."
>
> **Delivery:** Pick a *technical-judgment* hard part, not "debugging an import error." It positions you as
> someone who thinks about the right problem.

**Q22. What would you do differently / next?**
> **Answer:** "Add inline citations and a faithfulness metric so every claim traces to a chunk — that's the
> most-asked RAG question and my clearest gap. Then conversational multi-turn retrieval. I deliberately
> stopped adding retrieval techniques to keep the project something I can defend in depth rather than a
> shallow checklist."
>
> **Delivery:** The last sentence reframes "I stopped" as a *choice about depth* — it's a strong note to
> end the interview on.

---

## Part 7 — The 60-second close (if they ask "anything to add?")

> "The thing I'd highlight is that this isn't just a wired-up pipeline — it's a *measured* one. I built an
> evaluation harness, then made it adversarial on purpose, and the metrics caught hybrid search backfiring
> on near-duplicate documents — something I'd never have seen by eyeballing outputs. I diagnosed it and
> fixed it with a stronger reranker, and re-measured to confirm. That loop — measure, find a real failure,
> fix, re-measure — is what I'd bring to a production system."

**Delivery:** Memorize this almost word-for-word. It's your last impression: it says *engineer*, not
*tutorial-follower*. End there and stop talking.

---

## Cheat sheet — numbers to have on instant recall

| Thing | Number |
|---|---|
| Corpus | 91 chunks, 12 documents (incl. 3 distractors) |
| Gold set | 59 questions (55 answerable, 4 out-of-scope) |
| Hard-question Recall@1: hybrid → rerank | **0.333 → 0.583** (nearly 2×) |
| Hard-question MRR: hybrid → rerank | 0.653 → 0.792 |
| Overall Recall@1: vector / hybrid / rerank | 0.764 / 0.727 / **0.818** |
| Embedding model | `BAAI/bge-base-en`, 768-dim |
| Reranker model | `BAAI/bge-reranker-base` (cross-encoder) |
| Retrieve-then-rerank pool | 20 candidates → top 5 |
| RRF constant | 60 |
