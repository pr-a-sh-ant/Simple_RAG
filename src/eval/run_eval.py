import os
import json
import argparse
from datetime import date
from pathlib import Path
import re

from tabulate import tabulate
from src.utils.parse import parse_text

from src.services import VectorStoreService, DocumentService, PromptService
from src.model.model_gen import ChatService

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path(__file__).parent / "results"

vector_store_svc = VectorStoreService()
document_svc = DocumentService(upload_dir=UPLOAD_DIR)
prompt_svc = PromptService()
model_api = ChatService()


def ingest(file_path):
    """One-time: load a document, chunk it, and add it to the vector store."""
    documents = document_svc.load_file(file_path)
    chunks = document_svc.split_documents(documents, source=str(file_path))
    vector_store_svc.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunks from {file_path}")


def evaluate(data, k=5, mode="rerank"):
    responses = []

    for item in data:
        question = item['question']
        hits = vector_store_svc.search(question, k=k, mode=mode)

        context = document_svc.format_context(hits)
        prompt, system = prompt_svc.get_prompt()
        prompt = prompt.format(context=context , question=question)

        response = model_api.generate_content(system, prompt)
        responses.append({
            "id": item['id'],
            "question": question,
            "expected_answer": item['expected_answer'],
            "category": item['category'],
            "answerable": item['answerable'],
            "hits": hits,
            "relevant_keywords": item['relevant_keywords'],
            "response": response.choices[0].message.content
        })

    return responses

def recall_metric(answerable_responses, k=5):
    successes, misses = 0, []
    for data in answerable_responses:
        top_k = data['hits'][:k]
        kws = data['relevant_keywords']
        norm = lambda s: re.sub(r'\s+', ' ', s).lower()
        success = any(all(kw.lower() in norm(chunk_test) for kw in kws) for (chunk_test, meta, dist) in top_k)
        if success:
            successes += 1
        else:            
            misses.append(data['id'])
    return successes / len(answerable_responses) , misses

def judge_metrics(responses):
    judgements, score = [], 0
    for data in responses:
        if data['answerable']:
            response = model_api.judge(data['question'], data['expected_answer'], data['response'])
            judgement = parse_text(response.choices[0].message.content)
            score += judgement['score']            
            judgements.append({
                "id": data['id'],
                "question": data['question'],
                "expected_answer": data['expected_answer'],
                "model_answer": data['response'],
                "judgement": judgement
            })

    return judgements, score / len(responses)
    
def hallucination_metric(responses):
    hallucinations, score = [], 0
    for data in responses:
        response = model_api.hallucination(data['question'], data['response'])
        hallucination = parse_text(response.choices[0].message.content)
        if hallucination['verdict'] == "hallucinated":
            hallucinations.append({
                "id": data['id'],
                "question": data['question'],
                "model_answer": data['response'],
                "hallucination": hallucination
            })
            score += 1

    return hallucinations, score / len(responses) * 100
            


def main():
    parser = argparse.ArgumentParser(description="Run the RAG evaluation harness.")
    parser.add_argument("--ingest", help="Ingest a document into the vector store, then exit.")
    parser.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve (default: 5).")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions (for quick tests).")
    parser.add_argument("--label", default="baseline", help="Names the output file: results/<label>_<date>.json")
    parser.add_argument("--retriever", choices=["vector", "hybrid", "rerank"], default="rerank",
                        help="Retrieval mode: vector-only, hybrid (RRF), or hybrid+rerank (default: rerank).")
    args = parser.parse_args()

    # One-time ingestion path
    if args.ingest:
        ingest(args.ingest)
        return

    with open(os.path.join(os.path.dirname(__file__), 'gold_questions.json'), 'r') as file:
        gold = json.load(file)

    questions = gold['questions']
    if args.limit:
        questions = questions[:args.limit]

    responses = evaluate(questions, k=args.k, mode=args.retriever)

    answerable_responses = [r for r in responses if r['answerable']]

    recall_k, misses_k = recall_metric(answerable_responses, k=args.k)
    recall_3, misses_3 = recall_metric(answerable_responses, k=3)

    judgements, avg_score = judge_metrics(answerable_responses)

    out_of_scope = [r for r in responses if not r['answerable']]
    hallucinations, hallucination_rate = hallucination_metric(out_of_scope) if out_of_scope else ([], 0)

    table_data = [
        ["Metric", "Value"],
        [f"Recall (k={args.k})", round(recall_k, 4)],
        ["Recall (k=3)", round(recall_3, 4)],
        ["Judge Avg Score (/5)", round(avg_score, 4)],
        ["Hallucination Rate (%)", round(hallucination_rate, 2)],
    ]
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    print(f"Recall misses (k={args.k}): {misses_k}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.label}_{date.today().isoformat()}.json"
    with open(out_path, "w") as f:
        json.dump({
            "label": args.label,
            "date": date.today().isoformat(),
            "retriever": args.retriever,
            "k": args.k,
            "num_questions": len(questions),
            "summary": {
                f"recall_{args.k}": recall_k,
                "recall_3": recall_3,
                "avg_score": avg_score,
                "hallucination_rate": hallucination_rate,
            },
            "recall_misses": misses_k,
            "responses": answerable_responses,
            "judgements": judgements,
            "hallucinations": hallucinations,
        }, f, indent=4)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
