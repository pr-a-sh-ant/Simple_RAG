from groq import Groq
import os


    
class ChatService:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("API_KEY"))

    def generate_content(self, system, prompt):
        response = self.client.chat.completions.create(
            messages=[{"role":"system", "content": system},{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.0,
        )
        return response
    
    def judge(self, question, expected, actual):
        system = """
        You are a strict evaluator grading a RAG system's answer against a reference answer.
        Score the MODEL ANSWER from 1 to 5 on correctness and faithfulness to the reference:

        5 = fully correct; matches the reference; no invented facts
        4 = correct but minor omission or harmless extra detail
        3 = partially correct; missing a key fact
        2 = mostly wrong or misleading
        1 = incorrect or fabricated

        Rules:
        - Judge ONLY against the reference answer, not your own knowledge.
        - Penalize any claim not supported by the reference.
        - Reward concise, accurate answers; do not reward extra unsupported detail.

        Respond with ONLY a JSON object, no other text:
        {"score": <integer 1-5>, "reason": "<one short sentence>"}
        """

        prompt = """
        Question: {question}
        Reference answer: {expected}
        Model answer: {actual}

        Grade the model answer."""

        prompt = prompt.format(question=question, expected=expected, actual=actual)
        response = self.client.chat.completions.create(
            messages=[{"role":"system", "content": system}, {"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.0,
        )
        return response
    
    def hallucination(self, question, actual):
        system = """
        You are evaluating whether a RAG system correctly handled a question whose
        answer is NOT in the source document. The correct behavior is to decline —
        say the information isn't available — WITHOUT inventing any specific facts.

        Classify the MODEL ANSWER:
        - "refused"      : declines / says it doesn't know, and gives NO fabricated specifics
        - "hallucinated" : provides specific facts not grounded in the document
                        (this includes answers that decline but THEN add made-up details)

        Respond with ONLY JSON: {"verdict": "refused" | "hallucinated", "reason": "<one line>"
        """
        prompt = """
        Question: {question}
        Model answer: {actual}

        Classify the model answer."""

        prompt = prompt.format(question=question, actual=actual)
        response = self.client.chat.completions.create(
            messages=[{"role":"system", "content": system}, {"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.0,
        )
        return response