RAG Evals: The Page That Determines Whether You Ship
This is the page that separates RAG that works in production from RAG that quietly fails.

You cannot ship RAG without retrieval evals. I'll repeat that: you cannot ship RAG without retrieval evals.

Without an eval set, every change to your RAG system is faith-based. With one, you have a measuring stick that prevents regressions and proves improvements.

What to measure
Layer 1: Retrieval quality (do the right chunks come back?)
For a set of (query, expected_relevant_chunks) pairs, measure:

Recall@k - Of the truly relevant chunks for this query, how many appear in top-k? Target: 80%+ at k=5–10.
MRR (Mean Reciprocal Rank) - For the first truly relevant chunk in the results, what's its inverse rank? Range 0–1, higher is better.
NDCG (optional, when you have graded relevance) - Normalized Discounted Cumulative Gain. More sensitive than recall for fine-grained ranking quality.
If retrieval recall < 70%, stop optimizing the generation layer. Fix retrieval first. The LLM has nothing useful to work with.

Layer 2: Generation quality (does the answer use the chunks correctly?)
For a set of (query, expected_answer) pairs:

Faithfulness - Does the answer stay grounded in retrieved context, or does it hallucinate beyond what the chunks say? Score 1–5 via LLM-as-judge.
Answer correctness - Does the answer match the expected answer? LLM-as-judge for first pass; human spot-checks for high-stakes evaluation.
Citation accuracy - If the answer cites chunks, do the citations actually support the cited claim?
Building the eval set (the part everyone skips)
This is the work that separates engineers who ship RAG from engineers who quietly stop talking about their RAG project.

Pick 50–200 representative queries - real user queries if you have them; otherwise, brainstorm with PM/domain experts.
For each, manually label:
Which chunks (by ID) are truly relevant
What the ideal answer looks like (a paragraph)
Save it as a JSONL file. Version-control it. Treat it as a first-class engineering artifact.
Run your RAG system on each query, score automatically.
Track these scores on every change - chunking change, embedding upgrade, reranker swap, prompt tweak. Anything that moves these scores is real progress.
50 high-quality queries beats 5000 lazy ones. Quality of the eval set matters more than size.

The drop-in eval script
A 70-line rag_eval.py that runs end-to-end:

"""
rag_eval.py - A minimal but complete RAG evaluation harness.

Measures:
  - Recall@k for retrieval
  - MRR for retrieval
  - Faithfulness (LLM-as-judge) for generation
  - Answer correctness (LLM-as-judge) for generation

Usage:
  1. Build eval_set.jsonl with format:
       {"query": "...", "relevant_chunk_ids": ["c1", "c5"], "ideal_answer": "..."}
  2. Implement `your_rag.retrieve(query)` and `your_rag.answer(query, chunks)`
  3. Run: python rag_eval.py
"""

import json
from dataclasses import dataclass, field
from statistics import mean
from typing import Callable
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

# --- Judge models ------------------------------------------------------------

class FaithfulnessJudgement(BaseModel):
    score: int = Field(ge=1, le=5, description="1=fully hallucinated, 5=fully grounded in context")
    reasoning: str

class CorrectnessJudgement(BaseModel):
    score: int = Field(ge=1, le=5, description="1=wrong, 5=matches ideal answer")
    reasoning: str

judge = instructor.from_openai(OpenAI())

def judge_faithfulness(query: str, answer: str, context_chunks: list[str]) -> int:
    result = judge.chat.completions.create(
        model="gpt-4o-2024-08-06",
        response_model=FaithfulnessJudgement,
        messages=[{"role": "user", "content": f"""
Query: {query}
Retrieved context:
{chr(10).join(context_chunks)}

Generated answer: {answer}

Rate the faithfulness of the answer (1=hallucinated, 5=fully grounded in context).
"""}],
    )
    return result.score

def judge_correctness(query: str, answer: str, ideal: str) -> int:
    result = judge.chat.completions.create(
        model="gpt-4o-2024-08-06",
        response_model=CorrectnessJudgement,
        messages=[{"role": "user", "content": f"""
Query: {query}
Ideal answer: {ideal}
Generated answer: {answer}

Rate the correctness (1=wrong, 5=matches ideal answer).
"""}],
    )
    return result.score

# --- Retrieval metrics -------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(top_k & relevant) / len(relevant)

def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0

# --- Eval loop ---------------------------------------------------------------

@dataclass
class EvalResult:
    k: int = 5
    recalls: list[float] = field(default_factory=list)
    mrrs: list[float] = field(default_factory=list)
    faithfulness: list[int] = field(default_factory=list)
    correctness: list[int] = field(default_factory=list)

    def summary(self) -> dict:
        def safe_mean(lst): return round(mean(lst), 3) if lst else 0.0
        return {
            f"recall@{self.k}": safe_mean(self.recalls),
            "mrr": safe_mean(self.mrrs),
            "faithfulness_mean": safe_mean(self.faithfulness),
            "correctness_mean": safe_mean(self.correctness),
            "n": len(self.recalls),
        }

def run_eval(
    eval_set_path: str,
    retrieve_fn: Callable[[str], list[dict]],  # returns [{"id": ..., "text": ...}, ...]
    answer_fn: Callable[[str, list[dict]], str],
    k: int = 5,
) -> EvalResult:
    result = EvalResult(k=k)
    with open(eval_set_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            retrieved = retrieve_fn(row["query"])
            retrieved_ids = [c["id"] for c in retrieved]
            result.recalls.append(recall_at_k(retrieved_ids, row["relevant_chunk_ids"], k))
            result.mrrs.append(mrr(retrieved_ids, row["relevant_chunk_ids"]))
            answer = answer_fn(row["query"], retrieved[:k])
            chunk_texts = [c["text"] for c in retrieved[:k]]
            result.faithfulness.append(judge_faithfulness(row["query"], answer, chunk_texts))
            result.correctness.append(judge_correctness(row["query"], answer, row["ideal_answer"]))
    return result

# --- Run it ------------------------------------------------------------------
if __name__ == "__main__":
    # Replace with your actual retrieval + answer module
    # import your_rag
    # result = run_eval("eval_set.jsonl", your_rag.retrieve, your_rag.answer)
    # print(json.dumps(result.summary(), indent=2))
    print("Configure your retrieve_fn and answer_fn, then uncomment the lines above.")
This is the artifact people fork. ~70 lines. Drops into any RAG project. Gives you a measuring stick. The single biggest unlock for shipping RAG is having this file in your repo.

Online evals (after offline evals work)
Once your offline eval suite is green on every commit, add online evals on production traffic:

Log every production query + retrieved chunks + final answer
Sample 5–10% for LLM-as-judge evaluation in real time
Track moving averages of faithfulness + correctness over 7-day windows
Alert if any metric drops below threshold
This catches silent regressions that offline evals miss. The difference between "RAG works at launch" and "RAG works in 18 months without silent degradation."

Tools (in honest order)
Roll your own (the rag_eval.py above) - for most projects. Total control. Free.
Ragas - open-source RAG eval framework. Good if you want more metrics out of the box. Decent.
Arize Phoenix - hosted observability + evals. Use if you want a UI for tracking metrics over time.
TruLens - similar to Phoenix. Both work.
LangSmith - if you're already using LangChain. Otherwise overkill.
Strong opinion: don't pick a framework before you've rolled your own simple version. The frameworks add features you may not need; the simple version teaches you what eval actually is. Build the simple version first.

RAG Anti-Patterns I See All The Time
The Phase 2 equivalent of "what to skip." These are the mistakes that quietly cost engineers weeks.

❌ "I'll fine-tune the embedding model"
Almost never the right move in 2026. Spend that time on chunking, metadata, hybrid retrieval, and reranking instead. The lift is 10x for 1/10th the work.

❌ "I'll just retrieve top-10 chunks and let the LLM figure it out"
More chunks ≠ better answer. Past ~5–8 chunks, the LLM's attention degrades and it starts confusing details across chunks. Retrieve more (top-20 candidates) for reranker input, but only pass top-5 to the LLM.

❌ "I'll embed the whole document"
Embedding a 50-page PDF as one vector loses all granularity. The user queries one specific paragraph; you're matching their query against the average meaning of 50 pages. Chunk first.

❌ "I'll use LLM to rewrite every query"
Adds latency + cost to every retrieval. Only worth it for complex queries. For simple keyword-rich queries, original query + hybrid retrieval beats LLM-rewritten query + dense retrieval. Don't pay for the LLM call unless evals show it helps.

❌ "I'll skip metadata"
The single cheapest retrieval win. Adding even 2-3 metadata fields per chunk (source, date, tier, language) and filtering at retrieval typically lifts recall 10–20% on production workloads. The customer-support hallucination story I told in Section A traces back to this.

❌ "I'll cache embeddings"
You should - but also cache query results when possible. Most production RAG systems have a long tail of repeated queries (the same FAQ asked 1000 different ways). Cache the top-k results for the top-N most common queries. Saves money + latency.

❌ "I'll evaluate with vibes"
You read your RAG output, it seems right, you ship it. Three weeks later production tells you it's wrong on 30% of queries. Vibes ≠ evals. Build the eval set. Without it, you're flying blind.

❌ "I'll use the latest framework"
LangChain, LlamaIndex, Haystack all have RAG abstractions. They're tempting because they save you initial setup. They cost you debug time later when something opaque breaks. For your first production RAG, write the retrieval logic yourself (the Retrieval Patterns section's retrieve() function is ~30 lines). You'll understand every line. When something breaks, you'll know where to look.

Free Newsletter
Get The Weekly Decode
Production AI/LLM system design, broken down every week - same depth as 