
04 // Evaluating RAG
Retrieval Augmented Generation (RAG) is the most common AI system in production, and the most commonly evaluated wrong. The key move is decomposition: a RAG answer has two separate failure surfaces, and you must measure them separately.

The Mistake That Keeps RAG Systems Broken
Scoring only the final answer is the mistake that keeps RAG systems broken. When the answer is wrong, you have no idea whether retrieval fetched the wrong context or the generator ignored the right context. You cannot fix what you cannot localize. So you split the system in two:

query --> [ RETRIEVAL ] --> context --> [ GENERATION ] --> answer
                |                              |
      did we fetch the               given that context,
       right context?                is the answer good?
Rule: Evaluate each surface on its own, or you are debugging blind.

4.1 Retrieval Evaluation: Did We Fetch the Right Context?
The generator can only work with what retrieval hands it. If the right chunk was never retrieved, no amount of prompt tuning fixes the answer. Measure retrieval first.

Metric	Definition	Why It Matters
Recall@k	Of all the chunks relevant to the query, how many appear in the top k retrieved.	The most important retrieval metric. Low recall means the generator never even saw the answer.
Precision@k	Of the top k chunks retrieved, how many are actually relevant.	Signal vs. noise in context. Low precision floods the context with distraction.
MRR (Mean Reciprocal Rank)	How high the first hit ranks.	Rewards putting a relevant chunk near the top. Position matters, since models weight earlier context.
NDCG (Normalized Discounted Cumulative Gain)	Graded, position-weighted rank quality.	Accounts for degrees of relevance and rewards better ordering. The most complete ranking metric.
Hit Rate	Binary per query: was at least one relevant chunk in the top k.	A blunt but useful health check on whether anything relevant got in.
4.2 Generation Evaluation: Given the Context, Is the Answer Good?
Once you know retrieval delivered good context, evaluate what the generator did with it. These are typically measured with a calibrated LLM judge.

Faithfulness / Groundedness (The Anti-Hallucination Metric)

Is every claim in the answer actually supported by the retrieved context? The judge checks each claim against the context. This is the metric that stops your RAG system from making things up.

Answer Relevance (Did it answer the question?)

Does the answer actually address what the user asked, or drift off into related but useless territory?

Context Relevance (Was the fetched context on topic?)

Was the retrieved context actually relevant to the query? Bridges retrieval and generation.

4.3 The RAG Triad
Put three metrics together and you can localize almost any RAG failure without guessing: Context Relevance, Groundedness, and Answer Relevance. Read as a set, they point straight at the broken component.

Symptom	What the Triad Shows	Where the Bug Is	Solution / Fix
Answer is missing key info	Low context relevance, low recall	Retrieval	Fix chunking, embeddings, or the retriever.
Answer contradicts the sources	Context good, groundedness low	Generation	The model is ignoring or overriding context. Fix the prompt.
Answer wanders off topic	Answer relevance low	Generation	The model missed the user's intent.
Context is full of junk	Context relevance / precision low	Retrieval	Too much irrelevant material retrieved.
4.4 Building a RAG Evaluation Set
To build an evaluation set, you need:

Questions

Ground truth answers

Ideally, ground truth source chunks that should be retrieved

How to bootstrap:

Use an LLM to generate question and answer pairs directly from your document corpus.

Have humans review and correct them to produce a synthetic starter set fast.

Refine and expand the set over time using real user questions from production.

💡 In the Field
When a company says "our RAG isn't accurate enough," they almost never know if it is a retrieval problem or a generation problem, so they tweak prompts at random for weeks.

The engineer who walks in, decomposes it with the triad, and says "your recall at 5 is 0.62, this is a retrieval problem, prompt tuning will not fix it" is immediately the most valuable person in the room. That is a paid skill, not a demo skill.




Gemini is AI and can make mistakes.

Analyzing
# 04 // Evaluating RAG

Retrieval Augmented Generation (RAG) is the most common AI system in production, and the most commonly evaluated wrong. The key move is **decomposition**: a RAG answer has two separate failure surfaces, and you must measure them separately.

---

## The Mistake That Keeps RAG Systems Broken

Scoring only the final answer is the mistake that keeps RAG systems broken. When the answer is wrong, you have no idea whether retrieval fetched the wrong context or the generator ignored the right context. **You cannot fix what you cannot localize.** So you split the system in two:

```
query --> [ RETRIEVAL ] --> context --> [ GENERATION ] --> answer
                |                              |
      did we fetch the               given that context,
       right context?                is the answer good?
```

> **Rule:** Evaluate each surface on its own, or you are debugging blind.

---

## 4.1 Retrieval Evaluation: Did We Fetch the Right Context?

The generator can only work with what retrieval hands it. If the right chunk was never retrieved, no amount of prompt tuning fixes the answer. **Measure retrieval first.**

| Metric | Definition | Why It Matters |
| :--- | :--- | :--- |
| **Recall@k** | Of all the chunks relevant to the query, how many appear in the top $k$ retrieved. | **The most important retrieval metric.** Low recall means the generator never even saw the answer. |
| **Precision@k** | Of the top $k$ chunks retrieved, how many are actually relevant. | Signal vs. noise in context. Low precision floods the context with distraction. |
| **MRR** *(Mean Reciprocal Rank)* | How high the first hit ranks. | Rewards putting a relevant chunk near the top. Position matters, since models weight earlier context. |
| **NDCG** *(Normalized Discounted Cumulative Gain)* | Graded, position-weighted rank quality. | Accounts for degrees of relevance and rewards better ordering. The most complete ranking metric. |
| **Hit Rate** | Binary per query: was at least one relevant chunk in the top $k$. | A blunt but useful health check on whether anything relevant got in. |

---

## 4.2 Generation Evaluation: Given the Context, Is the Answer Good?

Once you know retrieval delivered good context, evaluate what the generator did with it. These are typically measured with a calibrated LLM judge.

* **Faithfulness / Groundedness** *(The Anti-Hallucination Metric)*  
  Is every claim in the answer actually supported by the retrieved context? The judge checks each claim against the context. This is the metric that stops your RAG system from making things up.
* **Answer Relevance** *(Did it answer the question?)*  
  Does the answer actually address what the user asked, or drift off into related but useless territory?
* **Context Relevance** *(Was the fetched context on topic?)*  
  Was the retrieved context actually relevant to the query? Bridges retrieval and generation.

---

## 4.3 The RAG Triad

Put three metrics together and you can localize almost any RAG failure without guessing: **Context Relevance**, **Groundedness**, and **Answer Relevance**. Read as a set, they point straight at the broken component.

| Symptom | What the Triad Shows | Where the Bug Is | Solution / Fix |
| :--- | :--- | :--- | :--- |
| **Answer is missing key info** | Low context relevance, low recall | **Retrieval** | Fix chunking, embeddings, or the retriever. |
| **Answer contradicts the sources** | Context good, groundedness low | **Generation** | The model is ignoring or overriding context. Fix the prompt. |
| **Answer wanders off topic** | Answer relevance low | **Generation** | The model missed the user's intent. |
| **Context is full of junk** | Context relevance / precision low | **Retrieval** | Too much irrelevant material retrieved. |

---

## 4.4 Building a RAG Evaluation Set

To build an evaluation set, you need:
1. Questions
2. Ground truth answers
3. Ideally, ground truth source chunks that should be retrieved

**How to bootstrap:**
* Use an LLM to generate question and answer pairs directly from your document corpus.
* Have humans review and correct them to produce a synthetic starter set fast.
* Refine and expand the set over time using real user questions from production.

---

> ### 💡 In the Field
> When a company says *"our RAG isn't accurate enough,"* they almost never know if it is a retrieval problem or a generation problem, so they tweak prompts at random for weeks. 
>
> The engineer who walks in, decomposes it with the triad, and says *"your recall at 5 is 0.62, this is a retrieval problem, prompt tuning will not fix it"* is immediately the most valuable person in the room. **That is a paid skill, not a demo skill.**
evaluating_rag.md
Displaying evaluating_rag.md.