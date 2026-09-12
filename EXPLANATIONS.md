# Explanations

Every number in this document was produced by a script in this repository and is
reproducible from the committed data files. The sources are
[`eval/eval_results.json`](eval/eval_results.json) (chunk sweep),
[`eval/failure_report.json`](eval/failure_report.json) (adversarial retrieval), and
`GET /metrics/summary` over `data/metrics.jsonl` (25 live queries).

Corpus: 4 documents, 23 chunks at the chosen configuration, 22 evaluation questions.
It is small. Where that limits a conclusion, this document says so rather than
implying more confidence than the measurement supports.

---

## 1. Chunk size

### The hard constraint comes first

`all-MiniLM-L6-v2` truncates its input at **256 word-piece tokens**. This is not a
tuning preference, it is a property of the model, and the service reads it back at
runtime rather than assuming it:

```python
Embedder("sentence-transformers/all-MiniLM-L6-v2").max_seq_length  # -> 256
```

A test pins this value (`tests/test_vectorstore.py::test_embedder_reports_the_dimension_and_the_256_token_ceiling`),
so a model swap that changes the ceiling fails the suite instead of silently
degrading retrieval.

Anything past the 256th token is discarded before the vector is produced. The stored
text stays complete, so nothing in the API response, the database, or the logs shows
that a chunk's tail contributed nothing to its own vector. This failure is invisible
unless tokens are counted deliberately, which is why the chunker measures word-piece
tokens from the model's own tokenizer rather than characters or whitespace words.

Counting characters instead would not merely be imprecise, it would be unsafe by a
variable amount. English prose runs about 1.3 word-piece tokens per word, but code,
tables and non-English text run three to four times denser. A character budget tuned
on prose silently overflows the ceiling the moment the corpus contains a table.

### The sweep

`python scripts/eval_chunking.py` ingests the corpus into an isolated collection per
configuration and runs all 22 questions against each.

| size / overlap | chunks | mean tokens | max tokens | truncated by model | recall@5 | top-1 doc correct | mean top-1 similarity |
|---|---|---|---|---|---|---|---|
| 100 / 20 | 40 | 75.5 | 99 | 0 | 1.000 | 1.000 | **0.6809** |
| 180 / 40 | 23 | 132.0 | 180 | 0 | 1.000 | 0.955 | 0.5808 |
| 300 / 60 | 13 | 232.5 | 292 | **5 of 13 (38%)** | 1.000 | 1.000 | 0.5405 |
| 500 / 100 | 10 | 328.0 | 458 | **6 of 10 (60%)** | 1.000 | 0.864 | 0.4875 |

Two findings, and one non-finding that matters.

**The truncation count is the headline.** At 300 tokens, 5 of 13 chunks exceed the
ceiling; at 500, 6 of 10 do, one of them by 202 tokens. Those chunks are stored whole
and embedded partial. This is the concrete cost of picking a chunk size without
reference to the model, and it is why `chunk_text` raises a `ValueError` when asked
for a size above the ceiling rather than accepting it:

```
chunk_size_tokens (400) exceeds the model ceiling (256); chunks would be
truncated before embedding
```

The sweep can still measure those configurations because it raises the guard
explicitly — a deliberate act, visible in the code, not a default.

**Mean top-1 similarity falls monotonically as chunks grow**: 0.6809 → 0.5808 →
0.5405 → 0.4875. This is the dilution effect. One vector must represent everything
its passage discusses, so the more a chunk covers, the weaker its match to any single
fact inside it. At 500 tokens, top-1 document accuracy also drops to 0.864 — the
system starts returning the wrong document first.

**recall@5 is 1.000 everywhere and is therefore uninformative here.** With 10–40
chunks in the corpus and `top_k=5`, retrieving five chunks means retrieving between
12% and 50% of everything indexed. recall@5 cannot discriminate at this corpus size.
Reporting it as evidence that 500-token chunks are fine would be wrong, and the
discriminating metrics — mean top-1 similarity and top-1 document accuracy — both say
the opposite. A corpus of hundreds of documents would be needed before recall@5 says
anything.

### The choice, and the trade

**180 tokens with 40 tokens of overlap**, which is what `config.py` defaults to.

100/20 scores higher on mean top-1 similarity (0.6809 vs 0.5808) and is genuinely the
better retrieval configuration on this corpus. It is not chosen, because retrieval
score is not the only objective: 100-token chunks produce 40 records where 180 produce
23, and each retrieved chunk carries less surrounding context for the model to reason
over. Several corpus facts — the HNSW parameter explanation, the bi-encoder/
cross-encoder contrast — need roughly 150 tokens to state completely. At 100 tokens
they split across records, and answering needs two retrieved chunks where one would
do.

180 sits at about 70% of the ceiling. That headroom is deliberate: a chunk is a slice
of real text, and the tokenizer's count for a given passage is not perfectly
predictable in advance, so leaving 76 tokens of margin means no ordinary chunk
approaches truncation. The measured maximum at this configuration is exactly 180.

The honest summary: this is a trade of about 0.10 of mean top-1 similarity for
answers with more context and half as many records to store. On a larger and more
heterogeneous corpus the sweep should be rerun, and if answers were being assembled
from fragments, 100/20 would be the better choice.

---

## 2. A retrieval failure, observed

From `python scripts/find_failures.py`. Ten questions across three adversarial
categories; **2 of 10 failed to retrieve the answer at all**, and 3 more retrieved it
below rank 1.

| category | questions | answer in top-5 | answer ranked 1st |
|---|---|---|---|
| vocabulary_mismatch | 4 | 3/4 | 2/4 |
| boundary_spanning | 3 | 3/3 | 2/3 |
| multi_hop | 3 | 2/3 | 1/3 |

### The clearest case

**Question:** *"How long may a passage be before the machine quietly clips the end off
it?"*

The corpus answers this directly. `embedding_models.md` chunk 1 contains "Its input is
truncated at 256 word-piece tokens."

**What was retrieved instead:**

| rank | score | source | chunk | contains the answer |
|---|---|---|---|---|
| 1 | **0.2297** | rag_notes.pdf | 2 | no |
| 2 | 0.2129 | rag_notes.pdf | 3 | no |
| 3 | 0.1877 | rag_notes.pdf | 5 | no |
| 4 | 0.1687 | rag_notes.pdf | 0 | no |
| 5 | 0.1573 | retrieval_evaluation.txt | 4 | no |

The correct chunk does not appear at all. Scored directly against it:

| question phrasing | similarity to the correct chunk |
|---|---|
| "How long may a passage be before the machine quietly clips the end off it?" | **0.1010** |
| "At how many word-piece tokens does all-MiniLM-L6-v2 truncate its input?" | **0.3875** |

**The same chunk, the same model, the same question in substance — and 0.2865 of
similarity lost to paraphrase alone.** The chunk that wrongly won scored 0.2297, more
than twice the correct chunk's 0.1010. It won because it is *about* passage length in
the abstract ("Chunk size is the single most consequential parameter...") and shares
surface vocabulary with the question, while the correct chunk states the specific fact
using words the question never uses: "truncate", "token", "256".

There is a second consequence. Every score here is below the 0.25 floor, so the
service refused the question entirely and returned "No relevant content was found."
The refusal is correct behaviour given what retrieval returned — but the corpus does
contain the answer. A vocabulary mismatch has turned an answerable question into a
refusal, and from outside the service that is indistinguishable from the corpus
genuinely lacking the information. The returned similarity scores are the only thing
that lets an operator tell the two apart, which is why they are part of the response
body rather than only the logs.

### The mechanism

A bi-encoder encodes the question and the passage independently, then compares the two
vectors. There is no opportunity for the question's specific terms to influence how the
passage is read, because the passage was encoded before the question existed.
Embeddings capture topical similarity well and specific-entity or specific-value
matching poorly. "Quietly clips the end off" and "truncates" are the same idea to a
human and roughly 0.10 apart to this model.

The second failure has a different mechanism. *"What similarity floor is recommended,
and how many dimensions does the model that produced those scores emit?"* needs two
facts that live in two different documents. Retrieval returned the floor half
confidently (0.4390) and never surfaced the chunk containing "384 dimensions". No
single vector can be close to a question with two unrelated halves; the question's
embedding lands somewhere between the two topics and matches whichever is nearer.

### What would fix it

1. **Hybrid retrieval — BM25 plus dense, fused.** BM25 matches on lexical overlap and
   would not rescue this specific question (the wording shares nothing), but it is the
   direct fix for the inverse and more common case: exact identifiers, error codes,
   product names and numbers, which dense retrieval handles poorly. The two methods
   fail on different inputs, which is what makes fusing them worthwhile. Reciprocal
   rank fusion needs no tuning and no training data.

2. **A cross-encoder reranker over the top 20.** Retrieve 20 with the bi-encoder, then
   rescore with a model that reads question and passage together. This directly
   addresses the failure above: a cross-encoder can see that "clips the end off"
   and "truncates" refer to the same operation, because it attends to both texts at
   once. Cost is roughly 20 forward passes per query — tens of milliseconds on CPU for
   a small reranker — which is affordable at this scale and is the change I would make
   first.

3. **For the multi-hop case, decompose the question.** Split it into single-fact
   sub-questions, retrieve for each, and merge the context. This is a retrieval
   orchestration change rather than a model change, and it is the only one of the three
   that addresses multi-hop failure at all.

---

## 3. The metric worth tracking

**Retrieval latency (as percentiles) and top-1 similarity (as a distribution).**

From `GET /metrics/summary` over 25 recorded queries — 22 answerable, 3 deliberately
out of domain.

### Retrieval latency

| statistic | ms |
|---|---|
| mean | 32.9 |
| p50 | 32.1 |
| p95 | 41.8 |
| p99 | 47.3 |
| min | 25.5 |
| max | 48.8 |

Retrieval is a small and stable fraction of the request. Generation dominates
completely: p50 5287 ms, p95 8810 ms, against retrieval's p50 of 32 ms — **roughly
160× the retrieval cost at the median**. Optimising retrieval latency here would be
effort spent on 0.6% of the response time. That is itself the finding: it says the
next performance work belongs in generation (streaming the response, or a smaller
model) and not in the index.

Percentiles rather than a mean because the distribution is right-skewed — max 48.8 ms
against a p50 of 32.1 ms. Two of these 25 queries would be hidden inside a mean.

### Top-1 similarity

| statistic | value |
|---|---|
| mean | 0.5808 |
| p50 | 0.6097 |
| p95 | 0.7737 |
| min | 0.2999 |
| max | 0.7925 |

Distribution over the 22 answered queries:

| bucket | count |
|---|---|
| 0.2–0.3 | 1 |
| 0.3–0.4 | 1 |
| 0.4–0.5 | 5 |
| 0.5–0.6 | 2 |
| 0.6–0.7 | 10 |
| 0.7–0.8 | 3 |

This is the metric that makes a bad answer diagnosable. When the best passage in the
collection scores 0.19 against a question, no prompt change will produce a correct
answer, because the information is not in the context. Recording the score turns "the
answer was wrong" into "the answer was wrong and the best available passage scored
0.19", which points at the corpus instead of the prompt.

### How the floor was set from this

The floor is not an intuition. Measured across 22 answerable and 6 unanswerable
questions against the same index:

| set | n | min | median | max |
|---|---|---|---|---|
| answerable | 22 | **0.2999** | 0.6130 | 0.7925 |
| unanswerable | 6 | 0.0067 | 0.0775 | **0.1377** |

The two distributions do not overlap. The weakest answerable question scores 0.2999;
the strongest unanswerable one scores 0.1377. **The gap is 0.1622, and there is no
value in it that misclassifies anything.**

`min_similarity = 0.25` sits inside that gap, with 0.05 of margin below the weakest
true positive and 0.11 above the strongest true negative. The asymmetry is deliberate:
the cost of a wrong refusal is a user who has to rephrase, while the cost of admitting
noise is a confident, fluent, wrong answer, which is worse and harder to detect. The
margin is therefore biased toward refusing.

The measured refusal rate over these 25 queries is **0.12** — exactly the 3 out-of-
domain questions, with no answerable question refused. That number is worth alerting
on: a sudden rise in refusal rate with unchanged code almost always means the corpus
changed, not the service.

**The caveat:** a 0.05 margin below the weakest true positive is thin, and it was
measured on 22 questions against 4 documents. The vocabulary-mismatch failure in
section 2 scored 0.2297 on an answerable question — below the floor, and it was
refused. So the floor is already known to produce at least one wrong refusal on a
paraphrased question. On a larger corpus I would re-measure before trusting 0.25, and
would expect to lower it if a reranker were added, since the reranker would then be
responsible for rejecting weak matches.
