# Embedding Models in Retrieval Systems

## What an embedding model does

An embedding model maps a span of text to a fixed-length vector of floating point
numbers. Passages whose meanings are close end up close together in that vector
space, which is what allows a search to match a question to a passage that shares no
words with it. The mapping is fixed once the model is trained; nothing about a
particular document collection changes it.

## Bi-encoders and cross-encoders

The model used in this service is a bi-encoder. A bi-encoder encodes the question and
the passage entirely separately and compares the two resulting vectors. This is what
makes retrieval fast: every passage in the collection can be encoded once, ahead of
time, and a query only requires encoding the question and comparing vectors.

A cross-encoder instead feeds the question and the passage through the network
together and outputs a single relevance score. It is substantially more accurate
because it can attend to the interaction between the two texts, but it cannot
pre-compute anything. Scoring one million passages with a cross-encoder means one
million forward passes per query, which is why cross-encoders are used as rerankers
over the top twenty or fifty results rather than as the primary retrieval mechanism.

## all-MiniLM-L6-v2

The specific model here is all-MiniLM-L6-v2 from the sentence-transformers family. It
has six transformer layers, produces vectors with 384 dimensions, and was trained on
roughly one billion sentence pairs. Its input is truncated at 256 word-piece tokens.

384 dimensions is small by current standards. Larger models such as all-mpnet-base-v2
produce 768-dimensional vectors and score several points higher on retrieval
benchmarks, at roughly three times the inference cost and twice the storage. For a
service that must run on CPU without a GPU budget, the smaller model is the
defensible choice.

## Word-piece tokenization

The 256-token limit is counted in word-piece tokens, not in words or characters. A
word-piece tokenizer splits rare words into fragments drawn from a fixed vocabulary,
so "tokenization" becomes "token" and "##ization". Common English prose runs about
1.3 word-piece tokens per whitespace word, but code, tables, chemical names and
non-English text run far denser -- sometimes three or four tokens per word.

This is precisely why a character budget is unsafe. A 1000-character chunk of plain
English prose is roughly 200 tokens and fits. The same 1000 characters of dense
technical notation can exceed 400 tokens, and everything past the 256th token is
discarded silently before the vector is produced. The stored text still looks
complete, so the loss is invisible unless tokens are counted explicitly.

## Normalisation and the distance metric

Vectors are L2-normalised to unit length at encode time. Once every vector has length
one, the cosine of the angle between two vectors equals their dot product, and cosine
distance is exactly one minus cosine similarity. A similarity of 1.0 means the two
texts encode identically; 0.0 means they are orthogonal, which in practice means
unrelated.

Observed similarity scores are not calibrated probabilities. A score of 0.45 does not
mean a 45 percent chance of relevance. The scores are only comparable to each other
within the same model, which is why any threshold must be set from the observed
distribution on a real corpus rather than picked from intuition.

## Matching the model on both sides

The same embedding model must encode the documents at ingestion and the questions at
query time. Encoding documents with one model and questions with another places the
two sets of vectors in unrelated coordinate systems. Nothing raises an error. The
index returns results, the scores look plausible, and the ranking is meaningless.
This is among the most common and hardest to diagnose faults in a retrieval system,
because every component reports success.
