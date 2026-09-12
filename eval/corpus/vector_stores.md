# Vector Stores and Approximate Search

## Exact versus approximate search

A brute-force nearest-neighbour search compares the query vector against every stored
vector. It is exact and, for a few thousand passages, fast enough that nothing else is
warranted: 10,000 comparisons of 384-dimensional vectors take a few milliseconds.

Beyond roughly 100,000 vectors, brute force stops being viable and an approximate
index is required. Approximate search trades a small amount of recall for a large
reduction in latency. The trade is real: an approximate index may simply fail to
return the true nearest neighbour, and no error is raised when it does.

## HNSW

The index used by Chroma is HNSW, Hierarchical Navigable Small World. It builds a
layered graph in which each node links to a bounded number of neighbours. A search
descends from a sparse top layer to the dense bottom layer, greedily following
whichever link moves closest to the query.

Two parameters govern its behaviour. The construction parameter, usually written
ef_construction, controls how much effort is spent building the graph; higher values
give a better graph at the cost of build time. The search parameter, ef_search,
controls how many candidates are examined per query; raising it improves recall and
increases latency roughly linearly. The default ef_search of 10 is aggressive and
often the reason a system appears to lose documents it definitely indexed.

## Chroma

Chroma is an embedded vector database. A PersistentClient writes to a local directory,
so there is no server to run and no network hop on a query. Collections carry metadata
alongside each vector, and searches can filter on that metadata before the vector
comparison happens.

Chroma's default distance metric is L2, squared Euclidean distance. For normalised
vectors L2 and cosine produce the same ranking, but the numbers differ, so a
collection intended to report cosine similarity must be created with the cosine space
set explicitly. Chroma returns a distance, never a similarity; converting one to the
other is the caller's responsibility, and getting the direction wrong inverts the
entire ranking.

By default Chroma also attaches its own embedding function to a collection and will
happily embed raw text with a model of its own choosing. In a system that controls
its embedder deliberately, that default must be disabled and vectors passed
explicitly, or the collection ends up holding vectors from two different models.

## FAISS and the alternatives

FAISS, from Meta, is a library rather than a database. It is faster than Chroma at
scale and supports quantisation schemes that cut memory dramatically, but it stores
no metadata and has no persistence layer of its own; both must be built around it.

pgvector puts vectors in PostgreSQL. It is the right answer when the vectors belong
alongside relational data that already lives there, and when operational simplicity
matters more than raw speed. Managed services such as Pinecone and Weaviate remove the
operational burden entirely at a recurring cost.

Keeping the store behind a narrow interface -- add, search, delete, count -- means
this choice can be revisited without touching the retrieval logic above it. The
interface is four methods, and the cost of defining it up front is close to zero.

## Metadata filtering

Filtering before the vector search is what makes per-document queries possible. A
filter on document identity restricts the candidate set, and the approximate search
then runs only over what survives. Filtering after the search instead would return
fewer results than requested whenever the filter excluded a top match, which is a
subtle and frequently shipped bug.
