# Chapter 2: Theoretical Background

## 2.1 Retrieval-Augmented Generation (RAG)

### 2.1.1 Fundamental Architecture

The foundational architecture of RAG consists of three core components: a retriever, a database of indexed documents, and a generative language model. The retriever operates on dense vector representations, where both queries and documents are mapped into a shared embedding space $\mathbb{R}^d$. Given a query vector $\mathbf{q} \in \mathbb{R}^d$ and a document vector $\mathbf{d} \in \mathbb{R}^d$, the retrieval mechanism computes relevance scores based on vector similarity.

The retrieval process can be formalized as follows. Given a corpus $\mathcal{D} = \{d_1, d_2, \ldots, d_n\}$ and a query $q$, the system first encodes both into dense vectors using an embedding function $\phi: \text{text} \rightarrow \mathbb{R}^d$. The retrieval subsystem then ranks documents by computing similarity scores:

$$\text{score}(q, d_i) = \text{sim}(\phi(q), \phi(d_i)) = \frac{\phi(q) \cdot \phi(d_i)}{\|\phi(q)\| \|\phi(d_i)\|}$$

This cosine similarity measure ranges from -1 to 1, where 1 indicates perfect alignment and -1 indicates perfect opposition. In practice, embeddings are trained to occupy regions of the hypersphere that correspond to semantic relatedness.

The retrieved top-$k$ documents $\{d_{i_1}, d_{i_2}, \ldots, d_{i_k}\}$ are concatenated and provided as context to the language model, which conditions its output on both the query and the retrieved passages. This architecture decoupling allows knowledge storage to remain external to the model's parameters, enabling updates without retraining.

### 2.1.2 Limitations of Dense Retrieval

Despite its effectiveness for factoid queries, dense retrieval suffers from three fundamental architectural constraints that become apparent under specific query conditions.

The first limitation concerns **context fragmentation**. When documents are split into fixed-length chunks (typically 300-600 tokens), the chunking process inadvertently severs long-range dependencies. Entities discussed across distant sections of a source document become isolated in separate vector representations. Consider a scenario where document $D$ describes entity $A$ in paragraph 1 and entity $B$ in paragraph 10. If $A$ and $B$ share a semantic relationship, that relationship is lost because the chunks containing each entity are indexed independently. The retriever can only match chunks based on surface-level similarity to the query string, not based on inter-chunk relationships.

The second limitation involves **multi-hop reasoning failure**. Many cognitively straightforward queries require chaining information across multiple documents. For instance, answering "What is the relationship between company $X$ and person $Y$?" might require traversing a path $X \rightarrow Z \rightarrow Y$, where $Z$ serves as an intermediary. Cosine similarity between the query and any single chunk containing $Z$ is typically insufficient to surface that chunk, as the query mentions $X$ and $Y$ but not $Z$. The flat topology of the embedding space provides no mechanism for representing or traversing these relational chains.

The third limitation manifests as **summarization blindness**. Queries that ask for corpus-level synthesis—such as "What are the main themes discussed in these documents?"—contain no specific lexical anchors for the retriever to match. Without keyword signals, the system defaults to returning noisy or unrepresentative passages, as the similarity computation has nothing substantial to optimize against.

These failure modes are not hyperparameters that can be tuned away; they are structural consequences of the flat vector representation.

### 2.1.3 Advanced RAG Variants

The research community has developed several advanced RAG variants addressing these limitations through architectural innovations.

**HyDE (Hypothetical Document Embeddings)** generates hypothetical answer documents from the query and uses these as retrieval targets, effectively performing inverse retrieval. **Reranking** pipelines employ cross-encoders to re-score initially retrieved passages. **Self-RAG** introduces reflective mechanisms where the language model evaluates retrieval necessity and relevance before generation.

However, these approaches remain constrained within the vector space paradigm and do not fundamentally address the relational architecture problem.

## 2.2 Knowledge Graphs as Structured Knowledge Representation

### 2.2.1 Graph Formalism

Knowledge graphs offer an alternative representational substrate that preserves relational structure. Formally, a knowledge graph is defined as a directed graph $G = (V, E)$ where:

- $V = \{v_1, v_2, \ldots, v_n\}$ is the vertex set representing entities
- $E \subseteq V \times V$ is the edge set representing relationships

Each vertex $v_i$ carries attributes $\text{attr}(v_i) = (\ell_i, \tau_i, d_i)$ where $\ell_i$ is the canonical label, $\tau_i$ is the entity type, and $d_i$ is a natural language description. Each edge $e_{ij} = (v_i, v_j, r_{ij})$ carries a relationship type $r_{ij}$ and potentially edge-specific attributes.

The critical distinction from vector embeddings lies in how similarity is computed. In embedding space, semantic proximity is a function of geometric distance in the high-dimensional hypersphere. In graph space, proximity is topological—it is determined by the number of edges on the shortest path connecting two vertices:

$$\text{dist}_G(v_i, v_j) = \text{shortest\_path_length}(v_i, v_j)$$

This topological notion of distance enables reasoning about relationships that are implicit in the data but never explicitly stated in any single document.

### 2.2.2 Advantages for Retrieval

Graph-structured knowledge provides three principal advantages for retrieval-augmented generation.

**Explicit multi-hop reasoning** becomes possible through graph traversal algorithms. Given anchor vertices identified in a query, the system can traverse $k$-hop neighborhoods $\mathcal{N}_k(v)$ to discover indirect relationships. Path-finding algorithms (BFS, Dijkstra, A*) operate on the explicit edge structure rather than approximating relationships through vector similarity.

**Hierarchical abstraction** emerges naturally from community detection. Vertices cluster into communities based on edge density, and these communities themselves form hierarchical structures. This enables retrieval at multiple granularities—fine-grained entity-level for specific queries or coarse-grained community-level for thematic synthesis.

**Explainability** derives from the interpretable edge structure. The path from query entity to answer entity can be traced and presented to users, providing attribution that vector-based retrieval cannot match.

## 2.3 Community Detection and Modularity Optimization

### 2.3.1 The Community Detection Problem

Community detection seeks to partition a graph into internally dense, externally sparse subgroups. Formally, given a graph $G = (V, E)$ with adjacency matrix $A$ where $A_{ij} = 1$ if $(v_i, v_j) \in E$ (and weighted otherwise), the goal is to find a partition $\mathcal{P} = \{C_1, C_2, \ldots, C_k\}$ of $V$ such that intra-community edge density significantly exceeds inter-community density.

### 2.3.2 Modularity Optimization

The most widely used quality function for community detection is **modularity** $Q$, introduced by Newman and Girvan (2004). For a partition $\mathcal{P}$, modularity is defined as:

$$Q(\mathcal{P}) = \frac{1}{2m} \sum_{ij} \left( A_{ij} - \frac{k_i k_j}{2m} \right) \delta(c_i, c_j)$$

where:
- $m = \frac{1}{2} \sum_{ij} A_{ij}$ is the total edge weight
- $k_i = \sum_j A_{ij}$ is the degree of vertex $i$
- $c_i$ is the community assignment of vertex $i$
- $\delta(c_i, c_j) = 1$ if $c_i = c_j$, else 0$

The term $\frac{k_i k_j}{2m}$ represents the expected number of edges between vertices $i$ and $j$ in a random graph with the same degree sequence. Thus, $Q$ measures the deviation of the actual graph from this random baseline—positive values indicate communities denser than expected by chance.

Maximizing $Q$ is NP-hard, motivating the development of heuristic algorithms.

### 2.3.3 The Louvain Algorithm

The Louvain algorithm (Blondel et al., 2008) maximizes modularity through a greedy two-phase iterative process:

**Phase 1 - Local Moving**: For each vertex $v$, evaluate the modularity gain $\Delta Q$ of moving $v$ to each neighboring community. Move $v$ to the community yielding maximum positive gain, repeating until no improvement is possible.

**Phase 2 - Aggregation**: Contract the graph by merging vertices within the same community into super-vertices, with edge weights between super-vertices equal to the sum of edges between their constituent vertices. Return to Phase 1 on the contracted graph.

The algorithm proceeds iteratively until convergence, producing a hierarchical decomposition where the finest partition is the result of the first pass and coarser partitions emerge from subsequent aggregations.

However, Traag et al. (2019) proved that Louvain can produce communities that are internally disconnected—up to 25% of detected communities exhibit this pathology. This occurs because Louvain optimizes modularity without enforcing connectivity constraints, leading to partitions where a community contains multiple disconnected components.

### 2.3.4 The Leiden Algorithm

The Leiden algorithm (Traag et al., 2019) addresses Louvain's deficiencies through a three-phase approach that guarantees well-connected communities:

**Phase 1 - Local Moving**: Identical to Louvain—vertices move to neighboring communities maximizing modularity gain.

**Phase 2 - Refinement**: Unlike Louvain, Leiden performs an additional refinement step where communities are split to ensure internal connectivity. Each vertex is initially assigned to its own community, then vertices are merged if this improves modularity while maintaining connectivity.

**Phase 3 - Aggregation**: The refined partition is contracted into a new network for the next iteration, exactly as in Louvain.

The key innovation is that Leiden's refinement phase ensures every community in the output partition is guaranteed to be connected—a property Louvain cannot ensure. Empirically, Leiden produces partitions with modularity at least as high as Louvain while guaranteeing connectivity.

### 2.3.5 Mathematical Properties

The Leiden algorithm satisfies several important mathematical properties:

**Connectivity Guarantee**: For any partition produced, every community $C \in \mathcal{P}$ is connected—every vertex can reach every other vertex within $C$ via paths that remain entirely within $C$.

**Modularity Bounds**: The modularity of Leiden's output satisfies $Q_{\text{Leiden}} \geq Q_{\text{Louvain}}$.

**Convergence**: The algorithm converges in finite time because modularity increases monotonically and is bounded above by 1.

**Resolution Limit**: Like all modularity-based methods, Leiden suffers from a resolution limit—it cannot detect communities smaller than a scale determined by total edge density. Communities below this threshold merge with neighboring communities regardless of internal structure.

## 2.4 GraphRAG Variants and Extensions

### 2.4.1 LightRAG

LightRAG (Guo et al., 2024) addresses computational cost through two innovations: dual-level retrieval and incremental updates.

**Dual-level retrieval** operates simultaneously at entity level (specific facts) and topic level (thematic summaries). This allows the system to route queries to the appropriate granularity—entity-level for specific questions, topic-level for thematic queries.

**Incremental updates** allow the graph to be extended without full re-indexing. When new documents arrive, only affected vertices, edges, and communities are updated, rather than rebuilding the entire structure.

### 2.4.2 HippoRAG

HippoRAG (Gutiérrez et al., 2024) draws on hippocampal indexing theory from neuroscience. The system models retrieval as an analog of how the hippocampus processes and retrieves memories:

- A pattern separator (like the dentate gyrus) encodes incoming information into distinctive representations
- Personalized PageRank over the knowledge graph (like CA3 recurrent connections) retrieves contextually relevant subgraphs

This neurobiological framing provides theoretical grounding for why graph-based retrieval succeeds on multi-hop reasoning tasks.

### 2.4.3 SubgraphRAG

SubgraphRAG (Li et al., 2024) decomposes the retrieval-generation pipeline into distinct stages:

1. **Retrieval Stage**: Extract relevant subgraphs from the knowledge graph based on query matching
2. **Reasoning Stage**: Apply graph neural network reasoning over the extracted subgraph to produce answers

This two-stage approach sacrifices global comprehensiveness for locally grounded precision.

### 2.4.4 Comparative Summary

| System | Graph Structure | Retrieval Strategy | Update Mechanism |
|--------|----------------|-------------------|-----------------|
| GraphRAG | Hierarchical communities (Leiden) | Local + Global (Map-Reduce) | Full re-indexing |
| LightRAG | Dual-level (entity + topic) | Low-level + High-level | Incremental |
| HippoRAG | Flat KG + PageRank | Personalized PageRank | Append-only |
| SubgraphRAG | Subgraph extraction | Two-stage (retrieve + reason) | Subgraph update |

## 2.5 Mathematical Foundations Summary

The theoretical foundation of GraphRAG rests on four mathematical pillars:

1. **Dense Vector Retrieval**: Cosine similarity in high-dimensional spaces enables semantic matching but loses relational structure.

2. **Graph Theory**: Topological distance in knowledge graphs captures multi-hop relationships that vector spaces cannot represent.

3. **Modularity Optimization**: The Leiden algorithm provides rigorous community detection with connectivity guarantees, enabling hierarchical abstraction.

4. **Map-Reduce Paradigm**: Distributed computation over community summaries enables corpus-wide synthesis without context window limitations.

These foundations combine to create a retrieval architecture capable of global sensemaking—a capability fundamentally unavailable to vector-only approaches.
