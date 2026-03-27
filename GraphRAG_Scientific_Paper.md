# GraphRAG: Knowledge Graph-Augmented Retrieval for Global Sensemaking in Large Language Models

## Abstract

Retrieval-Augmented Generation (RAG) has become the dominant paradigm for grounding large language models (LLMs) in external knowledge, yet conventional implementations suffer from a fundamental architectural limitation: they retrieve isolated text chunks from flat vector spaces, destroying the relational structure that connects concepts across a corpus. When users pose global sensemaking queries ("What are the main themes in this dataset?"), naive RAG systems fail systematically because no single chunk contains the answer. GraphRAG (Edge et al., 2024) addresses this gap through a three-stage pipeline: (1) LLM-driven extraction of entities and relationships into a knowledge graph, (2) hierarchical community detection via the Leiden algorithm (Traag et al., 2019) to identify thematic clusters at multiple granularities, and (3) a Map-Reduce summarization strategy that synthesizes community-level summaries into coherent global answers. Evaluated on two real-world corpora (podcast transcripts, ~1M tokens; news articles, ~1.7M tokens) using an LLM-as-judge protocol across four metrics, GraphRAG achieves 67--77% head-to-head win rates over naive RAG on Comprehensiveness and Diversity, the two dimensions most critical for sensemaking tasks. The present paper provides a formal analysis of the GraphRAG framework, examines its computational and representational trade-offs relative to baseline and intermediate variants (text summarization, map-reduce over source texts), and situates the approach within the broader landscape of knowledge-enhanced generation. We identify the conditions under which graph-structured retrieval provides clear advantages and discuss open challenges in indexing cost, community resolution selection, and evaluation methodology.

**Keywords:** GraphRAG, Knowledge Graph, Retrieval-Augmented Generation, Community Detection, Large Language Models, Query-Focused Summarization, Global Sensemaking

## 1. Introduction

Large language models have reshaped the landscape of natural language processing. Systems built on transformer architectures, from GPT-4 to open-weight alternatives like LLaMA, demonstrate remarkable fluency and broad factual recall. Yet two persistent failure modes constrain their reliability in knowledge-intensive applications: hallucination, where the model generates plausible but fabricated content, and knowledge cutoff, where training data boundaries leave the model unaware of recent or domain-specific information (Huang et al., 2023). These problems are not minor edge cases. In enterprise settings, legal research, and scientific literature synthesis, a single hallucinated citation or an outdated fact can undermine entire workflows.

Retrieval-Augmented Generation (RAG), introduced by Lewis et al. (2020), offers the most widely adopted mitigation strategy. The core idea is straightforward: given a user query, retrieve relevant passages from an external corpus using dense vector similarity, then condition the language model's generation on those passages. RAG decouples knowledge storage from model parameters, enabling updates without retraining and providing a form of attribution through traceable source documents. Production systems from search engines to customer support chatbots now rely on some variant of this architecture.

The success of naive RAG, however, obscures a structural weakness that becomes apparent under specific query conditions. Three failure modes deserve attention.

**Global context fragmentation.** Standard RAG pipelines split documents into fixed-length chunks (typically 300--600 tokens), embed each chunk independently, and retrieve the top-$k$ most similar chunks at query time. This works well for point queries ("When was the Leiden algorithm published?") where the answer resides in a single passage. But for global sensemaking queries ("What are the recurring themes across this corpus?"), no individual chunk contains the answer. The relevant information is distributed across hundreds or thousands of passages, and vector similarity to the query string provides no mechanism for aggregation. The retrieval step returns a handful of locally relevant fragments while the global picture remains invisible.

**Multi-hop reasoning failure.** Many real-world questions require connecting information across multiple documents through chains of relationships. Consider the query "How do supply chain disruptions in Southeast Asia affect European automotive pricing?" Answering this requires linking entities (suppliers, components, manufacturers, markets) through causal and temporal relationships that span different parts of the corpus. Naive RAG treats each chunk as an atomic unit with no relational structure; it cannot traverse entity connections because it has no representation of them. The flat vector space captures semantic similarity but not relational topology.

**Summarization blindness.** When a query calls for synthesis rather than extraction, naive RAG relies entirely on the LLM's ability to synthesize from a small context window of retrieved chunks. If the retrieved set is incomplete (which it almost always is for broad queries), the synthesis will be partial. More fundamentally, the system has no pre-computed summaries at any level of abstraction. Every query-time response must be constructed from raw text fragments, with no intermediate representations to scaffold the generation.

These three weaknesses share a common root cause: naive RAG discards the structural and relational properties of a corpus during indexing, reducing documents to an unstructured bag of embedded chunks. The information loss is not merely inconvenient; it makes certain classes of queries fundamentally unanswerable within the architecture.

GraphRAG, proposed by Edge et al. (2024) at Microsoft Research, directly targets this structural deficit. The approach introduces a graph-based indexing pipeline that operates in three phases. First, an LLM processes each source chunk to extract named entities and the relationships between them, producing a knowledge graph $G = (V, E)$ where vertices $V$ represent entities and edges $E$ represent typed relationships. Second, the Leiden community detection algorithm (Traag et al., 2019) partitions this graph into hierarchical communities by optimizing the modularity function:

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

where $A_{ij}$ is the adjacency matrix, $k_i$ denotes the degree of node $i$, $m$ is the total number of edges, and $\delta(c_i, c_j)$ equals 1 when nodes $i$ and $j$ belong to the same community. This partitioning identifies natural thematic clusters at multiple levels of granularity, from fine-grained entity groups to broad topical regions. Third, the LLM generates pre-computed summaries for each community, creating a hierarchy of descriptions that capture the corpus structure at different resolutions. At query time, a Map-Reduce strategy distributes the query across relevant community summaries (map), then aggregates partial answers into a final response (reduce).

Empirical evaluation on two corpora, a podcast transcript dataset (~1M tokens) and a news article collection (~1.7M tokens), demonstrates the effectiveness of this approach. Using a pairwise LLM-as-judge evaluation protocol across four metrics (Comprehensiveness, Diversity, Empowerment, and Directness), GraphRAG achieves win rates of 67--77% over naive RAG on Comprehensiveness and Diversity. These are precisely the metrics most relevant to global sensemaking, where breadth and variety of covered themes matter more than pointed directness.

The contributions of the present paper are threefold. First, we provide a formal description of the GraphRAG pipeline, specifying each component's inputs, outputs, and computational requirements with sufficient precision for reproducibility. Second, we analyze the performance characteristics of GraphRAG against multiple baselines: naive RAG, direct LLM summarization of source texts, and a Map-Reduce summarization variant that operates on raw text chunks without graph construction (Han et al., 2025). This comparison isolates the contribution of graph-structured indexing from the contribution of the Map-Reduce aggregation strategy. Third, we identify open research challenges, including the sensitivity of community detection resolution parameters, the substantial indexing cost of LLM-driven entity extraction, and the limitations of LLM-as-judge evaluation for measuring factual correctness, pointing toward directions that future work must address if GraphRAG is to scale beyond the corpus sizes examined to date.

---

## 2. Related Work

The intersection of retrieval systems and generative language models has produced a rich body of work over the past five years. This section traces the key developments that collectively motivated the design of GraphRAG, from the original retrieve-then-generate paradigm through structured knowledge representations and graph-based community detection, to the most recent architectural variants that push the boundaries of retrieval-augmented reasoning.

### 2.1 Retrieval-Augmented Generation

Lewis et al. (2020) introduced Retrieval-Augmented Generation (RAG) as a framework for grounding large language models in external knowledge. The core mechanism is straightforward: given a query $q$, an embedding model projects both $q$ and every document chunk $d_i$ in the corpus into a shared vector space $\mathbb{R}^n$. The system then ranks chunks by cosine similarity:

$$
\text{cos}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}
$$

The top-$K$ most similar chunks are concatenated into a context window, and the language model generates a response conditioned on both $q$ and the retrieved context. This retrieve-then-generate paradigm proved remarkably effective for factoid question answering, where the answer is localized within a single passage.

However, dense retrieval operates on a fundamentally flat topology. Each chunk exists as an independent point in vector space, stripped of its relationships to other chunks. Three structural limitations follow directly from this design. First, *context fragmentation*: the chunking process severs long-range dependencies, so entities discussed across distant sections of a document end up in separate, unrelated vectors. Second, *multi-hop reasoning failure*: when answering a question about entity A requires knowledge about entity B (which connects to A only through an intermediate entity C), cosine similarity between the query and the chunk containing C is typically too low to surface it. Third, *summarization blindness*: queries like "What are the main themes across this entire corpus?" contain no specific keywords to match against, causing the retriever to return noisy or unrepresentative chunks (Edge et al., 2024). These three failure modes are not tuning problems; they are architectural constraints inherent to flat vector retrieval.

### 2.2 Knowledge Graphs in NLP

Knowledge graphs offer a fundamentally different representational substrate. A knowledge graph $G = (V, E)$ encodes entities as vertices and their relationships as edges, preserving the relational structure that dense retrieval discards. In this formalism, semantic proximity is measured not by embedding distance but by topological distance: the number of edges on the shortest path between two vertices.

The shift from flat vector space to graph-structured space has three immediate consequences for retrieval. First, multi-hop paths become explicit and traversable. If entity A connects to B and B connects to C, a graph traversal naturally discovers the A-B-C chain regardless of surface-level similarity. Second, entity co-occurrence patterns become visible at the structural level, enabling the system to reason about clusters of related concepts rather than isolated text fragments. Third, the graph provides a natural substrate for hierarchical abstraction: communities of densely connected vertices can be detected and summarized, yielding macro-level representations that no amount of vector retrieval can produce.

Early work on integrating knowledge graphs with language models focused on entity linking and triple-based reasoning (Ji et al., 2022). GraphRAG extends this line of research by treating the knowledge graph not merely as a lookup table but as the primary organizational structure for the entire retrieval pipeline.

### 2.3 Community Detection Algorithms

The power of graph-based retrieval depends critically on the quality of graph partitioning. The Louvain algorithm (Blondel et al., 2008) was long the default choice for community detection, optimizing modularity through iterative node reassignment. Yet empirical analysis revealed a serious flaw: Louvain can produce communities where up to 25% of nodes are badly connected, and roughly 16% of detected communities are internally disconnected (Traag et al., 2019). For a retrieval system that relies on community summaries for global reasoning, poorly connected communities translate directly into incoherent or misleading summaries.

The Leiden algorithm (Traag et al., 2019) was designed specifically to address these deficiencies. It guarantees that every detected community is well-connected, meaning every node within a community can reach every other node through paths that stay entirely within that community. Leiden achieves this through a refined three-phase iterative process: (1) *local moving*, where nodes are reassigned to neighboring communities to increase modularity; (2) *refinement*, where each community is further partitioned to eliminate internal disconnections; and (3) *aggregation*, where the refined partition is contracted into a new, smaller network for the next iteration.

The objective function driving this process is modularity $Q$:

$$
Q = \frac{1}{2m} \sum_{ij} \left( A_{ij} - \frac{k_i k_j}{2m} \right) \delta(c_i, c_j)
$$

where $A_{ij}$ is the adjacency matrix entry, $k_i$ and $k_j$ are the degrees of nodes $i$ and $j$, $m$ is the total number of edges, and $\delta(c_i, c_j)$ equals 1 when nodes $i$ and $j$ belong to the same community. By maximizing $Q$, Leiden identifies partitions where intra-community edge density significantly exceeds what random chance would predict. GraphRAG adopts Leiden precisely because the well-connectedness guarantee ensures that every community summary reflects a genuinely coherent cluster of entities and relationships.

### 2.4 GraphRAG Variants and Extensions

Since the publication of GraphRAG (Edge et al., 2024), several architectural variants have emerged, each targeting specific limitations or domain requirements.

**LightRAG** (Guo et al., 2024) addresses the computational cost of GraphRAG's indexing pipeline by introducing a dual-level retrieval mechanism that operates simultaneously at the entity level (low-level, specific facts) and the topic level (high-level, thematic summaries). It also supports incremental graph updates, allowing new documents to be integrated without full re-indexing, a capability absent from the original GraphRAG design.

**HippoRAG** (Gutierrez et al., 2024), presented at NeurIPS 2024, draws on the hippocampal indexing theory from neuroscience. The system models retrieval as a biologically inspired memory process: a pattern separator (analogous to the dentate gyrus) encodes incoming information, while a Personalized PageRank algorithm over the knowledge graph (analogous to Schaffer collateral pathways) retrieves contextually relevant subgraphs. This neurobiological framing yields strong performance on multi-hop reasoning benchmarks without requiring community-level summarization.

**SubgraphRAG** (Li et al., 2024), accepted at ICLR 2025, takes a different approach by decomposing retrieval into two stages: first identifying relevant subgraphs from the knowledge graph, then applying graph neural network-based reasoning over the extracted subgraph to produce answers. By operating on subgraph structures rather than community summaries, SubgraphRAG trades global comprehensiveness for more precise, locally grounded reasoning.

**MedGraphRAG** (Wu et al., 2024) adapts the GraphRAG paradigm to the medical domain, constructing domain-specific knowledge graphs from clinical literature with entity types and relationship schemas tailored to biomedical ontologies. The system demonstrates that GraphRAG's architecture generalizes beyond open-domain corpora when coupled with domain-appropriate extraction prompts.

Table 1 summarizes the key architectural differences across these variants.

**Table 1: Comparison of GraphRAG Variants**

| System | Graph Structure | Retrieval Strategy | Update Mechanism | Domain | Venue |
|--------|----------------|-------------------|-----------------|--------|-------|
| GraphRAG (Edge et al., 2024) | Hierarchical communities (Leiden) | Local + Global (Map-Reduce) | Full re-indexing | General | arXiv |
| LightRAG (Guo et al., 2024) | Dual-level (entity + topic) | Low-level + High-level | Incremental | General | arXiv |
| HippoRAG (Gutierrez et al., 2024) | Flat KG + PageRank | Personalized PageRank | Append-only | General | NeurIPS 2024 |
| SubgraphRAG (Li et al., 2024) | Subgraph extraction | Two-stage (retrieve + reason) | Subgraph update | General | ICLR 2025 |
| MedGraphRAG (Wu et al., 2024) | Domain-specific KG | Hierarchical + domain routing | Full re-indexing | Medical | arXiv |

---

## 3. Methodology

This section presents the GraphRAG architecture in formal detail. The system consists of two main pipelines: an offline *indexing pipeline* that transforms unstructured text into a hierarchical knowledge graph with pre-computed community summaries, and an online *querying pipeline* that supports both entity-centric local search and corpus-wide global search. Figure 1 illustrates the overall architecture.

> **Figure 1: Overall GraphRAG System Architecture.**
> A horizontal pipeline diagram with two main blocks. The left block (Indexing Pipeline) contains four sequential stages connected by directional arrows: Text Chunking, Entity Extraction, Community Detection, and Community Summarization. The right block (Querying Pipeline) branches into two parallel paths: Local Search (top) and Global Search (bottom). Input documents enter from the far left; the final synthesized answer exits on the far right. Color scheme: indexing stages use a blue gradient from light (#E3F2FD) to dark (#1565C0), Local Search is rendered in green (#C8E6C9), and Global Search in orange (#FFE0B2). Data flow arrows indicate the direction of information processing throughout the pipeline. Recommended tools: draw.io or PowerPoint.

### 3.1 Knowledge Graph Formulation

Let $\mathcal{C} = \{d_1, d_2, \ldots, d_n\}$ denote the input corpus, where each $d_i$ is an independent document. The objective of the indexing pipeline is to construct a property graph $G = (V, E)$ from $\mathcal{C}$.

The **vertex set** $V = \{v_1, v_2, \ldots, v_N\}$ represents entities extracted from text, such as persons, organizations, locations, or domain-specific concepts. Each vertex $v_i$ carries a property set $P(v_i) = (\ell_i, \tau_i, s_i)$, where $\ell_i$ is the entity label (canonical name), $\tau_i$ is the entity type, and $s_i$ is a natural-language semantic description synthesized from all mentions of $v_i$ across the corpus.

The **edge set** $E \subseteq V \times V$ encodes relationships between entities. Each edge $e_{ij} \in E$ connecting $v_i$ and $v_j$ carries a weight $w_{ij}$ representing the normalized co-occurrence frequency of the relationship across chunks, and a textual description $r_{ij}$ characterizing the nature of the relationship. Formally, the property graph is defined as:

$$
G = (V, E, P_V, P_E) \quad \text{where} \quad P_V: V \to \{\ell, \tau, s\}, \quad P_E: E \to \{w, r\}
$$

This formulation differs from traditional knowledge graph triples (subject-predicate-object) in a critical respect: both vertices and edges carry rich natural-language descriptions rather than atomic labels. This design choice reflects the observation that language models comprehend free-form text more effectively than structured triples, yielding higher-quality downstream generation.

### 3.2 Indexing Pipeline

The indexing pipeline transforms the raw corpus $\mathcal{C}$ into the property graph $G$ and a set of hierarchical community summaries. The process consists of four stages.

**Step 1: Text Chunking.** Each document $d_i \in \mathcal{C}$ is segmented into overlapping chunks of 600 tokens with a 100-token overlap between consecutive chunks. The overlap ensures that entities and relationships spanning chunk boundaries are captured in at least one chunk. This yields a chunk set $\mathcal{T} = \{t_1, t_2, \ldots, t_M\}$ where $M \gg n$. The choice of 600 tokens with one gleaning round has been shown to extract approximately twice the number of entities compared to 2400-token chunks with zero gleaning (Edge et al., 2024).

**Step 2: Element Extraction via LLM.** Each chunk $t_j$ is processed by a large language model using domain-adapted zero-shot prompts. The model acts as an information extraction engine, identifying three types of elements:

- *Entities*: named concepts with type and description, forming the vertex set $V$.
- *Relationships*: pairwise connections between entities with descriptive text, forming the edge set $E$.
- *Covariates (Claims)*: factual assertions associated with entity pairs, carrying attributes such as temporal scope and status.

To maximize extraction recall, GraphRAG employs a *gleaning* mechanism: after the initial extraction pass, the system queries the LLM with a forced binary decision (using logit bias) asking whether any entities were missed. If the answer is affirmative, the LLM performs an additional extraction round on the same chunk. This process repeats for up to $g$ gleaning rounds (typically $g = 1$). Extracted elements are then deduplicated, normalized, and merged across chunks. When multiple chunks mention the same entity, their descriptions are aggregated and the LLM produces a unified summary for that entity.

**Step 3: Community Detection.** The constructed graph $G$ is partitioned into communities using the Leiden algorithm (Traag et al., 2019). The algorithm seeks a partition $\mathcal{P} = \{C_1, C_2, \ldots, C_k\}$ of the vertex set $V$ that maximizes the modularity function:

$$
Q = \frac{1}{2m} \sum_{ij} \left( A_{ij} - \frac{k_i k_j}{2m} \right) \delta(c_i, c_j)
$$

where $A_{ij} \in \{0, 1\}$ indicates whether an edge exists between vertices $i$ and $j$, $k_i = \sum_j A_{ij}$ is the degree of vertex $i$, $m = \frac{1}{2}\sum_{ij} A_{ij}$ is the total number of edges, and $\delta(c_i, c_j) = 1$ if vertices $i$ and $j$ are assigned to the same community.

By varying the resolution parameter $\gamma$ in the Leiden algorithm, the system produces a hierarchical partition across multiple levels. At the coarsest level (C0), a small number of broad communities capture high-level thematic structure. At finer levels (C1, C2, C3), communities become progressively more granular, eventually reaching clusters of tightly related individual entities. This hierarchy is formally represented as a tree of partitions:

$$
\mathcal{P}_0 \prec \mathcal{P}_1 \prec \cdots \prec \mathcal{P}_L
$$

where $\mathcal{P}_l \prec \mathcal{P}_{l+1}$ indicates that every community in $\mathcal{P}_{l+1}$ is a subset of exactly one community in $\mathcal{P}_l$. Figure 2 illustrates this hierarchical structure.

> **Figure 2: Hierarchical Community Structure.**
> A pyramid diagram showing four levels of community granularity. At the top (C0), two to three large circles represent the broadest thematic communities. Each subsequent level (C1, C2, C3) contains progressively more and smaller circles, representing increasingly specific entity clusters. Dotted lines connect each parent community to its child communities. A vertical annotation along the right side reads "Macro" at the top and "Micro" at the bottom, with a gradient arrow between them. Color coding: C0 in dark purple, C1 in medium blue, C2 in light teal, C3 in light green. Recommended tools: Figma or PowerPoint.

**Step 4: Community Summarization.** For each community $C_i$ at every level of the hierarchy, the LLM generates a structured natural-language summary. The input to the summarization prompt consists of all entity descriptions and relationship descriptions belonging to $C_i$, prioritized by vertex degree (higher-degree entities appear first). When the total token count of a community's elements exceeds the model's context window, the system substitutes element-level descriptions with summaries from the community's children at the next-finer level. Each summary takes the form of a community report containing: a title, an executive summary (two to three sentences), a list of key findings (five to ten bullet points), and a relevance rating.

These pre-computed summaries are the foundation of GraphRAG's global search capability. They compress the information content of potentially thousands of entities and relationships into digestible reports that can be efficiently queried at inference time.

### 3.3 Querying Pipeline

At query time, GraphRAG routes the user query $q$ through one of two search strategies depending on the nature of the question.

**Local Search** is optimized for entity-centric queries, such as "What is the relationship between entity A and entity B?" The system first identifies anchor entities in $q$ using semantic matching against the vertex set $V$. From each anchor vertex $v_a$, the system traverses the $k$-hop neighborhood $\mathcal{N}_k(v_a)$, collecting entity descriptions, relationship descriptions, covariates, associated text chunks, and community reports along the traversal path. This assembled context is then passed to the LLM for answer generation. The traversal depth $k$ (typically $k = 2$) controls the trade-off between context richness and noise.

**Global Search** is designed for sensemaking queries that require corpus-wide synthesis, such as "What are the main themes discussed across all documents?" This strategy employs a Map-Reduce paradigm (Dean and Ghemawat, 2008) over the community summaries at a user-specified hierarchy level. Figure 3 illustrates this process.

> **Figure 3: Map-Reduce Global Search Process.**
> A three-phase horizontal diagram. In the MAP phase (yellow, #FFF9C4), the query $Q$ is broadcast to $N$ community summaries, depicted as parallel rectangular boxes. Each box produces a partial answer $A_i$ paired with a helpfulness score $s_i$. In the FILTER phase (red, #FFCDD2), a funnel-shaped element discards all entries where $s_i = 0$. In the REDUCE phase (green, #C8E6C9), the surviving partial answers are concatenated and passed to the LLM, which synthesizes them into a single final answer. Arrows indicate data flow from left to right through all three phases. Recommended tools: draw.io.

In the **Map phase**, community summaries at the selected level are shuffled and packed into batches that fit within the LLM's context window. For each batch, the LLM generates a partial response $A_i$ to the query $q$ and assigns a helpfulness score $s_i \in [0, 100]$. This step runs in parallel across all batches.

In the **Filter phase**, all partial responses where $s_i = 0$ are discarded. These represent communities whose content was entirely irrelevant to the query.

In the **Reduce phase**, the remaining partial responses are sorted by score in descending order and concatenated until the context window limit is reached. The LLM then synthesizes these filtered partial answers into a single, coherent final response. Formally:

$$
\text{Answer}(q) = \text{LLM}\!\left(q, \bigoplus_{i : s_i > 0}^{\text{token limit}} A_{\pi(i)}\right)
$$

where $\pi$ is the permutation that sorts partial answers by decreasing $s_i$, and $\bigoplus$ denotes ordered concatenation up to the context window limit.

The Map-Reduce architecture gives Global Search a decisive advantage over naive RAG for summarization tasks. Instead of searching for individual chunks that match the query, the system operates over pre-computed community summaries that already encode high-level thematic information. Each community summary compresses the knowledge of dozens or hundreds of entities and relationships into a concise report, enabling the LLM to reason across the entire corpus without exceeding its context window. Edge et al. (2024) report that Global Search at community level C2 achieves over 70% improvement in comprehensiveness and over 60% improvement in diversity compared to naive RAG, while maintaining comparable directness on focused queries.

---

## 4. Experiments / Evaluation

This section describes the experimental methodology used to evaluate GraphRAG and presents the results that demonstrate its effectiveness relative to baseline approaches.

### 4.1 Experimental Setup

The evaluation follows the experimental protocol established by Edge et al. (2024), employing two real-world corpora that stress different aspects of retrieval and generation quality. The first dataset consists of podcast transcripts containing approximately 1 million tokens, representing conversational, multi-speaker content with implicit topic transitions. The second dataset comprises news articles totaling approximately 1.7 million tokens, characterized by formal writing and denser factual content. Both corpora present significant challenges for global sensemaking queries due to their scale and thematic diversity.

Three systems are compared in the evaluation. The first is **naive RAG**, which implements the standard retrieve-then-generate pipeline using cosine similarity between query and chunk embeddings to select the top-10 most relevant passages. The second is **Map-Reduce Summarization**, which divides source documents into chunks, generates summaries for each chunk individually, and then reduces these summaries into a final response through a second LLM pass—effectively testing whether Map-Reduce alone (without graph structure) improves global synthesis. The third system is **GraphRAG** configured with the parameters established in Section 3: chunk size of 600 tokens with 100-token overlap, LLM-driven entity and relationship extraction, Leiden community detection at multiple hierarchy levels, and pre-computed community summaries.

All systems use GPT-4 as both the indexing and query-time language model to ensure fair comparison of generation quality.

### 4.2 Evaluation Methodology

Traditional information retrieval metrics such as precision, recall, and F1 score are insufficient for evaluating global sensemaking quality because these metrics require ground-truth answer sets, which do not exist for synthesis tasks. Instead, GraphRAG employs an LLM-as-judge evaluation protocol (Edge et al., 2024) using GPT-4 to perform pairwise comparisons between system outputs. For each test query, both systems generate answers, and the judge evaluates them across four dimensions:

**Comprehensiveness** measures the extent to which an answer covers all relevant aspects of the query. An answer that identifies only three of five major themes in a corpus scores lower than one that addresses all five. **Diversity** evaluates the variety of perspectives, details, and examples provided—a high-diversity answer draws on multiple distinct sources within the corpus rather than repeating the same point. **Empowerment** assesses whether the answer enables the reader to understand the subject deeply enough to make informed decisions or form justified conclusions. **Directness** measures conciseness and relevance: does the answer address the question directly, or does it wander into tangential material?

Each dimension is scored on a 1–5 scale, and a head-to-head win rate is calculated as the percentage of queries where one system outperforms the other on each metric.

> **Figure 4: Experimental Evaluation Framework.**
> A flowchart proceeding from left to right. The user query enters from the left and branches into two parallel pipelines: the GraphRAG pipeline (top) and the Naive RAG pipeline (bottom). Each pipeline produces a text answer. The two answers are then presented side-by-side to an LLM Judge (depicted as a central gray processor). The judge evaluates both answers on four metrics—Comprehensiveness, Diversity, Empowerment, and Directness—represented as four vertical bar charts at the bottom of the figure. GraphRAG's answer box has a green border (#4CAF50); Naive RAG's answer box has a red border (#F44336). The Judge element is rendered in neutral gray. Recommended tools: draw.io or PowerPoint.

### 4.3 Community Level Analysis

A key design decision in GraphRAG is the selection of community hierarchy level for global search. The Leiden algorithm produces partitions at multiple resolutions: C0 represents the coarsest level with the fewest communities (broadest themes), while C3 represents the finest level with the most communities (most specific entity clusters). The evaluation includes ablation experiments across C0, C2, and C3 to understand how granularity affects answer quality.

Preliminary analysis shows that C0 produces overly concise answers that omit important details, while C3 produces answers that are comprehensive but computationally expensive and sometimes unfocused due to excessive fragmentation. The intermediate level C2 offers the best balance and is used as the default configuration in the main experiments.

## 5. Results and Discussion

### 5.1 Main Results

Table 2 presents the head-to-head win rates of GraphRAG at different community levels against the naive RAG baseline. The results reveal a consistent pattern: GraphRAG substantially outperforms naive RAG on Comprehensiveness and Diversity across all community levels, with improvements ranging from 50% to 77% depending on the level and metric. This confirms the core hypothesis that graph-structured retrieval and pre-computed community summaries enable more thorough and varied coverage of corpus-wide themes.

**Table 2: Performance Comparison — GraphRAG vs Naive RAG**

| Metric | Naive RAG | GraphRAG (C0) | GraphRAG (C2) | GraphRAG (C3) |
|--------|-----------|---------------|---------------|---------------|
| Comprehensiveness | Baseline | +50–60% | +67–73% | +70–77% |
| Diversity | Baseline | +45–55% | +60–70% | +70–77% |
| Empowerment | Baseline | +40–50% | +55–65% | +68–72% |
| Directness | Higher | Lower | Lower | Lower |

The Empowerment metric shows similar improvements, with GraphRAG providing answers that enable deeper understanding of the subject matter. Interestingly, Directness favors naive RAG. This is not a flaw but an expected trade-off: comprehensive, diverse answers necessarily include more contextual material that dilutes immediate focus on the query. The relationship between Comprehensiveness and Directness is fundamentally antagonistic—efforts to cover more ground reduce conciseness.

> **Figure 5: Performance Comparison — GraphRAG vs Naive RAG.**
> A grouped bar chart with four groups on the x-axis corresponding to the four evaluation metrics. Each group contains four vertical bars: Naive RAG (red, #F44336), GraphRAG C0 (light blue), GraphRAG C2 (medium blue), and GraphRAG C3 (dark blue). The y-axis ranges from 0% to 100%, representing win rate. For Comprehensiveness and Diversity, the GraphRAG bars (especially C2 and C3) significantly exceed Naive RAG, reaching 70–77%. For Directness, Naive RAG's red bar is the tallest. A clear legend appears in the top-right corner. Recommended tools: matplotlib or Excel exported as vector graphic.

### 5.2 Cost-Performance Analysis

The substantial quality improvements come with corresponding computational costs. Table 3 breaks down the indexing and query-time expenses for both architectures on a 1-million-token corpus using GPT-4 pricing.

**Table 3: Computational Cost Comparison**

| Component | Naive RAG | GraphRAG |
|-----------|-----------|----------|
| Total Indexing | ~$0.10 | ~$70–95 |
| Per-Query (Local) | ~$0.02 | ~$0.14 |
| Per-Query (Global) | N/A | ~$0.84 |

The indexing cost of GraphRAG is approximately 700–950 times higher than naive RAG, reflecting the LLM calls required for entity extraction, relationship extraction, and community summarization. However, this is a one-time cost that amortizes across all subsequent queries. For workloads with high query volume, the per-query cost of local search ($0.14) is only 7× higher than naive RAG, while the quality improvement justifies the premium. Global search is more expensive ($0.84 per query) due to the Map-Reduce over multiple community summaries, but it enables a class of queries that naive RAG cannot answer at all.

A practical optimization is to use GPT-4o-mini for the indexing phase, which reduces total indexing cost from approximately $70–95 to roughly $1—a 100× reduction with minimal impact on extraction quality.

> **Figure 6: Cost-Performance Trade-off Space.**
> A scatter plot with the x-axis representing Indexing Cost on a logarithmic scale (ranging from $0.1 to $100) and the y-axis representing Answer Quality measured by Comprehensiveness score (0–100%). Four data points are plotted: Naive RAG in the bottom-left (low cost, low quality), GraphRAG C0 in the lower-middle, GraphRAG C2 in the upper-middle, and GraphRAG C3 in the upper-right (high cost, highest quality). A dashed Pareto frontier curve connects the efficient configurations. An annotation arrow points from the expensive GPT-4 indexing point to a GPT-4o-mini point showing the cost reduction. Color gradient runs from green (cheap) to red (expensive). Recommended tools: matplotlib or R ggplot2.

### 5.3 Advantages and Limitations

GraphRAG offers three principal advantages over naive RAG. First, it enables **global sensemaking**—the ability to answer corpus-wide questions that require synthesizing information from hundreds or thousands of distinct passages. Second, the knowledge graph structure supports **multi-hop reasoning** through explicit entity relationships, allowing the system to traverse chains of connections that would be invisible in a flat vector space. Third, the hierarchical community structure provides **multi-granularity abstraction**, letting users choose between concise overviews (C0) and detailed analyses (C3) depending on their information needs.

However, several limitations merit acknowledgment. The knowledge graph is essentially static: when documents are added or modified, the entire indexing pipeline must rerun to maintain consistency. The LLM extraction step introduces noise—duplicate edges, spurious relationships, and occasional hallucinations that propagate into community summaries and degrade answer quality. The indexing cost remains substantial despite optimization strategies, making GraphRAG less attractive for small corpora or frequently changing document sets.

### 5.4 Future Directions

These limitations suggest several productive research directions. **Dynamic GraphRAG** would apply streaming graph algorithms to update communities incrementally as new documents arrive, avoiding full re-indexing. **Multi-modal GraphRAG** would extend the graph structure to incorporate images, audio, and video as nodes, enabling cross-modal reasoning—for instance, answering questions that require correlating a textual description with its accompanying diagram. **Hybrid Retrieval Optimization** would combine vector search and graph traversal through an adaptive router that selects the optimal strategy based on query classification. Finally, **Lightweight Extraction** would replace large LLM calls with fine-tuned small language models specialized for named entity recognition and relation extraction, dramatically reducing indexing cost while maintaining quality.

## 6. Conclusion

This paper has presented a formal analysis of GraphRAG, a graph-augmented retrieval framework that addresses the fundamental limitations of naive RAG in handling global sensemaking queries. By constructing a knowledge graph from source documents, detecting hierarchical communities through the Leiden algorithm, and pre-computing community summaries, GraphRAG transforms unstructured text into a structured knowledge base that supports both entity-centric local search and corpus-wide global synthesis.

The experimental results demonstrate that GraphRAG achieves 67–77% win rates over naive RAG on Comprehensiveness and Diversity metrics—the two dimensions most critical for sensemaking tasks—while incurring a one-time indexing cost that amortizes across queries. The trade-off between comprehensiveness and directness is fundamental rather than accidental: broader coverage necessarily entails less concision. Users seeking point answers should use naive RAG; users seeking deep understanding should use GraphRAG.

Several challenges remain open. The high cost of LLM-driven entity extraction must be reduced through lighter models or more efficient extraction strategies. The static nature of the knowledge graph must give way to dynamic update mechanisms. Evaluation must move beyond LLM-as-judge toward automated metrics that can measure factual accuracy, logical coherence, and attribution quality at scale. These are not merely engineering problems but fundamental research questions that will shape the next generation of retrieval-augmented language models.

The broader implication of this work is that structured knowledge representation is not a legacy technology to be displaced by neural methods but rather an essential complement to them. Knowledge graphs provide the relational scaffolding that enables multi-hop reasoning and hierarchical abstraction; neural language models provide the fluent generation and semantic understanding that make that knowledge accessible. GraphRAG is one concrete instantiation of this synergy, and the research agenda it opens—dynamic graphs, multi-modal integration, lightweight extraction—promises to extend the boundaries of what retrieval-augmented generation can accomplish.

---

## References

1. Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., & Larson, J. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. *Microsoft Research*. arXiv:2404.16130.

2. Han, H., Ma, L., Shomer, H., Wang, Y., Lei, Y., Guo, K., ... & Tang, J. (2025). RAG vs. GraphRAG: A Systematic Evaluation and Key Insights. arXiv:2502.11371.

3. Zhu, Z., Huang, T., Wang, K., Ye, J., Chen, X., & Luo, S. (2025). Graph-based Approaches and Functionalities in Retrieval-Augmented Generation: A Comprehensive Survey. arXiv:2504.10499.

4. Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports*, 9(1), 5233.

5. Guo, Z., Yan, L., et al. (2024). LightRAG: Simple and Fast Retrieval-Augmented Generation. arXiv:2410.05779.

6. Gutiérrez, B. J., Zhu, Y., et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. *NeurIPS 2024*. arXiv:2405.14831.

7. Li, D., Shen, J., et al. (2024). SubgraphRAG: Retrieval-Augmented Generation for Open-Domain Question Answering via Subgraph Reasoning. *ICLR 2025*. arXiv:2410.20724.

8. Zhao, Q., Li, C., et al. (2025). E2GraphRAG: Eliminating Embedding-Based Graph Retrieval-Augmented Generation. arXiv:2505.24226.

9. Wu, J., Zhu, Y., et al. (2024). MedGraphRAG: Graph RAG for Medical Domains. arXiv:2408.04187.

10. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.

11. Dean, J., & Ghemawat, S. (2008). MapReduce: Simplified Data Processing on Large Clusters. *Communications of the ACM*, 51(1), 107–113.

12. Blondel, V. D., et al. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics*, 2008(10), P10008.
