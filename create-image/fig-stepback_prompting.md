Create a clean professional diagram on a white background titled "Step-back Prompting in GraphRAG Retrieval".

Draw the diagram as a vertical flow with two parallel branches that merge at the bottom:

TOP: One box "Original User Query q" (blue fill #BBDEFB) with example text inside: "Does the CNTT major accept A00 combination?"

Below it, draw an "LLM Step-back" processing box (purple fill #E1BEE7) with a small gear icon, labeled "Generate abstract question q̂"

This produces TWO boxes side by side:
LEFT: "Original Query q" (blue fill #BBDEFB) — "Does the CNTT major accept A00 combination?"
RIGHT: "Step-back Query q̂" (orange fill #FFE0B2) — "What are all admission methods at HUSC?"

From each of these draw arrows going DOWN to their respective retrieval boxes:
LEFT → "Local Search\n(Entity → Graph traversal)" (green fill #C8E6C9)
RIGHT → "Global Search\n(Community summaries)" (light green fill #DCEDC8)

Both retrieval boxes then converge with arrows into a UNION box:
"Context C_final = Retrieve(q) ∪ Retrieve(q̂)" (yellow fill #FFF9C4) with a union ∪ symbol prominent

Then one final arrow to:
"LLM Generator → Richer Answer" (purple fill #E1BEE7)

Add a callout annotation on the right side: "Step-back question targets a higher abstraction level — capturing background knowledge the direct query misses."

Style: Clean academic flowchart, white background, drop shadows, sans-serif font, all labels in English/mixed, color coded branches (blue=specific, orange=abstract, green=retrieval, yellow=merge). 1400x1400px.