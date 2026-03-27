Create a professional technical diagram on a white background titled "Agentic RAG — Adaptive Retrieval Loop".

Draw a circular/cyclical flow diagram with these 5 nodes arranged in a pentagon/loop:

1. "User Query" — rectangle, light purple fill (#E1BEE7), at top
2. "Retriever Router\n(LLM decides tool)" — rounded rectangle, blue fill (#BBDEFB), right
3. "Tool Execution\n(Vector / BM25 / Cypher)" — rounded rectangle, green fill (#C8E6C9), bottom right. Show 3 mini icons: database icon, text search icon, graph icon stacked horizontally inside.
4. "LLM Generator\n(Draft answer)" — rounded rectangle, yellow fill (#FFF9C4), bottom left
5. "Answer Critic\n(Quality check)" — rounded rectangle, orange fill (#FFE0B2), left

Draw ARROWS forming a loop:
Query → Retriever Router → Tool Execution → LLM Generator → Answer Critic

From Answer Critic draw TWO arrows:
- One arrow UPWARD back to Retriever Router labeled "❌ Insufficient — retry" (dashed red)
- One arrow RIGHT going OUT of the loop labeled "✅ Sufficient — return answer" (solid green) pointing to a final "Final Answer" box (green fill)

Add small annotation showing iteration counter "Max iterations: N" near the retry arrow.

Add a label "Adaptive Loop" in the center of the cycle with a circular arrow watermark.

Style: Clean flowchart style, white background, drop shadows on nodes, bold arrowheads, sans-serif font, professional color palette, infographic quality. 1600x1200px.