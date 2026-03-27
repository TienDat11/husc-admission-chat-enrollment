Create a professional infographic diagram on a white background titled "RAG Evaluation Framework — 3-Layer Quality Assessment".

Draw THREE stacked horizontal layers (like a cake or pyramid cross-section) from bottom to top:

BOTTOM LAYER (widest, blue #BBDEFB):
Label: "Layer 1: Context Recall"
Icon: magnifying glass over a database
Description text: "Did the retriever find ALL necessary information?"
Failure mode text (right side, red): "Miss → Retriever is broken"

MIDDLE LAYER (medium, orange #FFE0B2):
Label: "Layer 2: Faithfulness"  
Icon: document with checkmark
Description text: "Does the answer stick to retrieved context only?"
Failure mode text (right side, red): "Low → LLM is hallucinating"

TOP LAYER (smallest, green #C8E6C9):
Label: "Layer 3: Answer Correctness"
Icon: trophy/target bullseye
Description text: "Is the final answer actually correct?"
Failure mode text (right side, red): "Low → End-to-end failure"

Below the stack, draw a 2x3 DIAGNOSTIC MATRIX table with colored cells:

| Context Recall | Faithfulness | → Diagnosis |
|----------------|--------------|-------------|
| HIGH ✓         | HIGH ✓       | Schema/ER error (green) |
| LOW ✗          | HIGH ✓       | LLM hallucination (red) |
| HIGH ✓         | LOW ✗        | Generation temperature issue (orange) |

Title the table: "Diagnostic Matrix"

Add an arrow on the left side of the stacked layers pointing upward labeled "Increasing abstraction level".

Style: Clean infographic/explainer diagram, white background, flat design with subtle shadows, bold icons, sans-serif font, professional academic style suitable for thesis. 1600x1100px.
