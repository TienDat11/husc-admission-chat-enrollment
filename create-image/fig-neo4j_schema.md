Create a clean professional Knowledge Graph schema diagram on a white background for a university admissions system, titled "Neo4j Schema — HUSC Admissions Knowledge Graph".

Draw the following NODES as shapes:
- "Đại Học Huế" (Hue University) — large ellipse, blue fill (#BBDEFB), center top
- "Khoa CNTT" (IT Faculty) — medium ellipse, blue fill (#BBDEFB), lower left
- "Ngành CNTT" (IT Major) — medium ellipse, blue fill (#BBDEFB), lower right
- "Chunk 1" — rounded rectangle, green fill (#C8E6C9), bottom left
- "Chunk 2" — rounded rectangle, green fill (#C8E6C9), bottom right
- "Cộng đồng CNTT" (IT Community) — rounded rectangle with thick border, orange fill (#FFE0B2), center bottom

Draw DIRECTED ARROWS (dark gray, with arrowheads) between nodes with relationship labels on the arrows:
- "Khoa CNTT" → "Đại Học Huế": label "THUỘC_TRƯỜNG" (BELONGS_TO)
- "Khoa CNTT" → "Ngành CNTT": label "CÓ_NGÀNH" (HAS_MAJOR)
- "Chunk 1" → "Khoa CNTT": label "HAS_ENTITY"
- "Chunk 2" → "Ngành CNTT": label "HAS_ENTITY"
- "Khoa CNTT" → "Cộng đồng CNTT": label "IN_COMMUNITY"
- "Ngành CNTT" → "Cộng đồng CNTT": label "IN_COMMUNITY"

Add a legend in bottom right corner:
- Blue ellipse = Entity node
- Green rectangle = Text Chunk node
- Orange rectangle = Community node

Style: Clean professional graph database diagram, white background, nodes have subtle drop shadow, relationship labels in small dark gray font on white pill-shaped background, arrows are thin with clean arrowheads. Font: sans-serif. 1600x1000px.
