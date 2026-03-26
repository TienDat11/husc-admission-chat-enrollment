import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_pipeline():
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    steps = [
        "Raw Documents",
        "Text Chunking",
        "Entity Extraction",
        "Knowledge Graph",
        "Community Detection",
        "Community Summaries"
    ]
    
    box_color = "#3498db"
    text_color = "white"
    arrow_color = "#2c3e50"

    x_start = 5
    y_center = 50
    width = 13
    height = 20
    spacing = 16

    for i, step in enumerate(steps):
        x = x_start + i * spacing
        rect = patches.FancyBboxPatch(
            (x, y_center - height/2), width, height,
            boxstyle="round,pad=0.5",
            linewidth=2, edgecolor="#2980b9", facecolor=box_color,
            alpha=0.9
        )
        ax.add_patch(rect)
        
        ax.text(
            x + width/2, y_center, step.replace(" ", "\n"),
            ha='center', va='center', fontsize=12, fontweight='bold',
            color=text_color, wrap=True
        )

        if i < len(steps) - 1:
            ax.annotate(
                '', xy=(x + width + spacing - 0.5, y_center), xytext=(x + width + 0.5, y_center),
                arrowprops=dict(arrowstyle='->', lw=2, color=arrow_color)
            )

    plt.title("GraphRAG Indexing Pipeline Architecture", fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("pipeline.png", bbox_inches='tight', dpi=300)
    print("Successfully saved pipeline.png")

if __name__ == "__main__":
    draw_pipeline()
