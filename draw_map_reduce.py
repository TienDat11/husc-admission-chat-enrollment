import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_map_reduce():
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    query_color = "#e74c3c"
    map_color = "#3498db"
    reduce_color = "#2ecc71"
    answer_color = "#f39c12"
    text_color = "white"
    
    ax.add_patch(patches.FancyBboxPatch((40, 85), 20, 10, boxstyle="round,pad=0.5", facecolor=query_color))
    ax.text(50, 90, "User Query", ha='center', va='center', color=text_color, fontweight='bold')

    community_centers = [20, 35, 50, 65, 80]
    for x in community_centers:
        ax.add_patch(patches.FancyBboxPatch((x-7, 55), 14, 15, boxstyle="round,pad=0.5", facecolor=map_color, alpha=0.8))
        ax.text(x, 62.5, "Community\nReport", ha='center', va='center', color=text_color, fontsize=9)
        ax.annotate('', xy=(x, 71), xytext=(50, 84), arrowprops=dict(arrowstyle='->', lw=1, color='gray', alpha=0.5))

    for x in community_centers:
        ax.add_patch(patches.Circle((x, 40), 3, facecolor=reduce_color, alpha=0.9))
        ax.text(x, 40, "Ans", ha='center', va='center', color=text_color, fontsize=8)
        ax.annotate('', xy=(x, 44), xytext=(x, 54), arrowprops=dict(arrowstyle='->', lw=1.5, color='#2c3e50'))

    ax.add_patch(patches.FancyBboxPatch((35, 15), 30, 12, boxstyle="round,pad=0.5", facecolor=answer_color))
    ax.text(50, 21, "Global Answer\n(Summarization)", ha='center', va='center', color=text_color, fontweight='bold')

    for x in community_centers:
        ax.annotate('', xy=(50, 28), xytext=(x, 36), arrowprops=dict(arrowstyle='->', lw=1.2, color='#2c3e50', alpha=0.7))

    plt.title("GraphRAG Global Search (Map-Reduce Architecture)", fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("map_reduce.png", bbox_inches='tight', dpi=300)
    print("Successfully saved map_reduce.png")

if __name__ == "__main__":
    draw_map_reduce()
