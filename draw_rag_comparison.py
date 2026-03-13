import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def draw_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Naive RAG vs GraphRAG Comparison', fontsize=20, fontweight='bold', y=0.95)

    # --- Naive RAG: Vector Search (Isolated Chunks) ---
    ax1.set_title("Naive RAG\n(Tìm kiếm theo từ khóa/ý nghĩa cô lập)", fontsize=14, color='darkblue', pad=20)
    
    # Generate points for "chunks"
    np.random.seed(42)
    chunks_x = np.random.rand(20)
    chunks_y = np.random.rand(20)
    
    # Scatter plot for chunks
    ax1.scatter(chunks_x, chunks_y, c='lightgrey', s=100, label='Dữ liệu (Chunks)')
    
    # Highlight retrieved chunks
    query_x, query_y = 0.5, 0.5
    ax1.scatter(query_x, query_y, c='red', marker='*', s=300, label='Câu hỏi (Query)', zorder=5)
    
    # Draw circle of similarity
    circle = plt.Circle((query_x, query_y), 0.25, color='red', fill=False, linestyle='--', alpha=0.5)
    ax1.add_patch(circle)
    
    # Highlight "hits"
    hits_mask = (chunks_x - query_x)**2 + (chunks_y - query_y)**2 < 0.25**2
    ax1.scatter(chunks_x[hits_mask], chunks_y[hits_mask], c='orange', s=120, edgecolors='black', label='Kết quả tìm thấy')
    
    ax1.text(0.5, -0.1, "Giống như tìm sách lẻ tẻ trên kệ\ndựa trên tên bìa.", 
             ha='center', fontsize=12, transform=ax1.transAxes, style='italic')
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.axis('off')
    ax1.legend(loc='upper right')

    # --- GraphRAG: Knowledge Graph + Communities ---
    ax2.set_title("GraphRAG\n(Kết nối ý tưởng & Hiểu toàn cảnh)", fontsize=14, color='darkgreen', pad=20)
    
    # Create a graph with communities
    G = nx.Graph()
    
    # Community 1: Marketing
    comm1 = [1, 2, 3, 4, 5]
    G.add_edges_from([(1,2), (1,3), (2,3), (3,4), (4,5), (5,1)])
    
    # Community 2: Product
    comm2 = [6, 7, 8, 9, 10]
    G.add_edges_from([(6,7), (6,8), (7,8), (8,9), (9,10), (10,6)])
    
    # Connect communities
    G.add_edge(3, 8)
    
    pos = nx.spring_layout(G, seed=42)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.3, edge_color='gray')
    
    # Draw nodes with community colors
    nx.draw_networkx_nodes(G, pos, nodelist=comm1, node_color='lightblue', node_size=500, ax=ax2, label='Chủ đề A (Marketing)')
    nx.draw_networkx_nodes(G, pos, nodelist=comm2, node_color='lightgreen', node_size=500, ax=ax2, label='Chủ đề B (Sản phẩm)')
    
    # Highlight the "Synthesis" path
    nx.draw_networkx_edges(G, pos, edgelist=[(3,8)], width=3, edge_color='red', ax=ax2)
    
    # Annotate GraphRAG features
    ax2.text(0.5, -0.1, "Kết nối các mẩu thông tin thành mạng lưới.\nHiểu được mối quan hệ giữa các chủ đề.", 
             ha='center', fontsize=12, transform=ax2.transAxes, style='italic')
    
    ax2.axis('off')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('naive_vs_graphrag.png', dpi=150)
    print("Successfully created naive_vs_graphrag.png")

if __name__ == "__main__":
    draw_comparison()
