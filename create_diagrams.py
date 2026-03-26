import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def create_naive_rag_vs_graph_rag():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('So sánh Naive RAG và GraphRAG', fontsize=16, fontweight='bold')

    # --- Diagram 1: Naive RAG ---
    ax1.set_title('Naive RAG (Vector Space)', fontsize=14)
    ax1.axis('off')
    
    # Draw Document
    doc_x, doc_y = 0.5, 0.9
    ax1.text(doc_x, doc_y, 'Tài liệu gốc', ha='center', va='center', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='blue'), fontsize=12)
    
    # Draw Chunks
    chunk_y = 0.6
    chunks = ['Chunk 1', 'Chunk 2', 'Chunk 3', '...']
    for i, chunk in enumerate(chunks):
        cx = 0.2 + i * 0.2
        ax1.text(cx, chunk_y, chunk, ha='center', va='center',
                 bbox=dict(boxstyle='square,pad=0.5', facecolor='lightgreen', edgecolor='green'), fontsize=10)
        ax1.annotate('', xy=(cx, chunk_y+0.05), xytext=(doc_x, doc_y-0.05),
                     arrowprops=dict(arrowstyle='->', color='gray'))

    # Draw Vector Database
    db_y = 0.3
    ax1.text(0.5, db_y, 'Cơ sở dữ liệu Vector', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', edgecolor='red'), fontsize=12)
    for i in range(len(chunks)):
        cx = 0.2 + i * 0.2
        ax1.annotate('', xy=(0.5, db_y+0.05), xytext=(cx, chunk_y-0.05),
                     arrowprops=dict(arrowstyle='->', color='gray'))

    # Draw Query
    q_x, q_y = 0.2, 0.1
    ax1.text(q_x, q_y, 'Truy vấn', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='orange'), fontsize=12)
    
    # Draw LLM / Response
    llm_x, llm_y = 0.8, 0.1
    ax1.text(llm_x, llm_y, 'LLM \n(Tạo phản hồi)', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='violet', edgecolor='purple'), fontsize=12)

    # Connections
    ax1.annotate('Tìm kiếm \ntương đồng', xy=(0.4, db_y-0.05), xytext=(q_x+0.1, q_y),
                 arrowprops=dict(arrowstyle='<->', color='black', linestyle='dashed'), fontsize=10)
    ax1.annotate('Ngữ cảnh \n(Top-K Chunks)', xy=(llm_x-0.1, llm_y), xytext=(0.6, db_y-0.05),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=10)


    # --- Diagram 2: GraphRAG ---
    ax2.set_title('GraphRAG (Knowledge Graph)', fontsize=14)
    ax2.axis('off')

    # Create a simple graph
    G = nx.Graph()
    nodes = ['A', 'B', 'C', 'D', 'E', 'F']
    edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E'), ('D', 'F'), ('E', 'F'), ('B', 'C')]
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=42) # Fixed layout
    
    # Shift graph up
    for k in pos:
        pos[k][1] += 0.4
        pos[k][0] = pos[k][0] * 0.5 + 0.5 # Scale and shift to right
        
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color='lightgreen', node_size=700)
    nx.draw_networkx_edges(G, pos, ax=ax2, edge_color='gray', width=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax2, font_size=10, font_weight='bold')
    
    # Highlight a community
    community_nodes = ['A', 'B', 'C']
    nx.draw_networkx_nodes(G, pos, nodelist=community_nodes, ax=ax2, node_color='lightcoral', node_size=800)
    ax2.text(0.5, 0.9, 'Trích xuất Thực thể & Quan hệ', ha='center', va='center', fontsize=11, style='italic', color='purple')
    
    # Community Summaries
    sum_y = 0.3
    ax2.text(0.5, sum_y, 'Tóm tắt Cộng đồng\n(Community Summaries)', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', edgecolor='darkorange'), fontsize=11)
             
    # Arrows from graph to summary
    ax2.annotate('', xy=(0.5, sum_y+0.1), xytext=(0.5, 0.6), arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Query and Response
    q_x2, q_y2 = 0.2, 0.1
    ax2.text(q_x2, q_y2, 'Truy vấn\n(Toàn cục)', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='orange'), fontsize=12)
             
    llm_x2, llm_y2 = 0.8, 0.1
    ax2.text(llm_x2, llm_y2, 'Map-Reduce LLM\n(Tổng hợp)', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='violet', edgecolor='purple'), fontsize=12)

    ax2.annotate('Gửi đến\ncác tóm tắt', xy=(0.4, sum_y-0.1), xytext=(q_x2+0.1, q_y2+0.05),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=10)
    ax2.annotate('Kết hợp\ncâu trả lời', xy=(llm_x2-0.15, llm_y2+0.05), xytext=(0.6, sum_y-0.1),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=10)

    plt.tight_layout()
    plt.savefig('diagram1.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_map_reduce_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Kiến trúc Truy vấn Toàn cục (Map-Reduce)', fontsize=16, fontweight='bold')
    ax.axis('off')

    # Query
    ax.text(0.5, 0.9, 'Truy vấn người dùng (Query)', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='orange'), fontsize=12)

    # Community Summaries (Map Phase)
    y_map = 0.6
    ax.text(0.1, y_map+0.1, 'Giai đoạn Map', ha='center', va='center', fontsize=12, fontweight='bold', color='blue')
    
    summaries = ['Tóm tắt\nCộng đồng 1', 'Tóm tắt\nCộng đồng 2', '...', 'Tóm tắt\nCộng đồng N']
    x_pos = np.linspace(0.2, 0.8, 4)
    
    for i, (x, text) in enumerate(zip(x_pos, summaries)):
        ax.text(x, y_map, text, ha='center', va='center',
                bbox=dict(boxstyle='square,pad=0.5', facecolor='lightgreen', edgecolor='green'), fontsize=10)
        # Arrow from Query
        ax.annotate('', xy=(x, y_map+0.1), xytext=(0.5, 0.85), arrowprops=dict(arrowstyle='->', color='gray'))

    # Partial Responses
    y_partial = 0.35
    partials = ['Trả lời\ncục bộ 1\n(Điểm: 8)', 'Trả lời\ncục bộ 2\n(Điểm: 9)', '...', 'Trả lời\ncục bộ N\n(Điểm: 5)']
    for i, (x, text) in enumerate(zip(x_pos, partials)):
        ax.text(x, y_partial, text, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='blue'), fontsize=9)
        # Arrow from Map
        ax.annotate('LLM', xy=(x, y_partial+0.1), xytext=(x, y_map-0.1), arrowprops=dict(arrowstyle='->', color='black'))

    # Reduce Phase
    y_reduce = 0.1
    ax.text(0.1, y_reduce+0.05, 'Giai đoạn Reduce', ha='center', va='center', fontsize=12, fontweight='bold', color='red')
    
    ax.text(0.5, y_reduce, 'LLM Tổng hợp\n(Câu trả lời cuối cùng)', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='violet', edgecolor='purple'), fontsize=12)

    # Arrows to Reduce
    for x in x_pos:
        ax.annotate('', xy=(0.5, y_reduce+0.1), xytext=(x, y_partial-0.1), arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig('diagram2.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    create_naive_rag_vs_graph_rag()
    create_map_reduce_diagram()
    print("Diagrams created.")
