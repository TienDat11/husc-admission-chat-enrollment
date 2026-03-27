# AMP SETUP: Skills + MCP (Context7 + Exa)

## 1) Skills nên cài cho dự án này

### Nhóm viết luận văn LaTeX
- `rag-latex-thesis-copilot` (custom, trong package này)
- `doc-coauthoring`
- `verification-before-completion`

### Nhóm code RAG/GraphRAG
- `rag-experiment-orchestrator` (custom, trong package này)
- `systematic-debugging`
- `test-driven-development`
- `amp-workflow-sop`
- `hyper-search`

### Nhóm tra cứu kiến thức
- `hyper-search` (bắt buộc khi research)
- `citation-output` (khi cần reference chặt)

---

## 2) Cấu trúc copy skill custom

Copy thư mục:
- `skills_custom/rag-latex-thesis-copilot/`
- `skills_custom/rag-experiment-orchestrator/`

vào thư mục skills của máy mới (ví dụ):
- `C:\Users\<YOU>\.claude\skills\`

Sau đó restart Amp/Claude CLI để nhận skill mới.

---

## 3) MCP config khuyến nghị (Context7 + Exa)

Nếu bạn dùng file cấu hình MCP JSON cục bộ, đảm bảo có 2 server:
- Context7
- Exa

Mẫu (điền command/token theo máy mới):

```json
{
  "mcpServers": {
    "Context7": {
      "command": "<context7-mcp-command>",
      "args": ["<args-if-any>"],
      "env": {
        "CONTEXT7_API_KEY": "${CONTEXT7_API_KEY}"
      }
    },
    "Exa": {
      "command": "<exa-mcp-command>",
      "args": ["<args-if-any>"],
      "env": {
        "EXA_API_KEY": "${EXA_API_KEY}"
      }
    }
  }
}
```

> Lưu ý: tên command/args phụ thuộc cách bạn cài MCP runtime ở máy mới.

---

## 4) Routing rule nên dùng hằng ngày
- Tài liệu thư viện/framework: **Context7 trước**.
- Xu hướng mới, paper, benchmark, code ví dụ ngoài đời: **Exa**.
- Không dùng web tool mặc định nếu đã có Exa/Context7.

---

## 5) Kiểm tra sau khi setup
1. Mở Amp và gọi skill custom trong prompt.
2. Test Context7: hỏi API docs cụ thể (ví dụ LangChain splitter).
3. Test Exa: hỏi paper/benchmark mới về chunking 2025/2026.
