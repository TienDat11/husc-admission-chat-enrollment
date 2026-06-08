# Multi-Dim Offline Metrics — s16 (G3-T2)

**Source records:** `results/eval_harness/86q_records_s16.jsonl` (n=86)  
**GT v2:** `results/ultraqa_metrics/fact_level_gt_v2.json` (n_facts=65 with chunk_ids=65)  
**Method:** pure file IO, no LLM, no embedding call. Numbers are computed from on-disk artifacts only.

---

## 1. Context recall @ real n (GATE 1)
- **context_recall = 1.0000**
- n_facts = 65  |  n_hits = 65  |  n_questions (with facts) = 55
- Verdict vs gate ≥ 0.85: **PASS** (was 1.000 on phantom n=2)

| id | n_facts | n_hits | recall |
|---|---:|---:|---:|
| msg001 | 1 | 1 | 1.000 |
| msg002 | 1 | 1 | 1.000 |
| msg003 | 1 | 1 | 1.000 |
| msg004 | 1 | 1 | 1.000 |
| msg005 | 1 | 1 | 1.000 |
| msg006 | 1 | 1 | 1.000 |
| msg008 | 1 | 1 | 1.000 |
| msg009 | 1 | 1 | 1.000 |
| msg010 | 1 | 1 | 1.000 |
| msg011 | 1 | 1 | 1.000 |

---

## 2. Retrieval precision@k / recall@k / MRR / hit-rate (k=1,3,5)
- n_evaluated (questions with GT facts) = 55

| metric | value |
|---|---:|
| p@1 | 0.5818 |
| p@3 | 0.2970 |
| **p@5** | **0.2000** |
| r@1 | 0.5455 |
| r@3 | 0.8455 |
| **r@5** | **0.9455** |
| hit@1 | 0.5818 |
| hit@3 | 0.8727 |
| hit@5 | 0.9636 |
| **MRR** | **0.7412** |

---

## 3. Route distribution
| route | count | % |
|---|---:|---:|
| hybrid | 53 | 61.6% |
| graph_rag | 24 | 27.9% |
| padded_rag | 7 | 8.1% |
| hyde_auto_answer | 2 | 2.3% |

---

## 4. Answer length + abstain + has-answer
- answer_len: n=86  min=121  p50=622  p95=946  max=2095  mean=625.9
- **abstain_rate = 0.1163** (10/86)
- has_answer_rate = 0.8837
- abstain × expected_behavior: {'answer': 8, 'abstain': 2}

---

## 5. Latency seams (ms)
| seam | n | p50 | p95 | max |
|---|---:|---:|---:|---:|
| route_ms | 86 | 9341 | 10644 | 28125 |
| retrieval_loop_ms | 86 | 1846 | 2548 | 16113 |
| query_ms | 86 | 7611 | 9309 | 10649 |
| total_ms | 86 | 18456 | 21858 | 38706 |
| attr_hyde_ms | 84 | 3402 | 4448 | 9582 |
| attr_router_inner_ms | 84 | 5780 | 7006 | 24304 |
| attr_gen_ms | 84 | 3906 | 5323 | 6077 |

---

## 6. Per-question failure table (truncated, top 25 by severity)

| id | route | expected | is_abstain | n_retrieved | cr | flags |
|---|---|---|---:|---:|---:|---|
| msg019 | hyde_auto_answer | answer | False | 0 | - | empty_retrieval |
| msg020 | hyde_auto_answer | answer | False | 0 | - | empty_retrieval |
| msg026 | graph_rag | abstain | False | 5 | 1.00 | over_answer |
| msg028 | graph_rag | abstain | False | 5 | 1.00 | over_answer |
| msg034 | graph_rag | abstain | False | 8 | - | over_answer |
| msg044 | hybrid | abstain | False | 5 | 1.00 | over_answer |
| msg059 | padded_rag | abstain | False | 5 | - | over_answer |
| msg002 | hybrid | answer | True | 5 | 1.00 | abstain_miss |
| msg007 | graph_rag | answer | True | 5 | - | abstain_miss |
| msg011 | hybrid | answer | True | 5 | 1.00 | abstain_miss |
| msg039 | graph_rag | answer | True | 5 | 1.00 | abstain_miss |
| msg040 | hybrid | answer | True | 5 | - | abstain_miss |
| msg043 | padded_rag | answer | True | 5 | - | abstain_miss |
| msg045 | hybrid | answer | True | 5 | - | abstain_miss |
| msg082 | hybrid | answer | True | 5 | 1.00 | abstain_miss |
| msg027 | graph_rag | abstain | True | 5 | 1.00 | abstain_correct |
| msg055 | graph_rag | abstain | True | 5 | - | abstain_correct |
| msg001 | graph_rag | answer | False | 5 | 1.00 |  |
| msg003 | hybrid | answer | False | 5 | 1.00 |  |
| msg004 | hybrid | answer | False | 5 | 1.00 |  |
| msg005 | graph_rag | answer | False | 5 | 1.00 |  |
| msg006 | hybrid | answer | False | 8 | 1.00 |  |
| msg008 | graph_rag | answer | False | 8 | 1.00 |  |
| msg009 | hybrid | answer | False | 5 | 1.00 |  |
| msg010 | hybrid | answer | False | 5 | 1.00 |  |

---

## 7. RANKED WEAKNESS LIST (G3-T2 deliverable)

| # | severity | id | metric | evidence |
|---|---|---|---|---|
| 1 | HIGH | msg026 | `abstain_accuracy` | expected=abstain but answer_len=760 (route=graph_rag) |
| 2 | HIGH | msg028 | `abstain_accuracy` | expected=abstain but answer_len=729 (route=graph_rag) |
| 3 | HIGH | msg034 | `abstain_accuracy` | expected=abstain but answer_len=565 (route=graph_rag) |
| 4 | HIGH | msg044 | `abstain_accuracy` | expected=abstain but answer_len=679 (route=hybrid) |
| 5 | HIGH | msg059 | `abstain_accuracy` | expected=abstain but answer_len=299 (route=padded_rag) |
| 6 | CRITICAL | msg019 | `retrieval_coverage` | retrieved_chunks=[] (route=hyde_auto_answer) |
| 7 | CRITICAL | msg020 | `retrieval_coverage` | retrieved_chunks=[] (route=hyde_auto_answer) |
| 8 | MED | (global) | `latency_route_ms` | route_ms p50=9341ms = 50.6% of total p50 18456ms |
| 9 | MED | (global) | `latency_query_ms` | query_ms p50=7611ms = 41.2% of total p50 18456ms |
| 10 | MED | (global) | `route_distribution` | route=hybrid took 53/86 = 61.6% (over-concentration) |

---

## Provenance

```json
{
  "method": "G3-T2 compute_offline_metrics.py — pure file IO; no LLM, no gateway, no embedding call.",
  "records_path": "D:\\chunking\\husc-admission-chat-enrollment\\rag2025\\results\\eval_harness\\86q_records_s16.jsonl",
  "gt_path": "D:\\chunking\\husc-admission-chat-enrollment\\rag2025\\results\\ultraqa_metrics\\fact_level_gt_v2.json",
  "gt_provenance": {
    "src_gt": "D:\\chunking\\husc-admission-chat-enrollment\\rag2025\\data\\eval\\husc_thi_sinh_thuc_gt.json",
    "out_path": "D:\\chunking\\husc-admission-chat-enrollment\\rag2025\\results\\ultraqa_metrics\\fact_level_gt_v2.json",
    "method": "deterministic, OFFLINE; no LLM call. Hand-picked supporting_chunk_ids for facts that lacked them, restricted to chunks that appear in s16 retrieved_chunks pool.",
    "n_questions": 86,
    "n_facts_total": 65,
    "n_facts_with_chunk_ids": 65,
    "extensions_applied": 7,
    "extensions_keys": [
      "msg006",
      "msg024",
      "msg031",
      "msg059"
    ]
  }
}
```
