import { useState } from "react";
import { useSearchParams, Link } from "react-router-dom";
import { sendChatMessage, type QueryResponse } from "@/lib/api";

/**
 * Ticket T09 — "Hỏi thử bot" (mockup 07, 05 §4.2 step 3): the teacher types
 * a question the way a student would and reads the live answer + sources.
 * The question is pre-filled when the screen is opened from the add-info
 * flow (`?question=...`).
 */
const AskTest = () => {
  const [params] = useSearchParams();
  const [question, setQuestion] = useState(params.get("question") ?? "");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);

  const ask = async () => {
    const trimmed = question.trim();
    if (!trimmed) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await sendChatMessage(trimmed);
      setResult(response);
    } catch {
      setError("Bot đang bận, chưa trả lời được. Thử lại sau 1 phút.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-5">
      <h1 className="text-2xl font-bold">Hỏi thử bot</h1>
      <p className="text-zinc-600">
        Hỏi như thí sinh để xem bot đang trả lời gì. Câu hỏi ở đây không tính
        vào thống kê.
      </p>

      <form
        className="space-y-3"
        onSubmit={(event) => {
          event.preventDefault();
          void ask();
        }}
      >
        <div className="field">
          <label htmlFor="hoi-thu-cau-hoi">Câu hỏi</label>
          <input
            id="hoi-thu-cau-hoi"
            type="text"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
          />
        </div>
        <div className="actions">
          <button
            type="submit"
            className="btn btn-primary"
            disabled={!question.trim() || loading}
          >
            {loading ? "Đang hỏi…" : "Hỏi"}
          </button>
        </div>
      </form>

      {loading ? (
        <p role="status" aria-live="polite" className="text-zinc-600">
          <span aria-hidden="true">⏳</span> Bot đang tìm thông tin và soạn
          câu trả lời…
        </p>
      ) : null}

      {error ? (
        <div className="space-y-2" role="alert">
          <p className="note bad">
            <span aria-hidden="true">⚠</span> {error}
          </p>
          <button type="button" className="btn" onClick={() => void ask()}>
            Thử lại
          </button>
        </div>
      ) : null}

      {result ? (
        <div className="answer" role="status" aria-live="polite">
          <p>
            <strong>Bot trả lời:</strong> {result.answer ?? "(không có câu trả lời)"}
          </p>

          {result.sources && result.sources.length > 0 ? (
            <div>
              <p className="hint">Nguồn (bấm để mở đúng chỗ):</p>
              <div className="flex flex-wrap gap-2">
                {result.sources.map((source, index) => (
                  <a
                    key={source.id ?? `${source.title}-${index}`}
                    className="cite"
                    href={source.url ?? "#"}
                    target={source.url ? "_blank" : undefined}
                    rel="noreferrer"
                  >
                    📄 {source.title ?? source.snippet?.slice(0, 60) ?? "Nguồn"}
                  </a>
                ))}
              </div>
            </div>
          ) : null}

          {result.data_gap_hints && result.data_gap_hints.length > 0 ? (
            <p className="note info">
              Câu này đã được đưa vào mục "Câu hỏi bot chưa trả lời được".{" "}
              <Link to="/quan-ly/cau-hoi">Xem danh sách</Link>
            </p>
          ) : null}
        </div>
      ) : null}

      <p className="text-sm text-zinc-600">
        Về <Link to="/quan-ly">Tổng quan</Link>
      </p>
    </div>
  );
};

export default AskTest;
