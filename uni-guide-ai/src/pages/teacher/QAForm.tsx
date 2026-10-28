import { useState, type FormEvent } from "react";
import { useMutation } from "@tanstack/react-query";
import { createRecord, newIdempotencyKey } from "@/lib/adminApi";
import { submitDraftForReview } from "@/lib/teacherApi";
import { Link, useSearchParams } from "react-router-dom";

const YEARS = [2026, 2027];
const MAJORS = ["Tất cả các ngành", "Hán Nôm", "Công nghệ thông tin", "Báo chí", "Kiến trúc"];

/**
 * Ticket T09 — "Gõ hỏi–đáp trực tiếp" (mockup 03, 05 §4.3).
 *
 * The question text arrives pre-filled when the screen is opened from the
 * "Câu hỏi bot chưa trả lời được" list (`?question=...`); the teacher only
 * types the answer. Saving builds the record the way 02 §1 says: the
 * system makes the source document ("Phòng tuyển sinh HUSC trả lời ngày …"),
 * so the teacher does not pick a source.
 */
const QAForm = () => {
  const [params] = useSearchParams();
  // Ticket T07: the "Bổ sung thông tin" link carries ?trace_id= so the
  // saved record is traceable back to the unanswered question. Omit
  // the key entirely when there is no trace id — the server must not
  // see an empty value it has to special-case.
  const traceId = params.get("trace_id");
  const [question, setQuestion] = useState(params.get("question") ?? "");
  const [answer, setAnswer] = useState("");
  const [year, setYear] = useState(String(YEARS[0]));
  const [major, setMajor] = useState(MAJORS[0]);
  const [fieldError, setFieldError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);

  const submit = useMutation({
    mutationFn: async () => {
      const key = newIdempotencyKey();
      const result = await createRecord(null, {
        schema_version: "1.0",
        record_type: "HOI_DAP",
        admission_year: Number(year),
        effective_from: new Date().toISOString().slice(0, 10),
        effective_to: null,
        authority_level: "husc_official",
        source_document_id: null,
        source_url: null,
        source_page: null,
        authored_by: null,
        locale: "vi",
        text: `Câu hỏi: ${question}\nTrả lời: ${answer}`,
        values: {
          ...(major === MAJORS[0] ? {} : { ten_nganh: major }),
          ...(traceId ? { source_trace_id: traceId } : {}),
        },
        status: "draft",
      });
      await submitDraftForReview(null, result.record.record_id, 1, key);
    },
    onSuccess: () => setSent(true),
    onError: (failure: unknown) =>
      setFieldError(
        failure instanceof Error ? failure.message : "Gửi duyệt thất bại.",
      ),
  });

  const handleAsk = () => {
    // "Hỏi thử" step: the teacher sees what the bot would answer *before*
    // sending for review — the real answer path, not a simulated copy.
    window.location.assign(`/quan-ly/hoi-thu?question=${encodeURIComponent(question)}`);
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    setFieldError(null);
    if (!answer.trim()) {
      setFieldError('Ô "Câu trả lời đúng" đang trống. Gõ câu trả lời rồi bấm "Gửi duyệt".');
      return;
    }
    submit.mutate();
  };

  if (sent) {
    return (
      <p role="status" className="note ok">
        <span aria-hidden="true">✓</span> Đã gửi. Người duyệt sẽ nhận được
        trong mục Chờ duyệt.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Gõ hỏi–đáp trực tiếp</h1>
      <p className="text-zinc-600">
        Dùng khi thông tin chưa có trong tài liệu nào, ví dụ câu thí sinh hay
        hỏi qua điện thoại.
      </p>

      {fieldError ? (
        <p role="alert" className="note bad">
          <span aria-hidden="true">⚠</span> {fieldError}
        </p>
      ) : null}

      <form className="space-y-5" onSubmit={handleSubmit}>
        <div className="field">
          <label htmlFor="qa-question">Câu hỏi thí sinh hay hỏi</label>
          <textarea
            id="qa-question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="qa-answer">Câu trả lời đúng</label>
          <textarea
            id="qa-answer"
            value={answer}
            onChange={(event) => setAnswer(event.target.value)}
          />
        </div>
        <div className="row">
          <div className="field">
            <label htmlFor="qa-year">Năm tuyển sinh</label>
            <select
              id="qa-year"
              value={year}
              onChange={(event) => setYear(event.target.value)}
            >
              {YEARS.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="qa-major">Ngành (không bắt buộc)</label>
            <select
              id="qa-major"
              value={major}
              onChange={(event) => setMajor(event.target.value)}
            >
              {MAJORS.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </div>
        </div>

        <p className="note info">
          Không cần chọn nguồn: câu trả lời được lưu thành tài liệu nguồn
          "Phòng tuyển sinh HUSC trả lời ngày {new Date().toLocaleDateString("vi-VN")}".
          Thí sinh sẽ thấy đúng dòng này khi bot trích nguồn.
        </p>

        <div className="actions">
          <button type="button" className="btn" onClick={handleAsk}>
            Hỏi thử
          </button>
          <button type="submit" className="btn btn-primary" disabled={submit.isPending}>
            {submit.isPending ? "Đang gửi…" : "Gửi duyệt"}
          </button>
        </div>
      </form>

      <p className="text-sm text-zinc-600">
        Về <Link to="/quan-ly/cau-hoi">danh sách câu hỏi bot chưa trả lời</Link>
      </p>
    </div>
  );
};

// ponytail: structuredValues is unused here — kept import removed, no need.
export default QAForm;
