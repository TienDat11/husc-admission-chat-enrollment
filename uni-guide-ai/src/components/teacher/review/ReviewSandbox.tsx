import type { ReviewPreviewResult } from "@/lib/adminApi";

type Props = {
  question: string;
  onQuestionChange: (value: string) => void;
  onAsk: (question: string) => void;
  pending: boolean;
  preview: ReviewPreviewResult | null;
};

/** Sandbox: ask the Bot a question against the draft and the serving index. */
export function ReviewSandbox({ question, onQuestionChange, onAsk, pending, preview }: Props) {
  return (
    <section aria-labelledby="cho-duyet-sandbox" className="rounded-[12px] border border-[#D6DEEA] bg-white p-5">
      <h2 id="cho-duyet-sandbox" className="!m-0 font-display text-[15px] font-semibold text-[#0F1B2D]">
        Hỏi thử Bot với dữ liệu này (Sandbox)
      </h2>
      <p className="mt-1 text-[13px] text-[#6B7686]">
        Chỉ tìm trong bản nháp và chỉ mục đang phục vụ. Không ghi vào phiên bản phát hành.
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        <label htmlFor="duyet-cau-hoi" className="sr-only">
          Câu hỏi thử của thí sinh
        </label>
        <input
          id="duyet-cau-hoi"
          type="text"
          className="min-w-[14rem] flex-1"
          value={question}
          onChange={(event) => onQuestionChange(event.target.value)}
          placeholder="Câu hỏi của thí sinh"
        />
        <button type="button" className="btn" disabled={!question.trim() || pending} onClick={() => onAsk(question.trim())}>
          {pending ? "Đang chạy…" : "Hỏi thử"}
        </button>
      </div>
      {preview ? (
        <div className="answer mt-3" role="status" aria-live="polite">
          <p>
            <strong>Bot sẽ trả lời:</strong> {preview.answer ?? "(chưa có câu trả lời)"}
          </p>
          {preview.sources.length > 0 ? (
            <div>
              <p className="hint">Nguồn trích dẫn:</p>
              {preview.sources.map((source) => (
                <span key={`${source.source_document_id ?? ""}|${source.section ?? ""}|${source.source}`} className="cite">
                  {source.title ?? source.section ?? source.source}
                </span>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
