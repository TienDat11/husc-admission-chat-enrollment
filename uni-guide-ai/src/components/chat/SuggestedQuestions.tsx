interface SuggestedQuestionsProps {
  /** Follow-ups derived from the last answer; empty renders nothing. */
  questions: string[];
  onSelect: (question: string) => void;
}

// Câu hỏi tiếp theo: chip gọn nằm trong khung câu trả lời, xuống dòng khi cần (390px không tràn ngang).
export function SuggestedQuestions({ questions, onSelect }: SuggestedQuestionsProps) {
  if (questions.length === 0) return null;

  return (
    <div className="mt-4 border-t border-[#EDF1F6] pt-3">
      <p className="mb-2 text-[12px] font-semibold leading-[16px] text-[#6B7686]">
        Câu hỏi tiếp theo
      </p>
      <div role="group" aria-label="Câu hỏi gợi ý" className="flex flex-wrap gap-2">
        {questions.map((question) => (
          <button
            type="button"
            key={question}
            onClick={() => onSelect(question)}
            className="min-w-0 max-w-full rounded-999 border border-[#D6DEEA] bg-white px-3 py-1.5 text-left text-[13px] font-medium leading-[18px] text-[#143980] transition-colors hover:bg-[#F3F5F9] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#143980]"
          >
            <span className="break-words">{question}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
