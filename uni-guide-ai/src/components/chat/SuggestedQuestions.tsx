interface SuggestedQuestionsProps {
  /** Follow-ups derived from the last answer; empty renders nothing. */
  questions: string[];
  onSelect: (question: string) => void;
}

// Câu hỏi tiếp theo: hàng viền 1px xếp dọc, co theo khung chat (390px không tràn ngang).
export function SuggestedQuestions({ questions, onSelect }: SuggestedQuestionsProps) {
  if (questions.length === 0) return null;

  return (
    <div className="max-w-[640px] pb-4">
      <p className="mb-2 text-[13px] font-semibold leading-[18px] text-[#6B7686]">
        Câu hỏi tiếp theo
      </p>
      <div role="group" aria-label="Câu hỏi gợi ý" className="flex flex-col gap-2">
        {questions.map((question) => (
          <button
            type="button"
            key={question}
            onClick={() => onSelect(question)}
            className="flex w-full min-w-0 items-center rounded-8 border border-[#D6DEEA] bg-white px-4 py-3 text-left text-[14px] font-medium leading-[20px] text-[#143980] transition-colors hover:bg-[#F3F5F9] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#143980]"
          >
            <span className="min-w-0 break-words">{question}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
