// Teacher step 2 (design frame B8y2M): blocks that need a check on the left, the
// PDF page they sit on in the middle, and the editor for the selected block on
// the right. Data is the real block list of the uploaded document. Saving PATCHes
// the block, the only draft-edit endpoint the backend offers.
import { useMutation, useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import {
  AdminApiError,
  listExceptions,
  patchBlock,
  type ExceptionsListResponse,
} from "@/lib/adminApi";
import { warningLabel } from "@/lib/blockWarnings";
import { ExceptionBar } from "@/components/ui-kit/ExceptionBar";
import { PrimaryButton } from "@/components/ui-kit/PrimaryButton";
import { SecondaryButton } from "@/components/ui-kit/SecondaryButton";

type Block = ExceptionsListResponse["blocks"][number];

const PREVIEW_CHARS = 90;

function blockTitle(block: Block): string {
  const text = block.text.trim().replace(/\s+/g, " ");
  if (text === "") return "Không có chữ đọc được";
  return text.length > PREVIEW_CHARS ? `${text.slice(0, PREVIEW_CHARS)}…` : text;
}

function messageOf(error: unknown): string {
  if (error instanceof AdminApiError) return error.failure.message;
  if (error instanceof Error) return error.message;
  return "Có lỗi xảy ra. Thử lại sau.";
}

interface DraftWorkspaceProps {
  documentId: string;
  pageCount: number;
  onRestart: () => void;
}

export function DraftWorkspace({ documentId, pageCount, onRestart }: DraftWorkspaceProps) {
  const exceptions = useQuery({
    queryKey: ["draft-blocks", documentId],
    queryFn: () => listExceptions(null, documentId),
  });
  const blocks = useMemo(
    () => [...(exceptions.data?.blocks ?? [])].sort((a, b) => a.page - b.page || a.ordinal - b.ordinal),
    [exceptions.data],
  );
  const flagged = blocks.filter((block) => block.warnings.length > 0);

  const [pick, setPick] = useState<string | null>(null);
  const selected = flagged.find((block) => block.block_id === pick) ?? flagged[0] ?? null;
  const selectedIndex = selected === null ? -1 : flagged.indexOf(selected);
  const next = selectedIndex >= 0 ? (flagged[selectedIndex + 1] ?? null) : null;

  const save = useMutation({
    mutationFn: ({ block, content }: { block: Block; content: string; nextId: string | null }) =>
      patchBlock(null, documentId, block.block_id, { content, expected_version: block.version }),
    onSuccess: async (_result, variables) => {
      setPick(variables.nextId);
      await exceptions.refetch();
    },
  });

  if (exceptions.isError) {
    return <ExceptionBar title="Không tải được các mục của tài liệu" detail={messageOf(exceptions.error)} />;
  }
  if (exceptions.isLoading) {
    return <p role="status" className="text-base">Đang tải các mục cần kiểm tra…</p>;
  }

  const pageBlocks = selected === null ? [] : blocks.filter((block) => block.page === selected.page);

  return (
    <section aria-label="Kiểm tra từng mục" className="space-y-4">
      <div className="rounded-12 border border-[#D6DEEA] bg-white px-4 py-3">
        <p className="font-semibold text-[#143980]">
          {flagged.length > 0
            ? `Cần bạn kiểm tra ${flagged.length} mục trong ${blocks.length} mục đã đọc.`
            : `Không còn mục nào cần kiểm tra trong ${blocks.length} mục đã đọc.`}
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)_minmax(0,1fr)]">
        <div className="rounded-12 border border-[#D6DEEA] bg-white p-3">
          <h2 className="mb-2 text-sm font-semibold text-[#143980]">
            Mục cần kiểm tra · {flagged.length}/{blocks.length}
          </h2>
          <ul className="space-y-2">
            {flagged.map((block) => {
              const active = selected?.block_id === block.block_id;
              const labels = [...new Set(block.warnings.map(warningLabel))];
              return (
                <li key={block.block_id}>
                  <button
                    type="button"
                    aria-pressed={active}
                    onClick={() => setPick(block.block_id)}
                    className={`w-full rounded-8 border p-3 text-left ${
                      active ? "border-[#143980] bg-[#E6ECF7]" : "border-[#D6DEEA] bg-white"
                    }`}
                  >
                    <span className="block text-sm font-semibold text-[#0F1B2D]">{blockTitle(block)}</span>
                    <span className="block text-xs text-[#6B7686]">Trang {block.page}</span>
                    <span className="mt-2 flex flex-wrap gap-1">
                      {labels.map((label) => (
                        <span key={label} className="rounded-999 bg-[#FFF3DC] px-2 py-0.5 text-xs font-semibold text-[#8A5A00]">
                          {label}
                        </span>
                      ))}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
          <p className="mt-3 text-xs text-[#6B7686]">{blocks.length - flagged.length} mục đã đạt chuẩn</p>
        </div>

        <div className="rounded-12 border border-[#D6DEEA] bg-[#F3F5F9] p-3">
          <h2 className="mb-2 text-sm font-semibold text-[#143980]">
            Văn bản PDF gốc{selected !== null ? ` · Trang ${selected.page} / ${pageCount}` : ""}
          </h2>
          <div className="space-y-2">
            {pageBlocks.map((block) => {
              const current = selected?.block_id === block.block_id;
              return (
                <p
                  key={block.block_id}
                  className={`whitespace-pre-wrap rounded-8 border p-3 text-sm text-[#0F1B2D] ${
                    current ? "border-[#143980] bg-[#FFF3DC]" : "border-transparent bg-white"
                  }`}
                >
                  {block.text}
                </p>
              );
            })}
          </div>
        </div>

        <div className="rounded-12 border border-[#D6DEEA] bg-white p-3">
          {selected === null ? (
            <p className="text-base">Không còn mục nào cần sửa.</p>
          ) : (
            <BlockEditor
              key={selected.block_id}
              block={selected}
              saving={save.isPending}
              errorMessage={save.isError ? messageOf(save.error) : null}
              onSave={(content) => save.mutate({ block: selected, content, nextId: next?.block_id ?? null })}
              onSkip={() => setPick(next?.block_id ?? null)}
            />
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-[#E3E8F0] pt-3">
        <SecondaryButton onClick={onRestart}>Quay lại Bước 1</SecondaryButton>
        <p className="text-sm text-[#6B7686]">
          {flagged.length > 0 ? `Còn ${flagged.length} mục cần kiểm tra.` : "Đã kiểm tra hết các mục."}
        </p>
      </div>
    </section>
  );
}

interface BlockEditorProps {
  block: Block;
  saving: boolean;
  errorMessage: string | null;
  onSave: (content: string) => void;
  onSkip: () => void;
}

function BlockEditor({ block, saving, errorMessage, onSave, onSkip }: BlockEditorProps) {
  const [text, setText] = useState(block.text);
  const dirty = text !== block.text;
  const labels = [...new Set(block.warnings.map(warningLabel))];
  return (
    <div className="space-y-3">
      <h2 className="text-lg font-semibold text-[#143980]">Sửa mục · Trang {block.page}</h2>
      <div className="flex flex-wrap gap-1">
        {labels.map((label) => (
          <span key={label} className="rounded-999 bg-[#FFF3DC] px-2 py-0.5 text-xs font-semibold text-[#8A5A00]">
            {label}
          </span>
        ))}
      </div>
      <label htmlFor="block-content" className="block text-sm font-semibold text-[#0F1B2D]">
        Nội dung
      </label>
      <textarea
        id="block-content"
        value={text}
        onChange={(event) => setText(event.target.value)}
        rows={12}
        className="w-full rounded-8 border border-[#D6DEEA] p-3 text-base text-[#0F1B2D] focus:outline-none focus:ring-2 focus:ring-[#143980]"
      />
      {errorMessage !== null ? (
        <p role="alert" className="text-sm text-red-800">
          {errorMessage}
        </p>
      ) : null}
      {dirty ? (
        <PrimaryButton onClick={() => onSave(text)} disabled={saving}>
          {saving ? "Đang lưu…" : "Lưu & mục tiếp theo →"}
        </PrimaryButton>
      ) : (
        <SecondaryButton onClick={onSkip}>Mục tiếp theo →</SecondaryButton>
      )}
    </div>
  );
}
