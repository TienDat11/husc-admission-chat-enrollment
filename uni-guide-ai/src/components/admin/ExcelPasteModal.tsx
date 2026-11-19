"use strict";

/**
 * ExcelPasteModal — unit U7.
 *
 * Accessible modal (focus trap, Esc closes and returns focus to trigger
 * button per the U6/U7 keyboard contract). Pastes Excel TSV on the frontend
 * (\t columns, \n rows, strips trailing empty line), previews rows/columns,
 * and submits via ``PATCH /admin/source-documents/{id}/blocks/{block_id}``.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { useToast } from "@/components/ui/use-toast";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { patchBlock } from "@/lib/adminApi";
import { useAdminSession } from "@/hooks/useAdminSession";

interface ExcelPasteModalProps {
  documentId: string;
  blockId: string;
  currentTsv?: string | null;
  onSuccess?: () => void;
}

export function ExcelPasteModal({
  documentId,
  blockId,
  currentTsv,
  onSuccess,
}: ExcelPasteModalProps) {
  const { token } = useAdminSession();
  const { toast } = useToast();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [rawTsv, setRawTsv] = useState(currentTsv ?? "");
  const [parseError, setParseError] = useState<string | null>(null);
  const [rows, setRows] = useState<string[][]>([]);
  const [submitting, setSubmitting] = useState(false);

  // Restore focus to the trigger when the modal closes.
  useEffect(() => {
    return () => {
      if (triggerRef.current) {
        triggerRef.current.focus({ preventScroll: true });
      }
    };
  }, []);

  const parseTsv = useCallback((text: string) => {
    setParseError(null);
    const lines = text.split("\n");
    while (lines.length > 0 && lines[lines.length - 1].trim() === "") {
      lines.pop();
    }
    if (lines.length === 0) {
      setRows([]);
      return;
    }
    const parsed = lines.map((line) => line.split("\t"));
    const colCount = parsed[0].length;
    for (const row of parsed) {
      if (row.length !== colCount) {
        setParseError(
          "Số cột không nhất quán giữa các dòng — hãy dán lại toàn bộ bảng từ Excel.",
        );
        setRows([]);
        return;
      }
    }
    setRows(parsed);
  }, []);

  const handlePaste = useCallback(
    (event: ClipboardEvent<HTMLTextAreaElement>) => {
      event.preventDefault();
      const text = event.clipboardData?.getData("text/plain");
      if (!text) return;
      setRawTsv(text);
      parseTsv(text);
    },
    [parseTsv],
  );

  const handleChange = useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      setRawTsv(event.target.value);
      parseTsv(event.target.value);
    },
    [parseTsv],
  );

  const nCols = rows.length > 0 ? rows[0].length : 0;

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (parseError || rows.length === 0) return;
    setSubmitting(true);
    try {
      const tsv = rows.map((r) => r.join("\t")).join("\n");
      await patchBlock(token, documentId, blockId, {
        table_tsv: tsv,
        content: tsv,
      });
      toast({ title: "Đã dán từ Excel", description: "Nội dung bảng đã được cập nhật." });
      onSuccess?.();
    } catch (err: unknown) {
      toast({
        title: "Dán thất bại",
        description: err instanceof Error ? err.message : "Lỗi không xác định",
        variant: "destructive",
      });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button ref={triggerRef} variant="outline" size="sm">
          Dán từ Excel
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Dán bảng từ Excel</DialogTitle>
          <DialogDescription>
            Dán vùng chọn từ Excel vào ô dưới. Cột cách bằng tab, dòng cách
            bằng xuống hàng. Dòng trống cuối sẽ bị bỏ qua.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-3 pt-4">
          <Label>Dán dữ liệu Excel (TSV)</Label>
          <Textarea
            value={rawTsv}
            onChange={handleChange}
            onPaste={handlePaste}
            rows={8}
            className="font-mono text-sm"
            placeholder="C1\tC2\tC3\nGiá 1\tGiá 2\tGiá 3\n…"
          />
          {parseError && <p className="text-xs text-destructive">{parseError}</p>}
          {!parseError && rows.length > 0 && (
            <div className="overflow-x-auto border rounded p-2 bg-muted/50 max-h-48 overflow-y-auto">
              <table className="text-xs border-collapse">
                <thead>
                  <tr>
                    {rows[0].map((cell, i) => (
                      <th key={i} className="border p-1 font-medium">
                        {cell}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.slice(1).map((row, rIdx) => (
                    <tr key={rIdx}>
                      {row.map((cell, cIdx) => (
                        <td key={cIdx} className="border p-1">{cell}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="text-xs text-muted-foreground mt-1">
                {rows.length - 1} dòng dữ liệu × {nCols} cột
              </p>
            </div>
          )}
        </form>
        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => {}}
            >Hủy</Button>
          <Button
            type="submit"
            disabled={submitting || parseError || rows.length === 0}
          >
            {submitting ? "Đang lưu…" : "Lưu bảng"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
