"use strict";

/**
 * PdfPageViewer — unit U7.
 *
 * Displays the source PDF page / pypdf vs Docling comparison and highlights
 * missing rows (``missing_pypdf_rows``), with 1-click "Chèn dòng thiếu"
 * (insert-missing-rows), "Dán từ Excel" (ExcelPasteModal), and "OCR lại
 * trang" (reocr).
 */
import { useCallback, useState } from "react";
import { useToast } from "@/components/ui/use-toast";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ExcelPasteModal } from "./ExcelPasteModal";
import {
  insertMissingRows,
  reocr,
  type ReocrRequest,
} from "@/lib/adminApi";
import { useAdminSession } from "@/hooks/useAdminSession";

interface PdfPageViewerProps {
  documentId: string;
  blockId: string;
  pageNumber: number;
  missingPypdfRows?: string[][];
  pdfUrl?: string | null;
  onResolve?: () => void;
}

export function PdfPageViewer({
  documentId,
  blockId,
  pageNumber,
  missingPypdfRows,
  pdfUrl,
  onResolve,
}: PdfPageViewerProps) {
  const { token } = useAdminSession();
  const { toast } = useToast();
  const [ocrRunning, setOcrRunning] = useState(false);

  const handleInsertMissing = useCallback(async () => {
    try {
      await insertMissingRows(token, documentId, blockId, {
        expected_version: undefined,
      });
      toast({
        title: "Đã chèn dòng thiếu",
        description: `Trang ${pageNumber}: đã thêm ${missingPypdfRows?.length ?? 0} dòng từ pypdf vào bảng.`,
      });
      onResolve?.();
    } catch (err: unknown) {
      toast({
        title: "Chèn dòng thất bại",
        description: err instanceof Error ? err.message : "Lỗi không xác định",
        variant: "destructive",
      });
    }
  }, [token, documentId, blockId, pageNumber, missingPypdfRows, onResolve]);

  const handleReocr = useCallback(async () => {
    const body: ReocrRequest = { page_numbers: [pageNumber] };
    setOcrRunning(true);
    try {
      await reocr(token, documentId, body);
      toast({
        title: "Đã đưa vào hàng chờ OCR",
        description: `Trang ${pageNumber} sẽ được OCR lại.`,
      });
    } catch (err: unknown) {
      toast({
        title: "OCR lại thất bại",
        description: err instanceof Error ? err.message : "Lỗi không xác định",
        variant: "destructive",
      });
    } finally {
      setOcrRunning(false);
    }
  }, [token, documentId, pageNumber]);

  const hasMissing = missingPypdfRows && missingPypdfRows.length > 0;

  return (
    <Card className="border">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">Trang PDF #{pageNumber}</CardTitle>
        <div className="flex gap-2">
          {hasMissing && (
            <Button
              size="sm"
              onClick={handleInsertMissing}
              disabled={!token}
            >
              Chèn dòng thiếu
            </Button>
          )}
          <Button size="sm" onClick={handleReocr} disabled={ocrRunning || !token}>
            {ocrRunning ? "Đang OCR…" : "OCR lại trang"}
          </Button>
          <ExcelPasteModal documentId={documentId} blockId={blockId} />
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="flex flex-col gap-2">
          {pdfUrl ? (
            <iframe
              className="w-full h-[400px] rounded border"
              src={pdfUrl + `#page=${pageNumber}`}
              title={`Trang ${pageNumber}`}
            />
          ) : (
            <div className="h-[300px] flex items-center justify-center rounded border border-dashed bg-muted/50 text-muted-foreground text-sm">
              Không có bản xem trước PDF
            </div>
          )}
          {hasMissing && (
            <div className="rounded bg-warning/10 border border-warning/20 p-3 text-sm">
              <p className="font-medium">Phát hiện {missingPypdfRows.length} dòng bị thiếu trên trang này.</p>
              <p className="text-xs text-muted-foreground mt-1">
                Dùng "Chèn dòng thiếu" để bổ sung từ pypdf, hoặc "Dán từ Excel" để sửa tay.
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
