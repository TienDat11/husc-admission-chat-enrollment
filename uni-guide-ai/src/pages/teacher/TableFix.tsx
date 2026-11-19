"use strict";

/**
 * TableFix — unit U7, Frame 4 & 9.
 *
 * Officer exception handling view: lists only exceptions/blockers, allows
 * 1-click insert-missing-rows, Excel paste modal, reOCR, and assigning
 * coordinate_key to UNANCHORED_DRAFT records. Shows ChunkHistogram preview.
 * Keeps "Gửi duyệt" (Submit) disabled while unresolved blockers remain.
 * Obeys the single-primary-button (`data-variant="primary"` #143980 per
 * section[data-workarea]) invariant.
 */
import { useCallback, useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useToast } from "@/components/ui/use-toast";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableRow, TableHeader } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import {
  createPreview,
  listExceptions,
  patchBlock,
  transitionRecord,
  type ExceptionsListResponse,
  type PreviewRequest,
  type PreviewResponse,
} from "@/lib/adminApi";
import { ChunkHistogram } from "@/components/admin/ChunkHistogram";
import { PdfPageViewer } from "@/components/admin/PdfPageViewer";

const RESOLVABLE_BLOCKERS = [
  "TABLE_CELL_DROP_SUSPECTED",
  "OCR_LOW_CONFIDENCE",
  "LAYOUT_UNKNOWN",
  "UNANCHORED_DRAFT",
];

const BLOCKER_LABELS: Record<string, string> = {
  TABLE_CELL_DROP_SUSPECTED: "Bảng thiếu dòng",
  OCR_LOW_CONFIDENCE: "OCR thấp",
  LAYOUT_UNKNOWN: "Bố cục chưa xác định",
  UNANCHORED_DRAFT: "Chưa gán tọa độ",
  AMBIGUOUS_SUPERSEDED: "Thay thế không rõ",
};

interface TableFixProps {}

export function TableFix() {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { documentId } = useParams<{ documentId?: string }>();
  const [exceptions, setExceptions] = useState<ExceptionsListResponse | null>(null);
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [submitDisabled, setSubmitDisabled] = useState(true);
  const [activeBlock, setActiveBlock] = useState<string | null>(null);

  const docId = documentId ?? "";

  const loadExceptions = useCallback(async () => {
    if (!docId) return;
    try {
      const data = await listExceptions(null, docId);
      setExceptions(data);
      const hasBlocker = data.blocks.some((b) =>
        b.warnings.some((w) => RESOLVABLE_BLOCKERS.includes(w)),
      );
      setSubmitDisabled(hasBlocker);
    } catch (err: unknown) {
      toast({
        title: "Không tải được danh sách ngoại lệ",
        description: err instanceof Error ? err.message : "Lỗi không xác định",
        variant: "destructive",
      });
    }
  }, [docId, toast]);

  useEffect(() => {
    loadExceptions();
  }, [loadExceptions]);

  const handlePatch = async (
    blockId: string,
    patch: { coordinate_key?: string; content?: string; expected_version?: number },
  ) => {
    setActiveBlock(blockId);
    try {
      await patchBlock(null, docId, blockId, {
        coordinate_key: patch.coordinate_key ?? null,
        content: patch.content ?? null,
        expected_version: patch.expected_version,
      });
      toast({ title: "Đã cập nhật block", description: "Trạng thái block đã được lưu." });
      setActiveBlock(null);
      await loadExceptions();
    } catch (err: unknown) {
      setActiveBlock(null);
      toast({
        title: "Cập nhật block thất bại",
        description: err instanceof Error ? err.message : "Lỗi không xác định",
        variant: "destructive",
      });
    }
  };

  const handlePreview = async (recordId: string, version: number) => {
    setPreviewLoading(true);
    try {
      const data = await createPreview(null, {
        record_id: recordId,
        expected_version: version,
      } satisfies PreviewRequest);
      setPreview(data);
    } catch (err: unknown) {
      toast({
        title: "Xem trước chunk thất bại",
        description: err instanceof Error ? err.message : "Lỗi không xác định",
        variant: "destructive",
      });
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleSubmit = async () => {
    if (submitDisabled) return;
    try {
      await transitionRecord(null, docId, {
        action: "submit",
        expected_version: 1,
        reason: "Gửi duyệt sau khi xử lý ngoại lệ",
      });
      toast({ title: "Đã gửi duyệt", description: "Bản ghi đã được chuyển sang trạng thái review." });
      navigate(-1);
    } catch (err: unknown) {
      toast({
        title: "Gửi duyệt thất bại",
        description: err instanceof Error ? err.message : "Lỗi không xác định",
        variant: "destructive",
      });
    }
  };

  const blockerCount = exceptions?.blocks.reduce((sum, b) => sum + b.warnings.length, 0) ?? 0;
  const unanchored = exceptions?.unanchored ?? [];

  const hasUnresolved = exceptions?.blocks.some((b) =>
    b.warnings.some((w) => RESOLVABLE_BLOCKERS.includes(w)),
  );

  return (
    <section data-workarea="table-fix" className="space-y-6">
      <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Xử lý ngoại lệ tài liệu</h1>
          <p className="text-muted-foreground text-sm">
            Mã hồ sơ: {docId || "—"} &nbsp;•&nbsp; {blockerCount} blocker chưa xử lý
          </p>
        </div>
        <div className="flex items-end gap-2">
          <Button
            variant="outline"
            onClick={() => navigate(-1)}
            disabled={false}
          >
            Về danh sách
          </Button>
          <Button
            data-variant="primary"
            onClick={handleSubmit}
            disabled={submitDisabled || blockerCount > 0}
          >
            {submitDisabled || blockerCount > 0 ? "Chưa thể gửi duyệt" : "Gửi duyệt"}
          </Button>
        </div>
      </div>

      {hasUnresolved && (
        <div className="rounded bg-warning/10 border border-warning/20 p-3">
          <p className="font-medium text-warning-foreground">
            Vẫn còn {blockerCount} blocker chưa xử lý — nút "Gửi duyệt" bị khóa cho đến khi tất cả ngoại lệ được giải quyết.
          </p>
        </div>
      )}

      <Tabs defaultValue="blocks">
        <TabsList className="grid w-full">
          <TabsTrigger value="blocks">Ngoại lệ bảng</TabsTrigger>
          <TabsTrigger value="unanchored">Chưa gán tọa độ</TabsTrigger>
        </TabsList>

        <TabsContent value="blocks" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Danh sách block có ngoại lệ</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <th>Block</th>
                    <th>Trang</th>
                    <th>Loại</th>
                    <th>Trạng thái</th>
                    <th>Cảnh báo</th>
                    <th>Hành động</th>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {exceptions?.blocks.map((b) => (
                    <TableRow key={b.block_id}>
                      <TableCell className="font-mono text-sm">{b.block_id}</TableCell>
                      <TableCell className="text-sm">{b.page}</TableCell>
                      <TableCell className="text-sm capitalize">{b.block_type}</TableCell>
                      <TableCell className="text-sm">
                        <Badge variant={b.state === "fixed" ? "success" : "default"}>
                          {b.state ?? "open"}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-sm">
                        <ul className="list-disc list-inside text-xs text-muted-foreground space-y-0">
                          {b.warnings.map((w) => (
                            <li key={w}>{BLOCKER_LABELS[w] ?? w}</li>
                          ))}
                        </ul>
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-col sm:flex-row gap-2">
                          {b.coordinate_key ? (
                            <Button size="sm" variant="outline" disabled>
                              Đã gán: {b.coordinate_key}
                            </Button>
                          ) : unanchored.some((u) => String(u.record_id) === b.block_id) && (
                            <Button
                              size="sm"
                              onClick={() => handlePatch(b.block_id, { coordinate_key: b.block_id })}
                              disabled={activeBlock === b.block_id}
                            >
                              Gán tọa độ
                            </Button>
                          )}
                          {b.warnings.includes("TABLE_CELL_DROP_SUSPECTED") && (
                            <PdfPageViewer
                              documentId={docId}
                              blockId={b.block_id}
                              pageNumber={b.page}
                              missingPypdfRows={b.missing_pypdf_rows}
                              onResolve={loadExceptions}
                            />
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                  {exceptions?.blocks.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={6} className="text-center text-muted-foreground">
                        Không có block nào có ngoại lệ.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="unanchored" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Bản ghi chưa gán tọa độ (unanchored)</CardTitle>
            </CardHeader>
            <CardContent>
              {unanchored.length === 0 ? (
                <p className="text-muted-foreground text-sm">Không có bản ghi chưa gán tọa độ.</p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <th>Record ID</th>
                      <th>Trạng thái</th>
                      <th>Lý do lưu trữ</th>
                      <th>Hành động</th>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {unanchored.map((u) => (
                      <TableRow key={String(u.record_id)}>
                        <TableCell className="font-mono text-sm">{String(u.record_id)}</TableCell>
                        <TableCell className="text-sm">{String(u.status ?? "archived")}</TableCell>
                        <TableCell className="text-sm">{String(u.archive_reason ?? "withdrawn")}</TableCell>
                        <TableCell>
                          <Button
                            size="sm"
                            onClick={() => handlePatch(String(u.record_id), { coordinate_key: String(u.record_id) })}
                            disabled={activeBlock === String(u.record_id)}
                          >
                            Gán tọa độ
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Chunk preview section */}
      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Xem trước chunk (theo record)</CardTitle>
              <CardDescription>
                Chạy chunker chuẩn trên một bản ghi, xem histogram token và cảnh báo trước khi gửi duyệt.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Input
                  placeholder="record_id (ví dụ CHI_TIEU:nganh:cntt:2026)"
                  className="flex-1 font-mono text-sm"
                  defaultValue="CHI_TIEU:nganh:cntt:2026"
                />
                <Button
                  onClick={() => handlePreview("CHI_TIEU:nganh:cntt:2026", 1)}
                  disabled={previewLoading}
                >
                  {previewLoading ? "Đang chạy…" : "Chạy xem trước"}
                </Button>
              </div>
              {preview && <ChunkHistogram preview={preview} />}
            </CardContent>
          </Card>
        </div>
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Tóm tắt ngoại lệ</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Tổng blocker</span>
                <span className="font-mono">{blockerCount}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Chưa gán tọa độ</span>
                <span className="font-mono">{unanchored.length}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Submit</span>
                <span className={`font-mono ${submitDisabled ? "text-destructive" : "text-success"}`}>
                  {submitDisabled ? "Bị khóa" : "Sẵn sàng"}
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
}
