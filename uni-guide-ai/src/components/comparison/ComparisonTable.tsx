// Ticket 24 — UI bảng so sánh nhiều ngành (comparison table).
//
// Renders a `ComparisonResponse` from POST /v2/compare exactly as the
// contract delivers it. The three rules this component exists to hold,
// each one a way a comparison table could lie to a teacher:
//
// 1. **Không suy diễn dữ liệu thiếu.** A cell the backend marks `missing`
//    (or that carries no value) renders "Chưa có thông tin chính thức" —
//    never the neighbour row's value, never another year's number. The
//    backend already refuses to infer; this render path is the FE-side
//    guard so a malformed payload cannot make the table invent a number.
// 2. **Mỗi ô có nguồn riêng.** Every present cell with provenance renders
//    its own openable citation (link to the official document when the
//    contract carries a URL, expandable record/page/chunk detail
//    otherwise) — per-cell, not per-table.
// 3. **Một khối so sánh chỉ một năm hiệu lực.** The backend splits
//    year-mismatched majors into `refused` and validates table uniformity,
//    but the FE re-checks: any row whose `effective_year` differs from the
//    table's is EXCLUDED from the body and surfaced explicitly in a
//    warning, alongside the `refused` list. Mixed-year data is never
//    silently merged into one comparison block.
//
// Accessibility: column headers carry `scope="col"` and each row's major
// label is a `scope="row"` header cell, so screen readers announce
// "Ngành A / điểm chuẩn / 24.5" rather than a grid of unlabeled numbers.
// On narrow screens the table scrolls horizontally inside its container
// (the shadcn Table wrapper); no column — least of all the citation
// affordances — is hidden at any width.
import type { components } from '@/lib/api-types';
import type { ComparisonResponsePayload as ComparisonResponse } from '@/lib/api';
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';

type ComparisonCell = components['schemas']['ComparisonCell'];
type ComparisonRow = components['schemas']['ComparisonRow'];
type RefusedMajor = components['schemas']['RefusedMajor'];

/** The exact string criterion 3 requires for a cell with no official data. */
export const NO_OFFICIAL_DATA = 'Chưa có thông tin chính thức';

interface ComparisonTableProps {
  response: ComparisonResponse;
}

/** One present cell's openable citation (rule 2). */
function CellCitation({ cell }: { cell: ComparisonCell }) {
  const source = cell.source;
  if (!source) {
    // The contract guarantees provenance for present cells. If a payload
    // breaks that promise the FE says so instead of showing a bare number
    // as if it were verified.
    return (
      <p className="mt-1 text-[11px] italic text-destructive">
        ô này chưa kèm nguồn để kiểm chứng
      </p>
    );
  }
  return (
    <details className="mt-1">
      <summary className="cursor-pointer select-none text-[11px] font-medium text-primary">
        Xem nguồn ({cell.major})
      </summary>
      <div className="mt-1 space-y-0.5 text-[11px] text-muted-foreground">
        {source.source_url ? (
          <a
            href={source.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="block font-medium text-primary underline underline-offset-2"
          >
            Tài liệu chính thức
          </a>
        ) : null}
        <p className="break-all">Bản ghi: {source.record_version_key}</p>
        {source.source_document_id ? (
          <p className="break-all">Tài liệu: {source.source_document_id}</p>
        ) : null}
        {source.source_page ? <p>Trang {source.source_page}</p> : null}
        {source.chunk_id ? <p className="break-all">Mẫu chỉ mục: {source.chunk_id}</p> : null}
      </div>
    </details>
  );
}

/** One data cell: value + its own citation, or the explicit no-data string. */
function DataCell({ cell }: { cell: ComparisonCell | undefined }) {
  // A cell absent from the payload row (metric not reported for this major)
  // is missing data, not a zero and not the neighbour's number.
  if (!cell || cell.state !== 'present' || cell.value == null || cell.value === '') {
    return (
      <TableCell>
        <span data-testid="missing-cell">{NO_OFFICIAL_DATA}</span>
      </TableCell>
    );
  }
  return (
    <TableCell>
      <span className="font-medium tabular-nums">{cell.value}</span>
      {cell.note ? (
        <p className="mt-0.5 text-[11px] italic text-amber-600 dark:text-amber-400">{cell.note}</p>
      ) : null}
      <CellCitation cell={cell} />
    </TableCell>
  );
}

export function ComparisonTable({ response }: ComparisonTableProps) {
  const table = response.table;
  const rows = table.rows ?? [];
  const refused: RefusedMajor[] = [...(table.refused ?? []), ...(response.refused ?? [])];

  // Rule 3, FE-side: only rows carrying the table's own effective year join
  // the comparison block. Anything else is surfaced below, never merged.
  const uniformRows = rows.filter((row) => row.effective_year === table.effective_year);
  const mixedYearRows = rows.filter((row) => row.effective_year !== table.effective_year);

  // A cell is looked up by metric; rows carry cells in their own order.
  const cellFor = (row: ComparisonRow, metric: string): ComparisonCell | undefined =>
    (row.cells ?? []).find((cell) => cell.metric === metric);

  return (
    <div className="my-2 rounded-md border border-border bg-card p-3">
      <Table className="min-w-[560px]">
        <TableCaption>
          So sánh {uniformRows.length} ngành — tuyển sinh năm {table.effective_year}
        </TableCaption>
        <TableHeader>
          <TableRow>
            <TableHead scope="col" className="text-xs">
              Ngành
            </TableHead>
            {table.columns.map((column) => (
              <TableHead key={column.metric} scope="col" className="text-xs">
                {column.display}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {uniformRows.map((row) => (
            <TableRow key={row.record_version_key}>
              <TableHead
                scope="row"
                className="whitespace-normal align-top font-semibold text-foreground"
              >
                {row.major}
                <span className="block text-[11px] font-normal text-muted-foreground">
                  năm {row.effective_year}
                </span>
              </TableHead>
              {table.columns.map((column) => (
                <DataCell key={column.metric} cell={cellFor(row, column.metric)} />
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>

      {mixedYearRows.length > 0 ? (
        <div
          role="alert"
          className="mt-2 rounded-md border border-destructive/40 bg-destructive/10 p-2 text-xs text-destructive-foreground dark:text-red-100"
        >
          <p className="font-semibold">
            Không đưa vào bảng so sánh: {mixedYearRows.length} ngành mang năm hiệu lực khác{' '}
            {table.effective_year}
          </p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4">
            {mixedYearRows.map((row) => (
              <li key={row.record_version_key}>
                {row.major} — năm hiệu lực {row.effective_year}
              </li>
            ))}
          </ul>
          <p className="mt-1">
            Hệ thống không gộp ngành khác năm vào cùng một khối so sánh và không tự suy ra số liệu
            giữa các năm.
          </p>
        </div>
      ) : null}

      {refused.length > 0 ? (
        <div className="mt-2 rounded-md border border-amber-400/50 bg-amber-50 p-2 text-xs text-amber-900 dark:bg-amber-950/40 dark:text-amber-100">
          <p className="font-semibold">Ngoài bảng so sánh (theo hợp đồng trả về):</p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4">
            {refused.map((item) => (
              <li key={`${item.major}@${item.effective_year}`}>
                {item.major} — năm {item.effective_year}: {item.reason}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {response.route !== 'graph_rag' || response.intent !== 'so_sanh' ? (
        <p className="mt-2 text-[11px] italic text-muted-foreground">
          Cảnh báo: phản hồi không mang lộ trình so sánh chính thức (route={response.route},
          intent={response.intent}) — bảng có thể chưa được cấp bởi pipeline đa bước.
        </p>
      ) : null}
    </div>
  );
}
