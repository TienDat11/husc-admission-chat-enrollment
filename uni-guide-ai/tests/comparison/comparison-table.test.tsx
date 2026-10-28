// Ticket 24 acceptance at the UI seam: the multi-major comparison table.
//
// Each test defends a checklist item that would otherwise be a claim about
// a screenshot:
//
//   * a cell with no official data renders "Chưa có thông tin chính thức"
//     and is NOT back-filled from the neighbouring major (criterion 1+3);
//   * every present cell carries its own openable citation with the record
//     version key and, when the contract has one, the official URL
//     (criterion 2) — the citation lives inside the data cell itself, so
//     no responsive rule can drop it without dropping the number too
//     (criterion 5);
//   * column headers are programmatically associated (`scope="col"`) and
//     each major labels its row (`scope="row"`) for screen readers
//     (criterion 4);
//   * a row whose effective_year differs from the table's is EXCLUDED from
//     the comparison block and surfaced in an explicit alert, never merged
//     silently (criterion 6); refused majors from the contract are listed
//     with their reasons;
//   * compareMajors() POSTs the generated CompareRequest shape to
//     /v2/compare and raises the server's `detail` on failure.
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ComparisonTable, NO_OFFICIAL_DATA } from '@/components/comparison/ComparisonTable';
import { compareMajors, type ComparisonResponsePayload } from '@/lib/api';
import type { components } from '@/lib/api-types';

type Row = components['schemas']['ComparisonRow'];
type Cell = components['schemas']['ComparisonCell'];

const cell = (
  major: string,
  metric: string,
  value: string | null,
  sourceKey?: string,
): Cell => ({
  major,
  metric,
  state: value === null ? 'missing' : 'present',
  value,
  source: sourceKey
    ? {
        record_version_key: sourceKey,
        source_url: `https://ts.husc.edu.vn/thong-bao/${sourceKey}`,
        source_document_id: 'docs/thong-bao-2026.pdf',
        source_page: 3,
        chunk_id: 'chunk-abc',
      }
    : null,
});

const row = (major: string, year: number, cells: Cell[]): Row => ({
  major,
  effective_year: year,
  record_version_key: `${major.toLowerCase().replace(/\s/g, '-')}@v3`,
  cells,
});

const payload = (rows: Row[], overrides?: { refused?: components['schemas']['RefusedMajor'][] }): ComparisonResponsePayload => ({
  effective_year: 2026,
  intent: 'so_sanh',
  query: 'so sánh các ngành năm 2026',
  route: 'graph_rag',
  refused: overrides?.refused,
  table: {
    effective_year: 2026,
    query: 'so sánh các ngành năm 2026',
    columns: [
      { metric: 'diem_chuan', display: 'Điểm chuẩn' },
      { metric: 'hoc_phi', display: 'Học phí' },
    ],
    rows,
  },
});

const CNTT = row('Công nghệ thông tin', 2026, [
  cell('Công nghệ thông tin', 'diem_chuan', '24.5', 'cntt@7'),
  cell('Công nghệ thông tin', 'hoc_phi', '16.000.000đ/năm', 'cntt@7'),
]);
const KTPM_MISSING_DIEM = row('Khoa học máy tính', 2026, [
  // The backend marked this cell missing: no value, no provenance.
  cell('Khoa học máy tính', 'diem_chuan', null),
  cell('Khoa học máy tính', 'hoc_phi', '14.500.000đ/năm', 'khmt@4'),
]);

const rowHeader = (name: RegExp) => screen.getByRole('rowheader', { name });
const dataRows = () => screen.getAllByRole('row').filter((r) => within(r).queryAllByRole('rowheader').length > 0);

describe('ComparisonTable — thiếu dữ liệu không suy diễn (criteria 1, 3)', () => {
  it('renders the Vietnamese no-official-data string and never back-fills the neighbour value', () => {
    render(<ComparisonTable response={payload([CNTT, KTPM_MISSING_DIEM])} />);
    const ktpm = within(dataRows().find((r) => within(r).queryByRole('rowheader', { name: /Khoa học/ }))!);
    const missingCell = ktpm.getByText(NO_OFFICIAL_DATA).closest('td')!;
    // CNTT's điểm chuẩn must NOT leak into the KTPM row.
    expect(within(missingCell).queryByText('24.5')).toBeNull();
    // A missing cell has nothing to cite — no fake citation affordance.
    expect(within(missingCell).queryByText(/Xem nguồn/)).toBeNull();
  });

  it('treats a malformed present-with-null-value cell as missing, not zero', () => {
    const broken = row('Hệ thống nhúng', 2026, [
      { ...cell('Hệ thống nhúng', 'diem_chuan', 'x'), state: 'present', value: null },
      cell('Hệ thống nhúng', 'hoc_phi', '15.000.000đ/năm', 'htnh@2'),
    ]);
    render(<ComparisonTable response={payload([broken])} />);
    const r = within(dataRows()[0]);
    const missingCell = r.getByText(NO_OFFICIAL_DATA).closest('td')!;
    expect(within(missingCell).queryByText(/Xem nguồn/)).toBeNull();
  });

  it('shows a metric the row omits entirely as missing', () => {
    const noHocPhi = row('Kỹ thuật phần mềm', 2026, [
      cell('Kỹ thuật phần mềm', 'diem_chuan', '23.0', 'ktpm@1'),
    ]);
    render(<ComparisonTable response={payload([noHocPhi])} />);
    expect(within(dataRows()[0]).getByText(NO_OFFICIAL_DATA)).toBeInTheDocument();
  });
});

describe('ComparisonTable — citation từng ô (criteria 2, 5)', () => {
  it('each present cell has its own citation opening onto the official document', () => {
    render(<ComparisonTable response={payload([CNTT])} />);
    const toggles = screen.getAllByText(/Xem nguồn/);
    // Two present cells => two independent citation affordances.
    expect(toggles).toHaveLength(2);
    const firstToggle = toggles[0];
    // The citation is INSIDE its data cell, so no layout can hide it
    // without hiding the number it vouches for.
    expect(firstToggle.closest('td')).not.toBeNull();
    expect(within(firstToggle.closest('td')!).getByText('24.5')).toBeInTheDocument();
    const details = firstToggle.closest('details')!;
    expect(details).toBeInTheDocument();
    within(details).getByText('Bản ghi: cntt@7');
    const link = within(details).getByRole('link', { name: /Tài liệu chính thức/ });
    expect(link).toHaveAttribute('href', 'https://ts.husc.edu.vn/thong-bao/cntt@7');
    expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'));
    within(details).getByText('Trang 3');
  });

  it('flags a present cell whose payload carries no provenance instead of showing an unverifiable number', () => {
    const noSource = row('An toàn thông tin', 2026, [
      { ...cell('An toàn thông tin', 'diem_chuan', '22.5'), source: null },
      cell('An toàn thông tin', 'hoc_phi', '15.500.000đ/năm', 'atto@1'),
    ]);
    render(<ComparisonTable response={payload([noSource])} />);
    expect(screen.getByText(/chưa kèm nguồn để kiểm chứng/)).toBeInTheDocument();
    // only the sourced cell keeps a citation toggle
    expect(screen.getAllByText(/Xem nguồn/)).toHaveLength(1);
  });

  it('citation toggle is focusable and opens on activation', async () => {
    render(<ComparisonTable response={payload([CNTT])} />);
    const summary = screen.getAllByText(/Xem nguồn/)[0].closest('summary');
    summary?.focus();
    expect(document.activeElement).toBe(summary);
    await userEvent.click(summary!);
    expect(summary?.parentElement).toHaveAttribute('open');
  });
});

describe('ComparisonTable — association header (criterion 4)', () => {
  it('column headers carry scope=col and each major labels its row with scope=row', () => {
    render(<ComparisonTable response={payload([CNTT, KTPM_MISSING_DIEM])} />);
    const colHeads = screen.getAllByRole('columnheader');
    expect(colHeads.length).toBeGreaterThan(0);
    for (const head of colHeads) {
      expect(head).toHaveAttribute('scope', 'col');
    }
    expect(within(screen.getByRole('columnheader', { name: 'Điểm chuẩn' })).getByText('Điểm chuẩn')).toBeInTheDocument();
    const rowHeads = screen.getAllByRole('rowheader');
    expect(rowHeads).toHaveLength(2);
    for (const head of rowHeads) {
      expect(head).toHaveAttribute('scope', 'row');
    }
  });
});

describe('ComparisonTable — một năm hiệu lực mỗi khối (criterion 6)', () => {
  it('excludes a different-year row from the block and surfaces it explicitly', () => {
    const Y2025 = row('Y đa học', 2025, [
      cell('Y đa học', 'diem_chuan', '26.0', 'ydh@9'),
      cell('Y đa học', 'hoc_phi', '40.000.000đ/năm', 'ydh@9'),
    ]);
    render(<ComparisonTable response={payload([CNTT, Y2025])} />);
    // The 2025 major is NOT a comparison row.
    expect(screen.queryByRole('rowheader', { name: /Y đa học/ })).toBeNull();
    // Its 2025 numbers never sit in the 2026 block.
    for (const r of dataRows()) {
      expect(within(r).queryByText('26.0')).toBeNull();
    }
    // ...and the mismatch is stated, not silent.
    const alert = screen.getByRole('alert');
    expect(alert).toHaveTextContent('Y đa học');
    expect(alert).toHaveTextContent('2025');
    expect(alert).toHaveTextContent('không gộp ngành khác năm');
    // The caption counts only the in-year rows.
    expect(screen.getByText(/So sánh 1 ngành — tuyển sinh năm 2026/)).toBeInTheDocument();
  });

  it('lists refused majors returned by the contract with their reasons', () => {
    render(
      <ComparisonTable
        response={payload([CNTT], {
          refused: [
            {
              major: 'Quản trị kinh doanh',
              effective_year: 2025,
              reason: 'khác năm hiệu lực với nhóm 2026',
            },
          ],
        })}
      />,
    );
    expect(screen.getByText(/Quản trị kinh doanh — năm 2025/)).toBeInTheDocument();
    expect(screen.getByText(/khác năm hiệu lực với nhóm 2026/)).toBeInTheDocument();
    expect(rowHeader(/Công nghệ thông tin/)).toBeInTheDocument();
  });
});

describe('compareMajors — API client', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it('POSTs the generated CompareRequest shape to /v2/compare', async () => {
    const response = payload([CNTT]);
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(response), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const out = await compareMajors({
      query: 'so sánh các ngành năm 2026',
      effective_year: 2026,
      intent: 'so_sanh',
      route: 'graph_rag',
      record_ids: ['cntt', 'khmt'],
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/v2\/compare$/);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual(
      expect.objectContaining({ record_ids: ['cntt', 'khmt'], effective_year: 2026 }),
    );
    expect(out.table.rows?.[0]?.major).toBe('Công nghệ thông tin');
  });

  it('raises the server detail on a contract violation', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: 'không tìm thấy ngành phù hợp' }), {
          status: 404,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );
    await expect(
      compareMajors({ query: 'q', effective_year: 2026, intent: 'so_sanh', route: 'graph_rag' }),
    ).rejects.toThrow('không tìm thấy ngành phù hợp');
  });
});
