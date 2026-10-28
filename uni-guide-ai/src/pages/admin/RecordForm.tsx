// Admissions record form — create a record, or edit one into a NEW version —
// ticket 19.
//
// The edit path is the one that needs explaining on screen: saving never
// overwrites the version the office can see in the list. It appends version
// N+1 as a draft, so a published version stays exactly as published (and needs
// re-approval before it can be published again). The banner says that in the
// officer's own language rather than in lifecycle jargon.
//
// Validation is field-level and blocks the save: `validateRecordForm` runs
// before any request, and a 422 that does come back is mapped onto the field
// FastAPI names (`AdminApiError.fieldErrors`), so the message lands at the
// input either way.
import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useAdminSession } from "@/hooks/useAdminSession";
import {
  AdminApiError,
  createRecord,
  editRecord,
  fetchRecord,
  type CanonicalRecord,
  type LifecycleWriteResult,
} from "@/lib/adminApi";
import {
  AUTHORITY_LABELS,
  AUTHORITY_LEVELS,
  EMPTY_RECORD_FORM,
  RECORD_TYPES,
  RECORD_TYPE_LABELS,
  STATUS_LABELS,
  formValuesToEdit,
  formValuesToNewRecord,
  recordToFormValues,
  validateRecordForm,
  type RecordFieldErrors,
  type RecordFormValues,
} from "@/lib/adminValidation";

interface FieldProps {
  id: string;
  label: string;
  hint?: string;
  error?: string;
  children: ReactNode;
}

/** One labelled input plus its error, so "the error is at the field" is a
 * property of the markup rather than of where a banner happens to render. */
function Field({ id, label, hint, error, children }: FieldProps) {
  return (
    <div className="space-y-1">
      <Label htmlFor={id} className="text-xs text-foreground dark:text-zinc-300">
        {label}
      </Label>
      {children}
      {hint ? (
        <p id={`${id}-hint`} className="text-[11px] text-muted-foreground dark:text-zinc-400">
          {hint}
        </p>
      ) : null}
      {error ? (
        <p id={`${id}-error`} role="alert" className="text-xs text-red-600 dark:text-red-400">
          {error}
        </p>
      ) : null}
    </div>
  );
}

function fieldProps(id: string, error?: string) {
  return {
    id,
    "aria-invalid": Boolean(error),
    "aria-describedby": error ? `${id}-error` : undefined,
    className: "h-8 !border-zinc-500 text-xs",
  } as const;
}

const RecordForm = ({ mode }: { mode: "create" | "edit" }) => {
  const { can } = useAdminSession();
  const navigate = useNavigate();
  const params = useParams<{ recordId?: string }>();
  const recordId = params.recordId ? decodeURIComponent(params.recordId) : "";

  const [values, setValues] = useState<RecordFormValues>(EMPTY_RECORD_FORM);
  const [errors, setErrors] = useState<RecordFieldErrors>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [saved, setSaved] = useState<LifecycleWriteResult | null>(null);

  const current = useQuery({
    queryKey: ["admin", "record", recordId],
    queryFn: () => fetchRecord(null, recordId),
    enabled: mode === "edit" && Boolean(recordId),
    retry: false,
  });

  useEffect(() => {
    if (mode === "edit" && current.data) setValues(recordToFormValues(current.data));
  }, [mode, current.data]);

  const save = useMutation({
    mutationFn: (payload: Record<string, unknown>) =>
      mode === "create"
        ? createRecord(null, payload)
        : editRecord(null, recordId, payload),
    onSuccess: (result) => {
      setSaved(result);
      setErrors({});
      setFormError(null);
    },
    onError: (failure: unknown) => {
      if (failure instanceof AdminApiError) {
        setErrors(failure.fieldErrors as RecordFieldErrors);
        setFormError(failure.message);
        return;
      }
      setFormError("Không lưu được record.");
    },
  });

  const update = (field: keyof RecordFormValues, value: string) => {
    setValues((previous) => ({ ...previous, [field]: value }));
    setErrors((previous) => ({ ...previous, [field]: undefined }));
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    setFormError(null);
    const found = validateRecordForm(values);
    setErrors(found);
    if (Object.keys(found).length > 0) {
      // Blocked before the request: invalid data never reaches the store.
      const firstInvalid = Object.keys(found)[0];
      document.getElementById(firstInvalid)?.focus();
      return;
    }
    save.mutate(
      mode === "create" || current.data === undefined
        ? formValuesToNewRecord(values)
        : formValuesToEdit(values, current.data),
    );
  };

  if (mode === "edit" && !can("records.edit")) {
    return (
      <p role="alert" className="text-xs text-red-600 dark:text-red-400">
        Vai trò của bạn không có quyền sửa dữ liệu tuyển sinh.
      </p>
    );
  }

  if (mode === "create" && !can("records.create")) {
    return (
      <p role="alert" className="text-xs text-red-600 dark:text-red-400">
        Vai trò của bạn không có quyền tạo dữ liệu tuyển sinh.
      </p>
    );
  }

  if (saved) {
    return (
      <div className="max-w-2xl space-y-3">
        <p role="status" className="text-xs text-emerald-700 dark:text-emerald-400">
          Đã lưu {saved.record.record_id}@v{saved.record.version} —{" "}
          {STATUS_LABELS[saved.record.status] ?? saved.record.status}.
        </p>
        <p className="text-xs text-muted-foreground dark:text-zinc-400">
          {mode === "edit"
            ? "Đây là một version mới. Version đã công bố trước đó vẫn nguyên vẹn và không bị ghi đè; version mới cần được duyệt lại trước khi công bố."
            : "Record mới được lưu ở trạng thái nháp. Gửi duyệt để chuyển sang bước duyệt."}
        </p>
        <div className="flex gap-2">
          <Button size="sm" className="bg-blue-600 text-white" onClick={() => navigate("/admin/records")}>
            Về danh sách
          </Button>
          {mode === "edit" ? (
            <Button asChild size="sm" variant="outline" className="!border-zinc-500">
              <Link to={`/admin/records/${encodeURIComponent(recordId)}/edit`}>Sửa tiếp</Link>
            </Button>
          ) : null}
        </div>
      </div>
    );
  }

  const isEdit = mode === "edit";
  const loaded: CanonicalRecord | undefined = current.data;

  return (
    <form className="max-w-2xl space-y-4" onSubmit={submit} noValidate>
      {isEdit && loaded ? (
        <div className="rounded-sm border border-amber-600/40 bg-amber-500/5 p-3">
          <p className="text-xs text-amber-700 dark:text-amber-400">
            Lưu sẽ tạo <strong>version v{loaded.version + 1} (nháp)</strong>. Version hiện tại
            v{loaded.version} ({STATUS_LABELS[loaded.status] ?? loaded.status}) không bị ghi đè
            {loaded.status === "published"
              ? " — bản đã công bố vẫn nguyên vẹn cho tới khi bản mới được duyệt và công bố."
              : "."}
          </p>
        </div>
      ) : null}

      {isEdit && current.isPending ? (
        <p role="status" className="text-xs text-muted-foreground dark:text-zinc-400">
          Đang tải record…
        </p>
      ) : null}

      {isEdit && current.isError ? (
        <p role="alert" className="text-xs text-red-600 dark:text-red-400">
          {current.error instanceof AdminApiError
            ? `Không tải được record: ${current.error.message}`
            : "Không tải được record."}
        </p>
      ) : null}

      {formError ? (
        <p role="alert" className="text-xs text-red-600 dark:text-red-400">
          {formError}
        </p>
      ) : null}

      <Field
        id="record_id"
        label="Mã record"
        hint="Định danh duy nhất, ví dụ: CHI_TIEU:nganh:7480201:2026:ttts"
        error={errors.record_id}
      >
        <Input
          {...fieldProps("record_id", errors.record_id)}
          value={values.record_id}
          readOnly={isEdit}
          onChange={(event) => update("record_id", event.target.value)}
        />
      </Field>

      <Field id="record_type" label="Loại dữ liệu" error={errors.record_type}>
        <select
          {...fieldProps("record_type", errors.record_type)}
          value={values.record_type}
          disabled={isEdit}
          onChange={(event) => update("record_type", event.target.value)}
        >
          {RECORD_TYPES.map((type) => (
            <option key={type} value={type}>
              {RECORD_TYPE_LABELS[type] ?? type}
            </option>
          ))}
        </select>
      </Field>

      <div className="grid grid-cols-2 gap-3">
        <Field id="admission_year" label="Năm tuyển sinh" error={errors.admission_year}>
          <Input
            {...fieldProps("admission_year", errors.admission_year)}
            type="number"
            value={values.admission_year}
            onChange={(event) => update("admission_year", event.target.value)}
          />
        </Field>

        <Field id="authority_level" label="Mức thẩm quyền nguồn" error={errors.authority_level}>
          <select
            {...fieldProps("authority_level", errors.authority_level)}
            value={values.authority_level}
            onChange={(event) => update("authority_level", event.target.value)}
          >
            {AUTHORITY_LEVELS.map((level) => (
              <option key={level} value={level}>
                {AUTHORITY_LABELS[level] ?? level}
              </option>
            ))}
          </select>
        </Field>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <Field
          id="effective_from"
          label="Hiệu lực từ"
          hint="YYYY-MM-DD"
          error={errors.effective_from}
        >
          <Input
            {...fieldProps("effective_from", errors.effective_from)}
            type="date"
            value={values.effective_from}
            onChange={(event) => update("effective_from", event.target.value)}
          />
        </Field>

        <Field
          id="effective_to"
          label="Hiệu lực đến"
          hint="Để trống nếu chưa có mốc kết thúc."
          error={errors.effective_to}
        >
          <Input
            {...fieldProps("effective_to", errors.effective_to)}
            type="date"
            value={values.effective_to}
            onChange={(event) => update("effective_to", event.target.value)}
          />
        </Field>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <Field id="ma_nganh" label="Mã ngành" hint="Dùng cho bộ lọc theo ngành.">
          <Input
            {...fieldProps("ma_nganh")}
            value={values.ma_nganh}
            onChange={(event) => update("ma_nganh", event.target.value)}
          />
        </Field>

        <Field id="ten_nganh" label="Tên ngành" hint="Dùng cho bộ lọc theo ngành.">
          <Input
            {...fieldProps("ten_nganh")}
            value={values.ten_nganh}
            onChange={(event) => update("ten_nganh", event.target.value)}
          />
        </Field>
      </div>

      <Field
        id="source_document_id"
        label="Tài liệu nguồn"
        hint="Bắt buộc để công bố: một record không có nguồn sẽ không publish được."
        error={errors.source_document_id}
      >
        <Input
          {...fieldProps("source_document_id", errors.source_document_id)}
          value={values.source_document_id}
          onChange={(event) => update("source_document_id", event.target.value)}
        />
      </Field>

      <div className="grid grid-cols-2 gap-3">
        <Field id="source_url" label="URL nguồn" error={errors.source_url}>
          <Input
            {...fieldProps("source_url", errors.source_url)}
            value={values.source_url}
            onChange={(event) => update("source_url", event.target.value)}
          />
        </Field>

        <Field id="source_page" label="Trang nguồn" error={errors.source_page}>
          <Input
            {...fieldProps("source_page", errors.source_page)}
            type="number"
            value={values.source_page}
            onChange={(event) => update("source_page", event.target.value)}
          />
        </Field>
      </div>

      <Field id="text" label="Nội dung" error={errors.text}>
        <Textarea
          {...fieldProps("text", errors.text)}
          className="min-h-[100px] !border-zinc-500 text-xs"
          value={values.text}
          aria-invalid={Boolean(errors.text)}
          aria-describedby={errors.text ? "text-error" : undefined}
          onChange={(event) => update("text", event.target.value)}
        />
      </Field>

      <div className="flex gap-2">
        <Button type="submit" size="sm" className="bg-blue-600 text-white" disabled={save.isPending}>
          {isEdit && loaded ? `Lưu thành version v${loaded.version + 1}` : "Lưu record mới"}
        </Button>
        <Button asChild size="sm" variant="ghost">
          <Link to="/admin/records">Hủy</Link>
        </Button>
      </div>
    </form>
  );
};

export default RecordForm;
