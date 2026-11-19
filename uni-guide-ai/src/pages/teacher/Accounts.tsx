// U10 — account management for administrators: list identities, create them
// (password and/or a one-time header token), change role, enable or disable.
// The backend enforces every rule; this screen only shows its refusals in
// Vietnamese. Non-admins see a notice and no requests are sent.
import { useCallback, useEffect, useState, type FormEvent } from "react";
import { useAdminSession } from "@/hooks/useAdminSession";
import {
  AdminApiError,
  createIdentity,
  listIdentities,
  patchIdentity,
  type IdentityRow,
  type Role,
} from "@/lib/adminApi";

const ROLES: { value: Role; label: string }[] = [
  { value: "viewer", label: "Chỉ xem" },
  { value: "editor", label: "Người soạn" },
  { value: "reviewer", label: "Người duyệt" },
  { value: "publisher", label: "Người xuất bản" },
  { value: "reviewer_publisher", label: "Người duyệt và xuất bản" },
  { value: "admin", label: "Quản trị" },
  { value: "ops", label: "Kỹ thuật" },
];

function errorText(error: unknown): string {
  if (error instanceof AdminApiError) {
    if (error.message.includes("LAST_ADMIN")) {
      return "Không thể bỏ quyền hoặc khóa quản trị viên cuối cùng.";
    }
    if (error.message.includes("SELF_ROLE_CHANGE_FORBIDDEN")) {
      return "Bạn không thể tự đổi vai trò hoặc tự khóa tài khoản của mình.";
    }
    if (error.message.includes("IDENTITY_EXISTS")) {
      return "Tài khoản này đã tồn tại.";
    }
    if (error.status === 423) return "Tài khoản đang bị khóa tạm thời.";
    return error.message;
  }
  return "Không kết nối được máy chủ.";
}

const Accounts = () => {
  const { can } = useAdminSession();
  const canManage = can("roles.manage");

  const [rows, setRows] = useState<IdentityRow[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [banner, setBanner] = useState<string | null>(null);
  const [createdToken, setCreatedToken] = useState<string | null>(null);

  const [reason, setReason] = useState("");
  const [actorId, setActorId] = useState("");
  const [role, setRole] = useState<Role>("viewer");
  const [password, setPassword] = useState("");
  const [issueToken, setIssueToken] = useState(false);
  const [busy, setBusy] = useState(false);

  const reload = useCallback(async () => {
    try {
      setRows(await listIdentities(null));
      setLoadError(null);
    } catch (error) {
      setLoadError(errorText(error));
    }
  }, []);

  useEffect(() => {
    if (canManage) void reload();
  }, [canManage, reload]);

  if (!canManage) {
    return (
      <section aria-labelledby="accounts-title" className="p-6">
        <h1 id="accounts-title" className="text-2xl font-bold text-[#143980]">
          Tài khoản
        </h1>
        <p className="mt-4">Chỉ quản trị viên mới quản lý tài khoản.</p>
      </section>
    );
  }

  const reasonMissing = reason.trim() === "";

  const onCreate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setBusy(true);
    setBanner(null);
    try {
      const created = await createIdentity(null, {
        actor_id: actorId.trim(),
        role,
        reason: reason.trim(),
        ...(password ? { password } : {}),
        issue_header_token: issueToken,
      });
      setCreatedToken(created.header_token ?? null);
      setActorId("");
      setPassword("");
      setIssueToken(false);
      setBanner(`Đã tạo tài khoản ${created.actor_id}.`);
      await reload();
    } catch (error) {
      setBanner(errorText(error));
    } finally {
      setBusy(false);
    }
  };

  const onChangeRole = async (row: IdentityRow, nextRole: Role) => {
    setBusy(true);
    setBanner(null);
    try {
      await patchIdentity(null, row.subject_id, { reason: reason.trim(), role: nextRole });
      setBanner(`Đã đổi vai trò của ${row.actor_id}.`);
      await reload();
    } catch (error) {
      setBanner(errorText(error));
    } finally {
      setBusy(false);
    }
  };

  const onToggleDisabled = async (row: IdentityRow) => {
    setBusy(true);
    setBanner(null);
    try {
      await patchIdentity(null, row.subject_id, {
        reason: reason.trim(),
        disabled: !row.disabled,
      });
      setBanner(row.disabled ? `Đã mở khóa ${row.actor_id}.` : `Đã khóa ${row.actor_id}.`);
      await reload();
    } catch (error) {
      setBanner(errorText(error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section aria-labelledby="accounts-title" className="space-y-6 p-6">
      <h1 id="accounts-title" className="text-2xl font-bold text-[#143980]">
        Tài khoản
      </h1>

      <div className="space-y-2">
        <label htmlFor="accounts-reason" className="block font-semibold">
          Lý do thay đổi (bắt buộc)
        </label>
        <input
          id="accounts-reason"
          className="w-full max-w-xl border border-zinc-400 px-3 py-2"
          value={reason}
          onChange={(event) => setReason(event.target.value)}
        />
      </div>

      {banner ? (
        <p role="status" className="border border-[#143980] bg-[#eef3fb] px-3 py-2">
          {banner}
        </p>
      ) : null}

      {createdToken ? (
        <div role="alert" className="border border-amber-600 bg-amber-50 px-3 py-3">
          <p className="font-semibold">Mã truy cập một lần — chỉ hiển thị lúc này:</p>
          <code className="mt-2 block break-all">{createdToken}</code>
          <p className="mt-2">Sao chép và gửi cho người được cấp. Sau khi đóng thông báo, mã không thể xem lại.</p>
          <button
            type="button"
            className="mt-2 border border-zinc-500 px-3 py-1"
            onClick={() => setCreatedToken(null)}
          >
            Đã sao chép, đóng
          </button>
        </div>
      ) : null}

      <form onSubmit={onCreate} className="grid max-w-xl gap-3" aria-label="Tạo tài khoản">
        <h2 className="text-lg font-semibold">Tạo tài khoản</h2>
        <label className="grid gap-1">
          Tên đăng nhập
          <input
            required
            className="border border-zinc-400 px-3 py-2"
            value={actorId}
            onChange={(event) => setActorId(event.target.value)}
          />
        </label>
        <label className="grid gap-1">
          Vai trò
          <select
            className="border border-zinc-400 px-3 py-2"
            value={role}
            onChange={(event) => setRole(event.target.value as Role)}
          >
            {ROLES.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label className="grid gap-1">
          Mật khẩu (để trống nếu chỉ cấp mã truy cập)
          <input
            type="password"
            autoComplete="new-password"
            className="border border-zinc-400 px-3 py-2"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={issueToken}
            onChange={(event) => setIssueToken(event.target.checked)}
          />
          Cấp mã truy cập một lần
        </label>
        <button
          type="submit"
          disabled={busy || reasonMissing || (!password && !issueToken)}
          className="w-fit border border-[#143980] bg-[#143980] px-4 py-2 text-white disabled:opacity-50"
        >
          Tạo tài khoản
        </button>
      </form>

      <h2 className="text-lg font-semibold">Danh sách tài khoản</h2>
      {loadError ? <p role="alert">{loadError}</p> : null}
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-left">
          <thead>
            <tr className="border-b border-zinc-400">
              <th className="px-2 py-2">Tên đăng nhập</th>
              <th className="px-2 py-2">Vai trò</th>
              <th className="px-2 py-2">Trạng thái</th>
              <th className="px-2 py-2">Đăng nhập gần nhất</th>
              <th className="px-2 py-2">Thao tác</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.subject_id} className="border-b border-zinc-200">
                <td className="px-2 py-2">{row.actor_id}</td>
                <td className="px-2 py-2">
                  <label className="sr-only" htmlFor={`role-${row.subject_id}`}>
                    Vai trò của {row.actor_id}
                  </label>
                  <select
                    id={`role-${row.subject_id}`}
                    className="border border-zinc-400 px-2 py-1"
                    value={row.role}
                    disabled={busy || reasonMissing}
                    onChange={(event) => void onChangeRole(row, event.target.value as Role)}
                  >
                    {ROLES.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <span className="ml-2 text-sm text-zinc-600">{ROLES.find((option) => option.value === row.role)?.label ?? row.role}</span>
                </td>
                <td className="px-2 py-2">{row.disabled ? "Đã khóa" : "Hoạt động"}</td>
                <td className="px-2 py-2">
                  {row.last_login_at ? new Date(row.last_login_at).toLocaleString("vi-VN") : "—"}
                </td>
                <td className="px-2 py-2">
                  <button
                    type="button"
                    disabled={busy || reasonMissing}
                    className="border border-zinc-500 px-3 py-1 disabled:opacity-50"
                    onClick={() => void onToggleDisabled(row)}
                  >
                    {row.disabled ? "Mở khóa" : "Khóa"}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
};

export default Accounts;
