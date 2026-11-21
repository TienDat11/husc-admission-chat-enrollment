// Quản lý tài khoản & Phân quyền cán bộ — account management for administrators:
// list identities, create them (password and/or a one-time header token), change
// role, enable or disable, and change one's own password. The backend enforces
// every rule; this screen only shows its refusals in Vietnamese. Non-admins see a
// notice and no requests are sent.
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

const roleLabel = (role: string): string => ROLES.find((option) => option.value === role)?.label ?? role;

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
  const { can, identity } = useAdminSession();
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
  const [ownPassword, setOwnPassword] = useState("");
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
  const self = rows.find((row) => row.actor_id === identity?.actor_id) ?? null;
  const locked = rows.filter((row) => row.disabled);
  const active = rows.length - locked.length;

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

  const onChangeOwnPassword = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!self) return;
    setBusy(true);
    setBanner(null);
    try {
      await patchIdentity(null, self.subject_id, { reason: "Đổi mật khẩu cá nhân", password: ownPassword });
      setOwnPassword("");
      setBanner("Đã cập nhật mật khẩu của bạn.");
    } catch (error) {
      setBanner(errorText(error));
    } finally {
      setBusy(false);
    }
  };

  const roleCounts = ROLES.map((option) => ({
    label: option.label,
    count: rows.filter((row) => !row.disabled && row.role === option.value).length,
  })).filter((entry) => entry.count > 0);

  return (
    <section aria-labelledby="accounts-title" className="space-y-6 p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="max-w-2xl space-y-2">
          <h1 id="accounts-title" className="text-3xl font-bold text-[#0f1b2d]">
            Quản lý tài khoản &amp; Phân quyền cán bộ
          </h1>
          <p className="text-base text-[#4a5568]">
            Cấp tài khoản mới, phân quyền duyệt bài và mở khóa nhanh khi cán bộ nhập sai mật khẩu.
          </p>
        </div>
        <a
          href="#personal-password"
          className="inline-flex min-h-11 items-center rounded-8 border border-[#143980] bg-white px-4 text-base font-semibold text-[#143980] hover:bg-[#eef3fb]"
        >
          Đổi mật khẩu của tôi
        </a>
      </div>

      <ul className="m-0 grid list-none gap-4 p-0 md:grid-cols-3">
        <li className="space-y-2 rounded-xl border border-[#C9D8EE] bg-[#E6ECF7] p-5">
          <h2 className="text-lg font-semibold text-[#143980]">Đang hoạt động: {active} cán bộ</h2>
          <p className="text-sm text-[#2d4a7a]">
            {roleCounts.length > 0
              ? roleCounts.map((entry) => `${entry.label}: ${entry.count}`).join(" · ")
              : "Chưa có cán bộ nào đang hoạt động."}
          </p>
        </li>
        <li className="space-y-2 rounded-xl border border-[#F0D9A6] bg-[#FFF3DC] p-5">
          <h2 className="text-lg font-semibold text-[#8A5A00]">Tài khoản đang tạm khóa: {locked.length}</h2>
          <p className="text-sm text-[#5c4a1c]">
            {locked.length > 0 ? locked.map((row) => row.actor_id).join(", ") : "Không có tài khoản nào bị khóa."}
          </p>
        </li>
        <li className="space-y-2 rounded-xl border border-[#BFDCC4] bg-[#E7F4E8] p-5">
          <h2 className="text-lg font-semibold text-[#1E6B34]">Bảo vệ phiên đăng nhập</h2>
          <p className="text-sm text-[#2f5a3a]">
            Mọi thay đổi vai trò, khóa tài khoản và đổi mật khẩu đều được ghi vào nhật ký kiểm toán.
          </p>
        </li>
      </ul>

      {banner ? (
        <p role="status" className="rounded-lg border border-[#143980] bg-[#eef3fb] px-4 py-3 text-base">
          {banner}
        </p>
      ) : null}

      {createdToken ? (
        <div role="alert" className="space-y-2 rounded-lg border border-[#8A5A00] bg-[#FFF3DC] px-4 py-3">
          <p className="font-semibold">Mã truy cập một lần — chỉ hiển thị lúc này:</p>
          <code className="block break-all rounded bg-white px-2 py-1">{createdToken}</code>
          <p className="text-sm">Sao chép và gửi cho người được cấp. Sau khi đóng thông báo, mã không thể xem lại.</p>
          <button
            type="button"
            className="min-h-11 rounded-8 border border-[#8A5A00] bg-white px-4 text-sm font-semibold text-[#6b4300]"
            onClick={() => setCreatedToken(null)}
          >
            Đã sao chép, đóng
          </button>
        </div>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <section aria-labelledby="accounts-list" className="space-y-3 rounded-xl border border-[#d6deea] bg-white p-5">
          <h2 id="accounts-list" className="text-lg font-semibold text-[#0f1b2d]">
            Danh sách cán bộ
          </h2>
          {reasonMissing ? (
            <p className="text-sm text-[#8A5A00]">
              Nhập ghi chú lý do ở biểu mẫu bên phải trước khi đổi vai trò hoặc khóa tài khoản.
            </p>
          ) : null}
          {loadError ? (
            <p role="alert" className="text-base text-[#9b1c1c]">
              {loadError}
            </p>
          ) : null}
          <ul className="m-0 list-none space-y-2 p-0">
            {rows.map((row) => {
              const isSelf = row.subject_id === self?.subject_id;
              return (
                <li key={row.subject_id} className="space-y-2 rounded-lg bg-[#f7f9fc] p-4">
                  <div className="space-y-1">
                    <p className="text-base font-semibold">
                      {row.actor_id} · {roleLabel(row.role)}
                    </p>
                    <p className="text-sm text-[#4a5568]">
                      {row.disabled ? "Đang bị tạm khóa" : "Đang hoạt động"}
                      {row.last_login_at
                        ? ` · Đăng nhập gần nhất: ${new Date(row.last_login_at).toLocaleString("vi-VN")}`
                        : ""}
                    </p>
                    {isSelf ? (
                      <p className="text-sm text-[#143980]">
                        Tài khoản của bạn (không thể tự hạ quyền hoặc tự khóa)
                      </p>
                    ) : null}
                  </div>
                  {isSelf ? null : (
                    <div className="flex flex-wrap items-center gap-2">
                      <label className="sr-only" htmlFor={`role-${row.subject_id}`}>
                        Vai trò của {row.actor_id}
                      </label>
                      <select
                        id={`role-${row.subject_id}`}
                        className="min-h-11 rounded-8 border border-[#d6deea] bg-white px-3 text-sm"
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
                      <button
                        type="button"
                        disabled={busy || reasonMissing}
                        className="min-h-11 rounded-lg border border-[#143980] bg-white px-4 text-sm font-semibold text-[#143980] hover:bg-[#eef3fb] disabled:opacity-50"
                        onClick={() => void onToggleDisabled(row)}
                      >
                        {row.disabled ? "Mở khóa tài khoản ngay" : "Tạm ngừng tài khoản"}
                      </button>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </section>

        <div className="space-y-6">
          <form
            onSubmit={onCreate}
            aria-label="Cấp tài khoản mới"
            className="space-y-4 rounded-xl border border-[#d6deea] bg-white p-5"
          >
            <h2 className="text-lg font-semibold text-[#0f1b2d]">Cấp tài khoản mới / Đổi vai trò</h2>
            <div className="space-y-1.5">
              <label htmlFor="accounts-actor" className="block text-sm font-semibold">
                Tên đăng nhập
              </label>
              <input
                id="accounts-actor"
                required
                className="min-h-11 w-full rounded-lg border border-[#d6deea] bg-white px-3 text-base"
                value={actorId}
                onChange={(event) => setActorId(event.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <label htmlFor="accounts-role" className="block text-sm font-semibold">
                Vai trò
              </label>
              <select
                id="accounts-role"
                className="min-h-11 w-full rounded-lg border border-[#d6deea] bg-white px-3 text-base"
                value={role}
                onChange={(event) => setRole(event.target.value as Role)}
              >
                {ROLES.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1.5">
              <label htmlFor="accounts-password" className="block text-sm font-semibold">
                Mật khẩu (để trống nếu chỉ cấp mã truy cập)
              </label>
              <input
                id="accounts-password"
                type="password"
                autoComplete="new-password"
                className="min-h-11 w-full rounded-lg border border-[#d6deea] bg-white px-3 text-base"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <label htmlFor="accounts-reason" className="block text-sm font-semibold">
                Ghi chú lý do cấp hoặc đổi quyền (lưu vào nhật ký kiểm toán)
              </label>
              <input
                id="accounts-reason"
                className="min-h-11 w-full rounded-lg border border-[#d6deea] bg-white px-3 text-base"
                value={reason}
                onChange={(event) => setReason(event.target.value)}
              />
            </div>
            <label className="flex min-h-11 items-center gap-2 text-sm">
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
              className="min-h-11 w-full rounded-lg bg-[#143980] px-4 text-base font-semibold text-white hover:bg-[#0f2d66] disabled:opacity-50"
            >
              Lưu thông tin tài khoản →
            </button>
          </form>

          <section id="personal-password" aria-labelledby="own-password-title" className="space-y-3 rounded-xl border border-[#d6deea] bg-white p-5">
            <h2 id="own-password-title" className="text-lg font-semibold text-[#0f1b2d]">
              Đổi mật khẩu cá nhân
            </h2>
            <form onSubmit={onChangeOwnPassword} className="space-y-3">
              <div className="space-y-1.5">
                <label htmlFor="own-password" className="block text-sm font-semibold">
                  Mật khẩu mới
                </label>
                <input
                  id="own-password"
                  type="password"
                  autoComplete="new-password"
                  className="min-h-11 w-full rounded-lg border border-[#d6deea] bg-white px-3 text-base"
                  value={ownPassword}
                  onChange={(event) => setOwnPassword(event.target.value)}
                />
              </div>
              <button
                type="submit"
                disabled={busy || !self || ownPassword === ""}
                className="min-h-11 rounded-lg border border-[#143980] bg-white px-4 text-base font-semibold text-[#143980] hover:bg-[#eef3fb] disabled:opacity-50"
              >
                Cập nhật mật khẩu
              </button>
            </form>
          </section>
        </div>
      </div>
    </section>
  );
};

export default Accounts;
