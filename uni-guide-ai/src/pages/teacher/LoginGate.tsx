// Đăng nhập của khu quản lý. A 401 elsewhere sends the user here with
// `?next=` (apiError.ts), and a successful sign-in returns them there.
import { useEffect, useState, type FormEvent } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useAdminSession } from "@/hooks/useAdminSession";
import { AdminApiError, loginAdmin } from "@/lib/adminApi";
import type { ApiFailure } from "@/lib/apiError";
import { ExceptionBar } from "@/components/ui-kit/ExceptionBar";
import { PrimaryButton } from "@/components/ui-kit/PrimaryButton";

const DEFAULT_NEXT = "/quan-ly";

/** Demo accounts shown on the sign-in page. Only the user name is filled in;
 *  the password is never shown here. */
const DEMO_ACCOUNTS = [
  { actorId: "biensoan", role: "Người soạn tài liệu" },
  { actorId: "duyetbai", role: "Người duyệt và xuất bản" },
  { actorId: "thayco", role: "Quản trị viên" },
] as const;

/** Only in-app admin paths: a `next` value never leaves the site. */
function safeNext(raw: string | null): string {
  return raw && /^\/(quan-ly|admin)(\/|$)/.test(raw) ? raw : DEFAULT_NEXT;
}

const LoginGate = () => {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const session = useAdminSession();
  const next = safeNext(params.get("next"));

  const [actorId, setActorId] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  const [remaining, setRemaining] = useState(0);

  // Lockout countdown: the server's Retry-After, counted down once a second.
  useEffect(() => {
    if (failure?.kind !== "locked") return;
    setRemaining(failure.retryAfterSeconds);
    const timer = window.setInterval(() => setRemaining((s) => Math.max(0, s - 1)), 1000);
    return () => window.clearInterval(timer);
  }, [failure]);

  const locked = remaining > 0;

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setBusy(true);
    setFailure(null);
    try {
      await loginAdmin(actorId.trim(), password);
      session.retry();
      navigate(next, { replace: true });
    } catch (error) {
      setFailure(
        error instanceof AdminApiError
          ? error.failure
          : { kind: "error", status: 0, code: null, message: "Không kết nối được máy chủ." },
      );
      setPassword("");
    } finally {
      setBusy(false);
    }
  };

  return (
    <main id="main-content" className="husc-admin flex min-h-screen bg-white text-[#0f1b2d]">
      <aside className="hidden w-[520px] shrink-0 flex-col justify-between bg-[#143980] p-12 text-white lg:flex">
        <div className="space-y-3">
          <p className="text-xl font-bold">Trường Đại học Khoa học, ĐH Huế</p>
          <p className="text-sm opacity-90">Hệ thống tuyển sinh · Quản trị dữ liệu Bot</p>
        </div>
        <p className="text-3xl font-bold leading-tight">Một nguồn chuẩn duy nhất cho Bot tuyển sinh.</p>
        <p className="text-sm opacity-90">Mọi thay đổi đều được ghi lại trong nhật ký để kiểm tra.</p>
      </aside>

      <div className="flex flex-1 items-center justify-center px-4 py-10">
        <section data-workarea="" aria-labelledby="login-title" className="w-full max-w-[360px] space-y-5">
          <div className="space-y-2">
            <h1 id="login-title" className="text-3xl font-bold text-[#143980]">
              Đăng nhập
            </h1>
            <p className="text-base text-[#4a5568]">Dùng tài khoản được cấp để vào khu quản trị.</p>
          </div>

          {failure?.kind === "locked" ? (
            <ExceptionBar
              title="Tài khoản bị khoá tạm thời"
              detail={locked ? `Thử lại sau ${remaining} giây.` : "Bạn có thể thử đăng nhập lại."}
            />
          ) : null}
          {failure?.kind === "account-disabled" ? (
            <ExceptionBar title={failure.message} />
          ) : null}
          {failure?.kind === "session-expired" ? (
            <ExceptionBar title="Sai tài khoản hoặc mật khẩu." />
          ) : null}
          {failure?.kind === "error" ? <ExceptionBar title={failure.message} /> : null}

          <form onSubmit={onSubmit} className="space-y-4" noValidate>
            <div className="space-y-1">
              <label htmlFor="login-actor" className="block text-sm font-semibold">
                Tên đăng nhập
              </label>
              <input
                id="login-actor"
                name="actor_id"
                autoComplete="username"
                value={actorId}
                onChange={(event) => setActorId(event.target.value)}
                disabled={busy || locked}
                className="min-h-11 w-full rounded-lg border border-[#d6deea] bg-white px-3 text-base"
              />
            </div>
            <div className="space-y-1">
              <label htmlFor="login-password" className="block text-sm font-semibold">
                Mật khẩu
              </label>
              <input
                id="login-password"
                name="password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                disabled={busy || locked}
                className="min-h-11 w-full rounded-lg border border-[#d6deea] bg-white px-3 text-base"
              />
            </div>
            <PrimaryButton type="submit" disabled={busy || locked} className="w-full">
              Đăng nhập
            </PrimaryButton>
          </form>

          <div className="space-y-3 border-t border-[#e3e8f0] pt-5">
            <p className="text-center text-sm text-[#4a5568]">Hoặc vào nhanh bằng tài khoản demo</p>
            <ul className="m-0 list-none space-y-2 p-0">
              {DEMO_ACCOUNTS.map((account) => (
                <li key={account.actorId}>
                  <button
                    type="button"
                    onClick={() => setActorId(account.actorId)}
                    disabled={busy || locked}
                    className="flex min-h-11 w-full items-center gap-3 rounded-lg border border-[#d6deea] bg-white p-3 text-left hover:bg-[#f7f9fc]"
                  >
                    <span aria-hidden="true" className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[#143980] text-sm font-bold text-white">
                      {account.actorId.charAt(0).toUpperCase()}
                    </span>
                    <span>
                      <span className="block text-sm font-semibold">{account.actorId}</span>
                      <span className="block text-sm text-[#4a5568]">{account.role}</span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
            <p className="text-sm text-[#4a5568]">Tài khoản demo chỉ dùng để thử nghiệm, không dùng cho dữ liệu thật.</p>
          </div>

          <address className="space-y-1 pt-2 text-sm not-italic text-[#4a5568]">
            <p>77 Nguyễn Huệ, Phường Thuận Hoà, TP. Huế</p>
            <p>ĐT: (0234) 3828 427 · Zalo: 0385 887 111</p>
          </address>
        </section>
      </div>
    </main>
  );
};

export default LoginGate;
