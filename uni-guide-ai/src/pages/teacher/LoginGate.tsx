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
  { actorId: "biensoan", title: "Cán bộ Tuyển sinh", detail: "Thêm thông tin, sửa nháp, gửi duyệt" },
  { actorId: "duyetbai", title: "Quản trị viên Duyệt", detail: "Duyệt độc lập và phát hành lên Bot" },
  { actorId: "thayco", title: "Quản trị viên", detail: "Quản trị toàn bộ hệ thống" },
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
      <aside className="hidden w-[600px] shrink-0 flex-col justify-between bg-[#103157] p-14 text-white lg:flex">
        <div className="space-y-3">
          <p className="text-xl font-bold">Trường Đại học Khoa học, ĐH Huế</p>
          <p className="text-sm opacity-90">Hệ thống tuyển sinh · Quản trị dữ liệu Bot</p>
          <p className="pt-10 text-4xl font-bold leading-tight">Một nguồn chuẩn duy nhất cho Bot tuyển sinh.</p>
        </div>
        <p className="text-base leading-relaxed opacity-90">
          Mọi điều khoản được kiểm tra, duyệt độc lập bởi hai người và có vết kiểm toán trước khi phát hành lên Bot.
        </p>
      </aside>

      <div className="flex flex-1 items-center justify-center px-6 py-10">
        <section data-workarea="" aria-labelledby="login-title" className="w-full max-w-[420px] space-y-6">
          <div className="space-y-2">
            <h1 id="login-title" className="text-4xl font-bold text-[#143980]">
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
          {failure?.kind === "account-disabled" ? <ExceptionBar title={failure.message} /> : null}
          {failure?.kind === "session-expired" ? <ExceptionBar title="Sai tài khoản hoặc mật khẩu." /> : null}
          {failure?.kind === "error" ? <ExceptionBar title={failure.message} /> : null}

          <form onSubmit={onSubmit} className="space-y-5" noValidate>
            <div className="space-y-1.5">
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
                className="min-h-11 w-full rounded-8 border border-[#d6deea] bg-white px-3 text-base"
              />
            </div>
            <div className="space-y-1.5">
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
                className="min-h-11 w-full rounded-8 border border-[#d6deea] bg-white px-3 text-base"
              />
            </div>
            <PrimaryButton type="submit" disabled={busy || locked} className="w-full">
              Đăng nhập →
            </PrimaryButton>
          </form>

          <div className="space-y-3 border-t border-[#e3e8f0] pt-6">
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
                    <span
                      aria-hidden="true"
                      className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-[#143980] text-sm font-bold text-white"
                    >
                      {account.actorId.charAt(0).toUpperCase()}
                    </span>
                    <span>
                      <span className="block text-sm font-semibold">
                        {account.title} · {account.actorId}
                      </span>
                      <span className="block text-sm text-[#4a5568]">{account.detail}</span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
            <p className="text-sm text-[#4a5568]">Tài khoản dùng thử cho đợt kiểm thử nội bộ.</p>
          </div>

          <address className="space-y-1 pt-2 text-sm not-italic text-[#4a5568]">
            <p>77 Nguyễn Huệ, Phường Thuận Hoà, TP. Huế</p>
            <p>ĐT: (0234) 3828 427 · Zalo: 0385 887 111</p>
            <p>
              <a className="font-semibold text-[#143980] underline" href="mailto:tuyensinh@husc.edu.vn">
                tuyensinh@husc.edu.vn
              </a>
            </p>
            <p>
              <a
                className="font-semibold text-[#143980] underline"
                href="https://tuyensinh.husc.edu.vn"
                target="_blank"
                rel="noreferrer"
              >
                tuyensinh.husc.edu.vn
              </a>
            </p>
          </address>
        </section>
      </div>
    </main>
  );
};

export default LoginGate;
