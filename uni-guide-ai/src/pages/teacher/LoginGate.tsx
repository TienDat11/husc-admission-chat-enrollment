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
    <main id="main-content" className="husc-admin flex min-h-screen items-center justify-center bg-white px-4 py-8 text-zinc-900">
      <section data-workarea="" aria-labelledby="login-title" className="w-full max-w-sm space-y-5">
        <h1 id="login-title" className="text-2xl font-bold text-[#143980]">
          Đăng nhập quản lý
        </h1>

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
            <label htmlFor="login-actor" className="block text-base font-semibold">
              Tên đăng nhập
            </label>
            <input
              id="login-actor"
              name="actor_id"
              autoComplete="username"
              value={actorId}
              onChange={(event) => setActorId(event.target.value)}
              disabled={busy || locked}
              className="min-h-11 w-full rounded-md border border-zinc-500 px-3 text-base"
            />
          </div>
          <div className="space-y-1">
            <label htmlFor="login-password" className="block text-base font-semibold">
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
              className="min-h-11 w-full rounded-md border border-zinc-500 px-3 text-base"
            />
          </div>
          <PrimaryButton type="submit" disabled={busy || locked} className="w-full">
            Đăng nhập
          </PrimaryButton>
        </form>
      </section>
    </main>
  );
};

export default LoginGate;
