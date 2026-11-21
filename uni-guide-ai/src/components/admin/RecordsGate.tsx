// The gate in front of the admissions CRUD screens — ticket 19.
//
// Sign-in is actor_id + password. The server answers with an HttpOnly session
// cookie, and that cookie is the only credential the UI holds. The provider
// lives in App.tsx, above every admin and teacher route.
import { useState, type FormEvent } from "react";
import { Outlet } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { useAdminSession } from "@/hooks/useAdminSession";

function LoginForm() {
  const { signIn, error } = useAdminSession();
  const [actorId, setActorId] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    try {
      await signIn(actorId, password);
    } finally {
      setPassword("");
      setSubmitting(false);
    }
  };

  return (
    <form className="w-full max-w-[420px] space-y-5" onSubmit={onSubmit} noValidate>
      <div className="space-y-2">
        <h2 className="text-4xl font-bold text-[#143980]">Đăng nhập vào khu quản trị</h2>
        <p className="text-base text-[#4a5568]">Dùng tài khoản được cấp để vào khu quản trị.</p>
      </div>
      <div className="space-y-1.5">
        <label htmlFor="admin-actor-id" className="block text-sm font-semibold text-[#0f1b2d]">
          Tên đăng nhập
        </label>
        <input
          id="admin-actor-id"
          type="text"
          autoComplete="username"
          className="min-h-11 w-full rounded-8 border border-[#d6deea] bg-white px-3 text-base text-[#0f1b2d]"
          value={actorId}
          aria-invalid={Boolean(error)}
          aria-describedby={error ? "admin-login-error" : undefined}
          onChange={(event) => setActorId(event.target.value)}
        />
      </div>
      <div className="space-y-1.5">
        <label htmlFor="admin-password" className="block text-sm font-semibold text-[#0f1b2d]">
          Mật khẩu
        </label>
        <input
          id="admin-password"
          type="password"
          autoComplete="current-password"
          className="min-h-11 w-full rounded-8 border border-[#d6deea] bg-white px-3 text-base text-[#0f1b2d]"
          value={password}
          aria-invalid={Boolean(error)}
          aria-describedby={error ? "admin-login-error" : undefined}
          onChange={(event) => setPassword(event.target.value)}
        />
        {error ? (
          <p id="admin-login-error" role="alert" className="text-sm text-[#9b1c1c]">
            {error}
          </p>
        ) : null}
      </div>
      <button
        type="submit"
        disabled={submitting}
        className="min-h-11 w-full rounded-8 bg-[#143980] px-4 text-base font-semibold text-white hover:bg-[#0f2d66] disabled:opacity-50"
      >
        Đăng nhập
      </button>
    </form>
  );
}

function RecordsSurface() {
  const { status, identity, error, signOut, retry } = useAdminSession();
  // Token-based so the gate reads correctly on both surfaces: the office shell
  // pins the light values, `/admin` follows the app theme.
  const muted = "text-muted-foreground";
  const heading = "text-foreground";
  const danger = "text-destructive";

  if (status === "anonymous") return <LoginForm />;

  if (status === "loading") {
    return (
      <p role="status" className={`text-xs ${muted}`}>
        Đang kiểm tra quyền truy cập…
      </p>
    );
  }

  if (status === "error" || identity === null) {
    return (
      <div className="space-y-3">
        <p role="alert" className={`text-xs ${danger}`}>
          {error ?? "Không kiểm tra được quyền truy cập."}
        </p>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" className="!border-zinc-500" onClick={retry}>
            Thử lại
          </Button>
          <Button size="sm" variant="ghost" onClick={() => void signOut()}>
            Đăng xuất
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <Button size="sm" variant="ghost" onClick={() => void signOut()}>
          Đăng xuất
        </Button>
      </div>
      <Outlet />
    </div>
  );
}

function RecordsGate() {
  return <RecordsSurface />;
}

export default RecordsGate;
