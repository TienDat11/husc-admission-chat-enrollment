// The gate in front of the admissions CRUD screens — ticket 19.
//
// Sign-in is actor_id + password. The server answers with an HttpOnly session
// cookie, and that cookie is the only credential the UI holds. The provider
// lives in App.tsx, above every admin and teacher route.
import { useState, type FormEvent } from "react";
import { Outlet } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
    <form className="max-w-sm space-y-3" onSubmit={onSubmit}>
      <div className="space-y-1">
        <h2 className="text-sm font-medium text-foreground dark:text-zinc-100">Dữ liệu tuyển sinh</h2>
        <p className="text-xs text-muted-foreground dark:text-zinc-400">
          Đăng nhập bằng tài khoản quản trị do quản trị viên cấp.
        </p>
      </div>
      <div className="space-y-1">
        <Label htmlFor="admin-actor-id" className="text-xs text-foreground dark:text-zinc-300">
          Tài khoản
        </Label>
        <Input
          id="admin-actor-id"
          type="text"
          autoComplete="username"
          className="!border-zinc-500"
          value={actorId}
          aria-invalid={Boolean(error)}
          aria-describedby={error ? "admin-login-error" : undefined}
          onChange={(event) => setActorId(event.target.value)}
        />
      </div>
      <div className="space-y-1">
        <Label htmlFor="admin-password" className="text-xs text-foreground dark:text-zinc-300">
          Mật khẩu
        </Label>
        <Input
          id="admin-password"
          type="password"
          autoComplete="current-password"
          className="!border-zinc-500"
          value={password}
          aria-invalid={Boolean(error)}
          aria-describedby={error ? "admin-login-error" : undefined}
          onChange={(event) => setPassword(event.target.value)}
        />
        {error ? (
          <p id="admin-login-error" role="alert" className="text-xs text-destructive">
            {error}
          </p>
        ) : null}
      </div>
      <Button type="submit" size="sm" className="bg-blue-600 text-white" disabled={submitting}>
        Đăng nhập
      </Button>
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
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 space-y-1">
          <h2 className={`text-sm font-medium ${heading}`}>Dữ liệu tuyển sinh</h2>
          <p className={`text-xs ${muted}`}>
            Đăng nhập với vai trò <span className="font-mono break-all">{identity.role}</span> (
            {identity.actor_id}). Chỉ những hành động vai trò này được cấp mới hiển thị.
          </p>
        </div>
        <Button size="sm" variant="ghost" onClick={() => void signOut()}>
          Đăng xuất
        </Button>
      </div>
      <Outlet />
    </div>
  );
}

const RecordsGate = () => <RecordsSurface />;

export default RecordsGate;
