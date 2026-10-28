//
// The credential is the HttpOnly `admin_session` cookie set by POST /admin/login.
// Nothing here reads or stores a token. A reload restores the session by probing
// GET /admin/identity with the cookie; a 401 there means "not signed in".
//
// The UI never decides what a role may do. It asks the backend
// (`GET /admin/identity`), which answers from the same role x action matrix the
// routes enforce (`src/domain/admin_identity.py`).
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  AdminApiError,
  isMissingCredential,
  fetchAdminIdentity,
  loginAdmin,
  logoutAdmin,
  type AdminIdentity,
} from "@/lib/adminApi";

export type AdminSessionStatus = "anonymous" | "loading" | "ready" | "error";

export interface AdminSession {
  status: AdminSessionStatus;
  identity: AdminIdentity | null;
  /** Message for the gate to show: a failed sign-in, or a failed session check. */
  error: string | null;
  /** Whether the caller's role holds `action` (e.g. "records.publish"). */
  can: (action: string) => boolean;
  signIn: (actorId: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  retry: () => void;
}

const AdminSessionContext = createContext<AdminSession | null>(null);

/** Lockout length the backend applies after five failed sign-ins (15 minutes). */
const LOCKOUT_MINUTES = 15;

function signInMessage(failure: unknown): string {
  if (failure instanceof AdminApiError) {
    if (failure.status === 401) return "Sai tài khoản hoặc mật khẩu.";
    if (failure.status === 423) {
      return `Tài khoản bị khóa tạm thời, thử lại sau ${LOCKOUT_MINUTES} phút.`;
    }
  }
  return "Không kết nối được máy chủ.";
}

export function useAdminSession(): AdminSession {
  const session = useContext(AdminSessionContext);
  if (session === null) {
    throw new Error("useAdminSession must be used inside <AdminSessionProvider>");
  }
  return session;
}

/**
 * Mounts the session state. Split from the public provider so a nested
 * `AdminSessionProvider` can short-circuit BEFORE this hook body runs — a
 * provider that reuses an outer session must not also fetch `/admin/identity`.
 */
function AdminSessionState({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const [identity, setIdentity] = useState<AdminIdentity | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<AdminSessionStatus>("loading");
  const [attempt, setAttempt] = useState(0);

  // Restore on mount (and on retry): the cookie, if any, answers /admin/identity.
  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    fetchAdminIdentity(null)
      .then((resolved) => {
        if (cancelled) return;
        setIdentity(resolved);
        setError(null);
        setStatus("ready");
      })
      .catch((failure: unknown) => {
        if (cancelled) return;
        setIdentity(null);
        if (isMissingCredential(failure)) {
          setError(null);
          setStatus("anonymous");
          return;
        }
        setError("Không kiểm tra được quyền truy cập. Kiểm tra kết nối tới máy chủ.");
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [attempt]);

  const signIn = useCallback(
    async (actorId: string, password: string) => {
      const trimmed = actorId.trim();
      if (!trimmed || !password) {
        setError("Nhập tài khoản và mật khẩu.");
        return;
      }
      setError(null);
      try {
        await loginAdmin(trimmed, password);
      } catch (failure: unknown) {
        setIdentity(null);
        setError(signInMessage(failure));
        setStatus("anonymous");
        return;
      }
      // The cookie is now set. Ask the server who it is, the same way a reload does.
      try {
        const resolved = await fetchAdminIdentity(null);
        setIdentity(resolved);
        setStatus("ready");
      } catch (failure: unknown) {
        setIdentity(null);
        if (isMissingCredential(failure)) {
          setError(null);
          setStatus("anonymous");
          return;
        }
        setError("Không kiểm tra được quyền truy cập. Kiểm tra kết nối tới máy chủ.");
        setStatus("error");
      }
    },
    [],
  );

  const signOut = useCallback(async () => {
    try {
      await logoutAdmin();
    } finally {
      // Cached admin data must not outlive the session that fetched it.
      queryClient.removeQueries({ queryKey: ["admin"] });
      setIdentity(null);
      setError(null);
      setStatus("anonymous");
    }
  }, [queryClient]);

  const retry = useCallback(() => setAttempt((count) => count + 1), []);

  const can = useCallback(
    (action: string) => Boolean(identity?.actions.includes(action)),
    [identity],
  );

  const value = useMemo<AdminSession>(
    () => ({ status, identity, error, can, signIn, signOut, retry }),
    [status, identity, error, can, signIn, signOut, retry],
  );

  return <AdminSessionContext.Provider value={value}>{children}</AdminSessionContext.Provider>;
}

export function AdminSessionProvider({ children }: { children: ReactNode }) {
  // A nested provider (TeacherLayout above the gate) must not mount a second
  // session: that would fire a duplicate /admin/identity fetch and shadow the
  // outer one. The check lives here, outside AdminSessionState, so the hook
  // body that fetches never runs for the nested case.
  const existing = useContext(AdminSessionContext);
  if (existing !== null) return <>{children}</>;
  return <AdminSessionState>{children}</AdminSessionState>;
}
