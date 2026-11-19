// @spec(PL-7.2 Frame 12 — anonymous student profile)
// useStudentProfile — session UUID v4 + anonymous student profile state.
//
// The session id is a client-generated UUID v4 persisted in localStorage so it
// survives tab switches and page reloads without resetting. The profile itself
// (raw_score, method, to_hop, uu_tien_kv/dt, target_majors, ...) is the
// anonymous (no name/email/phone/cccd/ip) record stored server-side under
// student_profiles; the FE mirrors it here so the recommender/chat can read
// it without re-fetching on every render.
//
// Never logs the session id together with query text — that rule is enforced
// in lib/api.ts, which only attaches X-Student-Session after setStudentSessionId
// is called.
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react';
import { setStudentSessionId } from '@/lib/api';

const SESSION_ID_KEY = 'husc:student_session_id';
const PROFILE_KEY = 'husc:student_profile';

export interface StudentProfile {
  student_session_id: string;
  raw_score: number | null;
  method: string | null;
  to_hop: string | null;
  uu_tien_kv: string | null;
  uu_tien_dt: string | null;
  target_majors: string[];
  admission_year: number | null;
  created_at: string | null;
  updated_at: string | null;
  expires_at: string | null;
}

interface StudentProfileContextValue {
  sessionId: string;
  profile: StudentProfile | null;
  setProfile: (profile: StudentProfile | null) => void;
  clearProfile: () => void;
  hasProfile: boolean;
}

const StudentProfileContext =
  createContext<StudentProfileContextValue | null>(null);

// Persist the session id across tab switches / reloads without resetting.
function _persistedSessionId(): string {
  if (typeof window === 'undefined') return crypto.randomUUID();
  const existing = localStorage.getItem(SESSION_ID_KEY);
  if (existing) return existing;
  const id = crypto.randomUUID();
  try {
    localStorage.setItem(SESSION_ID_KEY, id);
  } catch {
    // storage disabled — session lives in memory only
  }
  return id;
}

function _loadProfile(): StudentProfile | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = localStorage.getItem(PROFILE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as StudentProfile;
  } catch {
    return null;
  }
}

export function StudentProfileProvider({
  children,
}: {
  children: ReactNode;
}): JSX.Element {
  const [sessionId] = useState<string>(_persistedSessionId);
  const [profile, setProfileState] = useState<StudentProfile | null>(() =>
    _loadProfile(),
  );

  // Hydrate profile from localStorage on mount so a reload doesn't reset it.
  useEffect(() => {
    setProfileState(_loadProfile());
  }, []);

  const setProfile = useCallback(
    (next: StudentProfile | null) => {
      setProfileState(next);
      try {
        if (next) {
          localStorage.setItem(PROFILE_KEY, JSON.stringify(next));
        } else {
          localStorage.removeItem(PROFILE_KEY);
        }
      } catch {
        // storage disabled
      }
      // Attach / detach the header in lib/api.ts based on profile presence.
      setStudentSessionId(next ? sessionId : null);
    },
    [sessionId],
  );

  const clearProfile = useCallback(() => setProfile(null), []);

  // Ensure lib/api.ts knows the current session id even if the profile was
  // hydrated from storage before the provider rendered.
  useEffect(() => {
    setStudentSessionId(profile ? sessionId : null);
  }, [sessionId, profile]);

  return (
    <StudentProfileContext.Provider
      value={{
        sessionId,
        profile,
        setProfile,
        clearProfile,
        hasProfile: profile !== null,
      }}
    >
      {children}
    </StudentProfileContext.Provider>
  );
}

export function useStudentProfile(): StudentProfileContextValue {
  const ctx = useContext(StudentProfileContext);
  if (!ctx) {
    throw new Error(
      'useStudentProfile must be used within a StudentProfileProvider',
    );
  }
  return ctx;
}
