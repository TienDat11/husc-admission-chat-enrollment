import { lazy, Suspense } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Outlet } from "react-router-dom";
import { ThemeProvider } from "@/hooks/useTheme";
import { AdminSessionProvider } from "@/hooks/useAdminSession";
import { StudentProfileProvider } from "@/hooks/useStudentProfile";

const Index = lazy(() => import("./pages/Index"));
const NotFound = lazy(() => import("./pages/NotFound"));
const RecordsGate = lazy(() => import("./components/admin/RecordsGate"));
const TeacherLayout = lazy(() => import("./pages/teacher/TeacherLayout"));
const Overview = lazy(() => import("./pages/teacher/Overview"));
const AddInfo = lazy(() => import("./pages/teacher/AddInfo"));
const QAForm = lazy(() => import("./pages/teacher/QAForm"));
const PendingReview = lazy(() => import("./pages/teacher/PendingReview"));
const Vault = lazy(() => import("./pages/teacher/Vault"));
const Unanswered = lazy(() => import("./pages/teacher/Unanswered"));
const AskTest = lazy(() => import("./pages/teacher/AskTest"));
const History = lazy(() => import("./pages/teacher/History"));
const Accounts = lazy(() => import("./pages/teacher/Accounts"));
const LoginGate = lazy(() => import("./pages/teacher/LoginGate"));
const Ops = lazy(() => import("./pages/teacher/Ops"));
const TableFix = lazy(() => import("./pages/teacher/TableFix"));
const Feedback = lazy(() => import("./pages/teacher/Feedback"));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30_000, refetchOnWindowFocus: false, retry: 1 },
  },
});

function RouteLoadingSkeleton() {
  return (
    <div role="status" aria-live="polite" className="flex items-center justify-center h-screen">
      <div className="animate-pulse text-muted-foreground">Đang tải...</div>
    </div>
  );
}

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <StudentProfileProvider>
        <BrowserRouter>
          <Suspense fallback={<RouteLoadingSkeleton />}>
            <Routes>
            <Route path="/" element={<Index />} />
            <Route
              element={
                <AdminSessionProvider>
                  <Outlet />
                </AdminSessionProvider>
              }
            >
            {/* Ticket T09: 7 teacher-facing screens (05-ux-spec), all behind
                the same credential gate as the records screens. */}
            <Route path="/quan-ly/dang-nhap" element={<LoginGate />} />
            <Route path="/quan-ly" element={<TeacherLayout />}>
              {/* The gate is a pathless layout route: an *index* route may
                  not have children, so `/quan-ly` itself resolves through
                  this wrapper rather than nesting inside an index route. */}
              <Route element={<RecordsGate />}>
                <Route index element={<Overview />} />
              </Route>
              <Route path="them" element={<RecordsGate />}>
                <Route index element={<AddInfo />} />
                <Route path="ngoai-le" element={<TableFix />} />
                <Route path="ngoai-le/:documentId" element={<TableFix />} />
                <Route path="qa" element={<QAForm />} />
              </Route>
              <Route path="cho-duyet" element={<RecordsGate />}>
                <Route index element={<PendingReview />} />
              </Route>
              <Route path="kho" element={<RecordsGate />}>
                <Route index element={<Vault />} />
              </Route>
              <Route path="cau-hoi" element={<RecordsGate />}>
                <Route index element={<Unanswered />} />
              </Route>
              <Route path="phan-hoi" element={<RecordsGate />}>
                <Route index element={<Feedback />} />
              </Route>
              <Route path="hoi-thu" element={<RecordsGate />}>
                <Route index element={<AskTest />} />
              </Route>
              <Route path="lich-su" element={<RecordsGate />}>
                <Route index element={<History />} />
              </Route>
              <Route path="van-hanh" element={<RecordsGate />}>
                <Route index element={<Ops />} />
              </Route>
              <Route path="tai-khoan" element={<RecordsGate />}>
                <Route index element={<Accounts />} />
              </Route>
            </Route>
            </Route>
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
        </StudentProfileProvider>
      </TooltipProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;
