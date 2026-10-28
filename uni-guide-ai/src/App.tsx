import { lazy, Suspense } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate, Outlet } from "react-router-dom";
import { ThemeProvider } from "@/hooks/useTheme";
import { AdminSessionProvider } from "@/hooks/useAdminSession";

const Index = lazy(() => import("./pages/Index"));
const NotFound = lazy(() => import("./pages/NotFound"));
const AdminLayout = lazy(() => import("./pages/admin/AdminLayout"));
const ConfigEditor = lazy(() => import("./pages/admin/ConfigEditor"));
const ComingSoon = lazy(() => import("./pages/admin/ComingSoon"));
const RecordsGate = lazy(() => import("./pages/admin/RecordsGate"));
const RecordList = lazy(() => import("./pages/admin/RecordList"));
const RecordForm = lazy(() => import("./pages/admin/RecordForm"));
const ReleaseStatus = lazy(() => import("./pages/admin/ReleaseStatus"));
const ReviewPreview = lazy(() => import("./pages/admin/ReviewPreview"));
const TeacherLayout = lazy(() => import("./pages/teacher/TeacherLayout"));
const Overview = lazy(() => import("./pages/teacher/Overview"));
const AddInfo = lazy(() => import("./pages/teacher/AddInfo"));
const QAForm = lazy(() => import("./pages/teacher/QAForm"));
const PendingReview = lazy(() => import("./pages/teacher/PendingReview"));
const Vault = lazy(() => import("./pages/teacher/Vault"));
const Unanswered = lazy(() => import("./pages/teacher/Unanswered"));
const AskTest = lazy(() => import("./pages/teacher/AskTest"));
const History = lazy(() => import("./pages/teacher/History"));

const queryClient = new QueryClient();

function RouteLoadingSkeleton() {
  return (
    <div className="flex items-center justify-center h-screen">
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
            <Route path="/admin" element={<AdminLayout />}>
              <Route index element={<Navigate to="config" replace />} />
              <Route path="config" element={<RecordsGate />}>
                <Route index element={<ConfigEditor />} />
              </Route>
              {/* Admissions CRUD (ticket 19). The gate asks for an admin
                  credential before any record is read or written. */}
              <Route path="records" element={<RecordsGate />}>
                <Route index element={<RecordList />} />
                <Route path="new" element={<RecordForm mode="create" />} />
                <Route path=":recordId/edit" element={<RecordForm mode="edit" />} />
                {/* Review diff + preview (ticket 25): the officer sees the
                    change against the published version and the real answer
                    before submitting, and submits from that screen. */}
                <Route
                  path=":recordId/review"
                  element={<ReviewPreview />}
                />
              </Route>
              <Route path="ingest" element={<ComingSoon title="Ingest" />} />
              <Route path="retrieval" element={<ComingSoon title="Retrieval" />} />
              <Route path="debt" element={<ComingSoon title="Debt" />} />
              {/* Release + index-job screen (ticket 26). Inside the gate for the
                  same reason the records screens are: it reads release state
                  and can roll one back, both of which need a credential. */}
              <Route path="releases" element={<RecordsGate />}>
                <Route index element={<ReleaseStatus />} />
              </Route>
            </Route>
            {/* Ticket T09: 7 teacher-facing screens (05-ux-spec), all behind
                the same credential gate as the records screens. */}
            <Route path="/quan-ly" element={<TeacherLayout />}>
              {/* The gate is a pathless layout route: an *index* route may
                  not have children, so `/quan-ly` itself resolves through
                  this wrapper rather than nesting inside an index route. */}
              <Route element={<RecordsGate />}>
                <Route index element={<Overview />} />
              </Route>
              <Route path="them" element={<RecordsGate />}>
                <Route index element={<AddInfo />} />
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
              <Route path="hoi-thu" element={<RecordsGate />}>
                <Route index element={<AskTest />} />
              </Route>
              <Route path="lich-su" element={<RecordsGate />}>
                <Route index element={<History />} />
              </Route>
            </Route>
            </Route>
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
      </TooltipProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;
