import { adminConfig, NS24_API_KEY, NS24_API_KEY_NOTE } from "@/constants/adminConfig";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const ConfigEditor = () => {
  const rows = [
    ...adminConfig,
    {
      key: "NS24_API_KEY",
      label: "ns24 API Key",
      value: NS24_API_KEY,
      source: "Loaded from /srv/atlas-lab/shared/atlas.env on the server",
      howToChange: "Set NS24_API_KEY in /srv/atlas-lab/shared/atlas.env. Never expose the full key.",
    },
  ];

  return (
    <div className="mt-1">
      <div className="space-y-1 mb-4">
        <h2 className="text-sm font-medium text-foreground dark:text-zinc-100">Config</h2>
        <p className="text-xs text-muted-foreground dark:text-zinc-400">
          Mirrored from the backend config (rag2025/config). These are the values
          the server runs with; the table is a reference, not a live read.
        </p>
      </div>
      <div className="border border-border dark:border-zinc-800 rounded-sm">
        <Table>
          <TableHeader>
            <TableRow className="border-border dark:border-zinc-800 hover:bg-muted/50 dark:hover:bg-zinc-900/50">
              <TableHead className="text-muted-foreground dark:text-zinc-400 h-8 px-3 text-xs">
                Setting
              </TableHead>
              <TableHead className="text-muted-foreground dark:text-zinc-400 h-8 px-3 text-xs">
                Value
              </TableHead>
              <TableHead className="text-muted-foreground dark:text-zinc-400 h-8 px-3 text-xs">
                Source
              </TableHead>
              <TableHead className="text-muted-foreground dark:text-zinc-400 h-8 px-3 text-xs">
                How to change
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.key} className="border-border dark:border-zinc-800">
                <TableCell className="px-3 py-1.5 text-xs text-foreground dark:text-zinc-100 font-mono align-top">
                  {row.key}
                </TableCell>
                <TableCell
                  className={`px-3 py-1.5 text-xs font-mono align-top ${
                    row.value === "[REDACTED]"
                      ? "text-amber-700 dark:text-amber-400"
                      : "text-foreground dark:text-zinc-300"
                  }`}
                >
                  <span>{row.value}</span>
                  {row.key === "NS24_API_KEY" && (
                    <span className="ml-2 text-muted-foreground dark:text-zinc-400 font-sans normal-case text-[10px]">
                      — {NS24_API_KEY_NOTE}
                    </span>
                  )}
                </TableCell>
                <TableCell className="px-3 py-1.5 text-xs text-muted-foreground dark:text-zinc-400 font-sans align-top">
                  {row.source}
                </TableCell>
                <TableCell className="px-3 py-1.5 text-xs text-muted-foreground dark:text-zinc-400 font-sans align-top">
                  {row.howToChange}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
};

export default ConfigEditor;
