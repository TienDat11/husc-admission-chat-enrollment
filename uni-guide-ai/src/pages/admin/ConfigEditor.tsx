import { adminConfig, RAMCLOUDS_API_KEY, RAMCLOUDS_API_KEY_NOTE } from "@/constants/adminConfig";
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
      key: "RAMCLOUDS_API_KEY",
      label: "RAM Clouds API Key",
      value: RAMCLOUDS_API_KEY,
      source: "Loaded from .env (rag2025/.env)",
      howToChange: "Set RAMCLOUDS_API_KEY in rag2025/.env. Never expose the full key.",
    },
  ];

  return (
    <div className="mt-1">
      <div className="space-y-1 mb-4">
        <h2 className="text-sm font-medium text-zinc-100">Config</h2>
        <p className="text-xs text-zinc-400">
          Active configuration. Values are loaded at build time from
          environment variables.
        </p>
      </div>
      <div className="border border-zinc-800 rounded-sm">
        <Table>
          <TableHeader>
            <TableRow className="border-zinc-800 hover:bg-zinc-900/50">
              <TableHead className="text-zinc-400 h-8 px-3 text-xs">
                Setting
              </TableHead>
              <TableHead className="text-zinc-400 h-8 px-3 text-xs">
                Value
              </TableHead>
              <TableHead className="text-zinc-400 h-8 px-3 text-xs">
                Source
              </TableHead>
              <TableHead className="text-zinc-400 h-8 px-3 text-xs">
                How to change
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.key} className="border-zinc-800">
                <TableCell className="px-3 py-1.5 text-xs text-zinc-100 font-mono align-top">
                  {row.key}
                </TableCell>
                <TableCell
                  className={`px-3 py-1.5 text-xs font-mono align-top ${
                    row.value === "[REDACTED]"
                      ? "text-amber-400"
                      : "text-zinc-300"
                  }`}
                >
                  <span>{row.value}</span>
                  {row.key === "RAMCLOUDS_API_KEY" && (
                    <span className="ml-2 text-zinc-500 font-sans normal-case text-[10px]">
                      — {RAMCLOUDS_API_KEY_NOTE}
                    </span>
                  )}
                </TableCell>
                <TableCell className="px-3 py-1.5 text-xs text-zinc-500 font-sans align-top">
                  {row.source}
                </TableCell>
                <TableCell className="px-3 py-1.5 text-xs text-zinc-400 font-sans align-top">
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
