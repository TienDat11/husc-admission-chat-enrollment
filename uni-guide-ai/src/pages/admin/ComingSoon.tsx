interface ComingSoonProps {
  title?: string;
  description?: string;
}

const ComingSoon = ({ title = "Coming Soon", description = "This section is under development. Stay tuned." }: ComingSoonProps) => {
  return (
    <div className="flex flex-col items-center justify-center h-full">
      <h2 className="text-lg font-semibold text-foreground dark:text-zinc-100">{title}</h2>
      <p className="mt-2 text-sm text-muted-foreground dark:text-zinc-400">{description}</p>
    </div>
  );
};

export default ComingSoon;
