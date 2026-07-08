"use client";

import { ReactNode } from "react";

export function Card({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-xl border border-border-token bg-surface p-4 shadow-sm">
      <header className="mb-3">
        <h2 className="text-sm font-semibold tracking-wide">{title}</h2>
        {subtitle && <p className="text-xs text-muted mt-0.5">{subtitle}</p>}
      </header>
      {children}
    </section>
  );
}

export function StatTile({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-xl border border-border-token bg-surface px-4 py-3">
      <div className="text-xs text-secondary">{label}</div>
      <div className="text-2xl font-semibold tabular-nums mt-0.5">{value}</div>
      {hint && <div className="text-xs text-muted mt-0.5">{hint}</div>}
    </div>
  );
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "good" | "serious";
}) {
  const tones = {
    neutral: "bg-surface-2 text-secondary",
    good: "bg-surface-2 text-good",
    serious: "bg-surface-2 text-serious",
  } as const;
  return (
    <span
      className={`inline-block rounded-full px-2 py-0.5 text-[11px] font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function Table({
  head,
  rows,
}: {
  head: string[];
  rows: ReactNode[][];
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-muted border-b border-border-token">
            {head.map((h) => (
              <th key={h} className="py-1.5 pr-3 font-medium whitespace-nowrap">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((cells, i) => (
            <tr
              key={i}
              className="border-b border-border-token/60 last:border-0 hover:bg-surface-2/60"
            >
              {cells.map((c, j) => (
                <td key={j} className="py-1.5 pr-3 whitespace-nowrap tabular-nums">
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
