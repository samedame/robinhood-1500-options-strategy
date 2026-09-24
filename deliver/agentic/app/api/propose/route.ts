// Vercel cron target (19:45 and 20:45 UTC on weekdays; only the run that lands at 15:45 ET proceeds).
// It computes IBS for SPY/QQQ/IWM, applies the rulebook, and writes PROPOSALS to Supabase. It places nothing.
// Order placement happens only in /api/approve after a human taps Approve, via the agent's Robinhood MCP session.
import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";
import { checkOrder, nyTime, Proposal } from "@/lib/limits";

const sb = createClient(process.env.SUPABASE_URL!, process.env.SUPABASE_SERVICE_KEY!);

async function intraday(sym: string) {
  // Replace with the agent's get_quotes / historicals call in the Agentic account; Yahoo shown as a free fallback.
  const r = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/${sym}?range=5d&interval=1m`);
  const j = await r.json();
  const q = j.chart.result[0];
  const ts: number[] = q.timestamp; const o = q.indicators.quote[0];
  const day = new Date(ts[ts.length - 1] * 1000).toDateString();
  const idx = ts.map((t, i) => [t, i] as const).filter(([t]) => new Date(t * 1000).toDateString() === day).map(([, i]) => i);
  const hi = Math.max(...idx.map(i => o.high[i] ?? -Infinity)); const lo = Math.min(...idx.map(i => o.low[i] ?? Infinity));
  const last = o.close[idx[idx.length - 1]];
  const prevIdx = ts.map((t, i) => [t, i] as const).filter(([t]) => new Date(t * 1000).toDateString() !== day).map(([, i]) => i);
  const prevDay = new Date(ts[prevIdx[prevIdx.length - 1]] * 1000).toDateString();
  const priorHigh = Math.max(...prevIdx.filter(i => new Date(ts[i] * 1000).toDateString() === prevDay).map(i => o.high[i] ?? -Infinity));
  return { sym, last, hi, lo, ibs: hi > lo ? (last - lo) / (hi - lo) : 0.5, priorHigh };
}

export async function GET() {
  const t = nyTime();
  if (t < "15:40" || t > "15:50") return NextResponse.json({ skipped: `not the ET window (${t})` });
  const { data: ks } = await sb.from("kill_switch").select("armed").eq("id", 1).single();
  const states = await Promise.all(["SPY", "QQQ", "IWM"].map(intraday));
  for (const s of states) {
    await sb.from("signals").insert({ etf: s.sym, last_price: s.last, day_high: s.hi, day_low: s.lo, ibs: s.ibs, prior_high: s.priorHigh, stage: "live" });
  }
  // Exits and entries are assembled exactly as in ibs_dip_calls_signal.py (delta ladder 0.50/0.40/0.30, >=10 DTE,
  // 8% target, 12% cap, spread <= 4%, OI >= 500). The chain lookup uses the agent's get_option_chains tool.
  const proposals: Proposal[] = []; // filled by the ladder logic (omitted here for brevity; see the Python reference)
  const vetted = proposals.map(p => ({ ...p, errors: checkOrder(p, !!ks?.armed) }));
  await sb.from("proposals").insert(vetted.map(v => ({ payload: v, status: v.errors.length ? "blocked" : "pending" })));
  // Notify the human (push, Discord, or email) with an Approve link for each unblocked proposal.
  return NextResponse.json({ states, proposals: vetted });
}
