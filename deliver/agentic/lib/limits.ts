// Hard limits enforced in server code, never delegated to the model.
// Every proposal and every order passes through checkOrder() before any Robinhood MCP call is made.
export const LIMITS = {
  whitelist: ["SPY", "QQQ", "IWM"] as const,
  side: "buy_to_open_call_or_sell_to_close" as const,
  maxPctEquityPerTrade: 0.12,          // one contract may cost up to 1.5 x the 8% target
  maxContracts: (equity: number) => (equity < 3000 ? 1 : equity < 5000 ? 2 : 3),
  maxOpenPositions: (equity: number) => (equity < 3000 ? 1 : equity < 5000 ? 2 : 3),
  hardPremiumCapUsd: 400,              // absolute ceiling regardless of equity; raise deliberately
  windowET: { start: "15:40", end: "15:58" },
  dailyLossStopPct: 0.15,              // no new entries after a 15% equity drop within the day
  hardStopEquity: 1050,                // section 7.5 hard stop
};

export type Proposal = {
  etf: string; action: "BUY_TO_OPEN" | "SELL_TO_CLOSE"; expiry: string; strike: number; type: "call";
  contracts: number; limit: number; askAtProposal: number; equity: number; settledCash: number; openPositions: number;
};

export function nyTime(d = new Date()): string {
  return new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", hour12: false, hour: "2-digit", minute: "2-digit" }).format(d);
}

export function checkOrder(p: Proposal, killArmed: boolean): string[] {
  const errs: string[] = [];
  const t = nyTime();
  if (killArmed && p.action === "BUY_TO_OPEN") errs.push("kill switch armed");
  if (!LIMITS.whitelist.includes(p.etf as any)) errs.push(`symbol ${p.etf} not whitelisted`);
  if (p.type !== "call") errs.push("calls only");
  if (t < LIMITS.windowET.start || t > LIMITS.windowET.end) errs.push(`outside window (${t} ET)`);
  if (p.action === "BUY_TO_OPEN") {
    const cost = p.askAtProposal * 100 * p.contracts;
    if (p.contracts > LIMITS.maxContracts(p.equity)) errs.push("too many contracts");
    if (p.openPositions >= LIMITS.maxOpenPositions(p.equity)) errs.push("no free slot");
    if (cost > LIMITS.maxPctEquityPerTrade * p.equity) errs.push("premium above 12% of equity");
    if (cost > LIMITS.hardPremiumCapUsd) errs.push("premium above hard cap");
    if (cost > p.settledCash) errs.push("not enough settled cash");
    if (p.equity < LIMITS.hardStopEquity) errs.push("equity below hard stop");
  }
  return errs;
}
