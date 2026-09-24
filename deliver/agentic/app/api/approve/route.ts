// Human-approved placement. The agent (an MCP client holding the Robinhood Agentic-account session) is asked to:
//   1) review_option_order(proposal)  -> show pre-trade alerts, re-check limits
//   2) place_option_order(proposal)   -> limit order, day TIF, at the approved limit
//   3) walk the limit $0.01 every 30-60 s up to mid + 25% of spread until 15:58 ET, then cancel (entries)
// Limits are re-checked server-side immediately before step 2; the model cannot override them.
// Liability: Robinhood states the account holder is responsible for every trade the agent places.
import { NextResponse } from "next/server";
import { checkOrder } from "@/lib/limits";
export async function POST(req: Request) {
  const { proposal, killArmed } = await req.json();
  const errs = checkOrder(proposal, killArmed);
  if (errs.length) return NextResponse.json({ placed: false, errs }, { status: 400 });
  // await agent.call("review_option_order", {...}); await agent.call("place_option_order", {...});
  return NextResponse.json({ placed: "delegated-to-agent", proposal });
}
