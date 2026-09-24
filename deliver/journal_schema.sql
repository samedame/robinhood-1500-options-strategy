-- IBS Dip Calls: trade journal + guardrail tables (Supabase / Postgres)
-- One row per signal check, one row per trade, a daily equity mark, and a single kill-switch row.

create table if not exists signals (
  id              bigserial primary key,
  run_at          timestamptz not null default now(),       -- when the 3:45 pm ET check ran
  etf             text not null check (etf in ('SPY','QQQ','IWM')),
  last_price      numeric(10,2) not null,
  day_high        numeric(10,2) not null,
  day_low         numeric(10,2) not null,
  ibs             numeric(6,4) not null,                    -- (last - low) / (high - low)
  prior_high      numeric(10,2) not null,
  vix             numeric(6,2),
  close_ibs       numeric(6,4),                             -- filled after the close, for 3:45-vs-close tracking
  is_signal       boolean generated always as (ibs < 0.10) stored,
  stage           text not null default 'paper' check (stage in ('paper','live'))
);

create table if not exists trades (
  id                 bigserial primary key,
  stage              text not null check (stage in ('paper','live')),
  etf                text not null check (etf in ('SPY','QQQ','IWM')),
  signal_id          bigint references signals(id),
  entry_at           timestamptz not null,
  expiry             date not null,
  strike             numeric(10,2) not null,
  rung_delta         numeric(4,2) not null check (rung_delta in (0.50,0.40,0.30)),
  delta_at_entry     numeric(5,3),
  entry_iv           numeric(6,4),
  entry_bid          numeric(8,2) not null,
  entry_ask          numeric(8,2) not null,
  entry_limits_tried numeric(8,2)[],                         -- every limit price used while walking
  entry_fill         numeric(8,2) not null,                  -- paper: mid + 25% of spread
  contracts          int not null check (contracts >= 1),
  equity_at_entry    numeric(12,2) not null,
  settled_at_entry   numeric(12,2) not null,
  premium_paid       numeric(12,2) generated always as (entry_fill * 100 * contracts) stored,
  pct_equity         numeric(6,4) generated always as (entry_fill * 100 * contracts / nullif(equity_at_entry,0)) stored,
  und_entry          numeric(10,2) not null,
  exit_at            timestamptz,
  exit_reason        text check (exit_reason in ('bounce','time','kill','manual')),
  exit_bid           numeric(8,2),
  exit_ask           numeric(8,2),
  exit_fill          numeric(8,2),
  fees               numeric(8,2) default 0,
  und_exit           numeric(10,2),
  pnl                numeric(12,2),
  ret_on_premium     numeric(8,4),
  und_ret            numeric(8,5),
  slippage_vs_mid    numeric(8,4),                           -- (fill - mid)/spread, entry and exit averaged
  rule_violation     boolean default false,
  notes              text
);

create table if not exists equity_daily (
  d          date primary key,
  equity     numeric(12,2) not null,
  settled    numeric(12,2) not null,
  peak       numeric(12,2) not null,
  drawdown   numeric(6,4) generated always as (equity / nullif(peak,0) - 1) stored
);

create table if not exists kill_switch (
  id          int primary key default 1 check (id = 1),
  armed       boolean not null default false,                -- true = no new entries
  reason      text,
  set_at      timestamptz default now()
);
insert into kill_switch (id, armed) values (1, false) on conflict do nothing;

-- Rolling monitors used by the kill-switch rules (section 7.5 of the book)
create or replace view live_monitor as
select
  count(*)                                             as live_trades,
  avg(ret_on_premium)                                  as mean_ret,
  avg(ret_on_premium) filter (where rn <= 20)          as mean_ret_last20,
  avg(und_ret) filter (where rn <= 30)                 as mean_und_ret_last30,
  avg(slippage_vs_mid) filter (where rn <= 10)         as mean_slippage_last10
from (
  select *, row_number() over (order by exit_at desc) as rn
  from trades where stage = 'live' and exit_at is not null
) t;

-- Agent proposals awaiting human approval (Agentic MCP path, section 8 of the book)
create table if not exists proposals (
  id          bigserial primary key,
  created_at  timestamptz not null default now(),
  payload     jsonb not null,                          -- the Proposal object incl. limit-check errors
  status      text not null check (status in ('pending','blocked','approved','rejected','placed','expired')),
  decided_at  timestamptz,
  order_id    text                                     -- Robinhood order id once placed
);
