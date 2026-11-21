// Roblox Up-and-Coming chart forecaster — roblox-up-and-coming-forecast-01.
//
// Model: rank-diffusion Monte Carlo. The chart's ranks wander — a random walk
// whose step grows with rank (the top 10 is sticky, the tail churns) — games
// leave the chart (fading, or ageing past the 28-day publish window) and new
// games enter at the bottom. The forecast is the time-averaged marginal: for
// each place, the probability a game holds it, averaged over every hour of the
// scored window. That is exactly what the log score rewards, so the
// probabilities are honest and spread — never p=1 on one order. A missed place
// at one hour costs ln(0.0001) = -9.21, so mass is spread over every plausible
// game per place, and the unnamed share is left for games the simulation says
// will enter from outside the named set.
//
// Everything is fit from the history in /input at run time (no hardcoded
// dates, reads, or window): the per-hour rank volatility by rank band and the
// chart exit rate come from consecutive reads, the window length from
// task.json, and a game's 28-day ageing exit from its details.created when
// the history carries it. So it works on the longer, denser final history as
// well as the practice one.
//
// Deterministic: one fixed seed drives every random draw; two runs on the same
// input are byte-identical. Node 22, standard library only, no network.

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Overall diffusion scale, tuned on the practice window. */
const SIGMA_SCALE = 1.2;
/** New games are more volatile in their first day (they surge or fade fast). */
const SURGE_FACTOR = 1.5;
const SURGE_HOURS = 24;
/** Monte Carlo paths; 2000 keeps the sampling noise near ±0.05 nats. */
const N_PATHS = 2000;
/** Fixed seed: the run is deterministic. */
const SEED = 20261112;
/** A game is on the Up-and-Coming chart only while published within 28 days. */
const MAX_AGE_MS = 28 * 24 * 60 * 60 * 1000;
/** Rows below this probability are dropped (they cost bytes, not score). */
const MIN_P = 0.0005;
/** Fallbacks when the history is too short to fit (defaults from the practice data). */
const FALLBACK_SIGMA = [0.88, 2.08, 3.18, 3.74];
const FALLBACK_EXIT = 0.008;

/** mulberry32: a small deterministic PRNG. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal draws from the seeded PRNG (Box–Muller, with a spare). */
function makeGauss(rand: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    do {
      u = rand();
    } while (u <= 0);
    const v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    const t = 2 * Math.PI * v;
    spare = r * Math.sin(t);
    return r * Math.cos(t);
  };
}

interface ChartRow {
  rank: number;
  universe_id: number;
}

interface GameDetails {
  created?: string;
}

interface Snapshot {
  at: string;
  up_and_coming: ChartRow[];
  details?: Record<string, GameDetails | null> | null;
}

interface Task {
  places: number;
  output: string;
  window?: { slots?: number; slot_minutes?: number };
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as Task;
const PLACES = task.places ?? 10;
const HOURS = task.window?.slots ?? 72;
const OUTPUT = task.output ?? 'forecast.json';

const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '');
const snaps = lines.map((l) => JSON.parse(l) as Snapshot);
const times = snaps.map((s) => Date.parse(s.at));
const historyEnd = times[times.length - 1]!;

/** Rank bands for the volatility fit: [lo, hi] inclusive, hi = Infinity for the tail. */
const BANDS: Array<[number, number]> = [
  [1, 10],
  [11, 30],
  [31, 60],
  [61, Infinity],
];

/** Per-hour rank-change variance by band, fit from consecutive reads (var ∝ dt). */
function fitSigma(): number[] {
  const sums = BANDS.map(() => 0);
  const counts = BANDS.map(() => 0);
  for (let i = 0; i + 1 < snaps.length; i++) {
    const dt = (times[i + 1]! - times[i]!) / 3600000;
    if (!(dt > 0)) continue;
    const cur = new Map<number, number>();
    for (const r of snaps[i]!.up_and_coming) cur.set(r.universe_id, r.rank);
    for (const r of snaps[i + 1]!.up_and_coming) {
      const prev = cur.get(r.universe_id);
      if (prev === undefined) continue;
      const band = BANDS.findIndex(([lo, hi]) => prev >= lo && prev <= hi);
      if (band < 0) continue;
      const dr = r.rank - prev;
      sums[band]! += (dr * dr) / dt;
      counts[band]! += 1;
    }
  }
  return BANDS.map((_, b) =>
    counts[b]! > 0 ? Math.sqrt(sums[b]! / counts[b]!) : FALLBACK_SIGMA[b]!,
  );
}

/** Chart exit rate per game per hour, fit from consecutive reads. */
function fitExitRate(): number {
  let exits = 0;
  let gameHours = 0;
  for (let i = 0; i + 1 < snaps.length; i++) {
    const dt = (times[i + 1]! - times[i]!) / 3600000;
    if (!(dt > 0)) continue;
    const prev = new Set(snaps[i]!.up_and_coming.map((r) => r.universe_id));
    const cur = new Set(snaps[i + 1]!.up_and_coming.map((r) => r.universe_id));
    for (const u of prev) if (!cur.has(u)) exits += 1;
    gameHours += prev.size * dt;
  }
  return gameHours > 0 ? exits / gameHours : FALLBACK_EXIT;
}

/**
 * The ranks at which games first appear on the chart, fit from the history.
 * New games do not all start at the bottom: they debut across the chart
 * (mean ~65, median ~73, but a real tail reaches the top 10), and a debutant
 * can surge straight to #1. Sampling an entry rank from this list reproduces
 * that. Falls back to below the chart when the history is too short.
 */
function fitEntryRanks(): number[] {
  const seen = new Set<number>();
  const ranks: number[] = [];
  for (const s of snaps) {
    for (const r of s.up_and_coming) {
      if (!seen.has(r.universe_id)) {
        seen.add(r.universe_id);
        ranks.push(r.rank);
      }
    }
  }
  return ranks;
}

/** The last chart, best rank first. */
const lastRows = [...snaps[snaps.length - 1]!.up_and_coming].sort((a, b) => a.rank - b.rank);
const chartSize = lastRows.length;

/**
 * The hour (1-based, from the history end) at which each game ages past 28 days
 * and must leave the chart, from the newest details that name it. Infinity when
 * its creation date is unknown or it stays young through the window.
 */
function fitAgeExit(): Map<number, number> {
  const created = new Map<number, number>();
  for (const s of snaps) {
    const d = s.details;
    if (!d || typeof d !== 'object') continue;
    for (const [key, det] of Object.entries(d)) {
      if (det && typeof det.created === 'string') {
        const t = Date.parse(det.created);
        if (Number.isFinite(t)) created.set(Number(key), t);
      }
    }
  }
  const out = new Map<number, number>();
  for (const [id, t] of created) {
    const exitMs = t + MAX_AGE_MS;
    const h = Math.ceil((exitMs - historyEnd) / 3600000);
    if (h >= 1 && h <= HOURS) out.set(id, h);
    else if (h < 1) out.set(id, 0);
  }
  return out;
}

const sigma = fitSigma();
const exitRate = fitExitRate();
const ageExit = fitAgeExit();
const entryRanks = fitEntryRanks();

/** Draw a debut rank for a newcomer from the fitted entry distribution. */
function sampleEntryRank(): number {
  if (entryRanks.length === 0) return chartSize + 20;
  return entryRanks[Math.floor(rand() * entryRanks.length)]!;
}

function sigmaFor(rank: number): number {
  const band = BANDS.findIndex(([lo, hi]) => rank >= lo && rank <= hi);
  return (band < 0 ? FALLBACK_SIGMA[FALLBACK_SIGMA.length - 1]! : sigma[band]!) * SIGMA_SCALE;
}

const rand = mulberry32(SEED);
const gauss = makeGauss(rand);

/** counts[place] maps universe_id -> samples holding that place. Placeholders (id 0) are never counted. */
const counts: Array<Map<number, number>> = Array.from({ length: PLACES }, () => new Map());
const totalSamples = N_PATHS * HOURS;

for (let path = 0; path < N_PATHS; path++) {
  // Latent scores: score = -rank, so sorting by score descending is the chart order.
  // `entered` is 0 for games on the chart at the start, else the hour a newcomer debuted.
  const state: Array<{ id: number; score: number; entered: number }> = lastRows.map((r, i) => ({
    id: r.universe_id,
    score: -(i + 1),
    entered: 0,
  }));
  for (let hour = 1; hour <= HOURS; hour++) {
    state.sort((a, b) => b.score - a.score);
    for (let i = 0; i < state.length; i++) {
      const g = state[i]!;
      // A game that aged past 28 days leaves now (replaced below by a newcomer).
      if ((ageExit.get(g.id) ?? Infinity) <= hour) continue;
      // Newcomers surge in their first day: extra volatility on top of the rank band's.
      const surge = g.id === 0 && hour - g.entered <= SURGE_HOURS ? SURGE_FACTOR : 1;
      g.score += gauss() * sigmaFor(i + 1) * surge;
    }
    // Exits: a game leaves with a rate that grows with rank (the tail churns
    // hardest); it is replaced by an unnamed newcomer debuting at a rank drawn
    // from the fitted entry distribution. Games that aged out are removed outright.
    const next: Array<{ id: number; score: number; entered: number }> = [];
    for (let i = 0; i < state.length; i++) {
      const g = state[i]!;
      if ((ageExit.get(g.id) ?? Infinity) <= hour) {
        next.push({ id: 0, score: -sampleEntryRank(), entered: hour });
        continue;
      }
      const rank = i + 1;
      const pExit = exitRate * (0.2 + (rank / Math.max(chartSize, 1)) * 3);
      if (rand() < pExit) next.push({ id: 0, score: -sampleEntryRank(), entered: hour });
      else next.push(g);
    }
    next.sort((a, b) => b.score - a.score);
    state.length = 0;
    state.push(...next);
    for (let k = 0; k < PLACES; k++) {
      const id = state[k]!.id;
      if (id !== 0) {
        const m = counts[k]!;
        m.set(id, (m.get(id) ?? 0) + 1);
      }
    }
  }
}

/** Probabilities: the time-averaged marginal, floored to 6 decimals (so no place sums over 1). */
const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];
for (let k = 0; k < PLACES; k++) {
  for (const [id, c] of counts[k]!) {
    const p = Math.floor((c / totalSamples) * 1e6) / 1e6;
    if (p > MIN_P) probabilities.push({ universe_id: id, place: k + 1, p });
  }
}
probabilities.sort((a, b) => a.place - b.place || b.p - a.p || a.universe_id - b.universe_id);

/** top10: the ten games with the lowest expected rank over the window (rank 11 when outside the top 10). */
const expectedRank = new Map<number, number>();
for (let k = 0; k < PLACES; k++) {
  for (const [id, c] of counts[k]!) {
    expectedRank.set(id, (expectedRank.get(id) ?? 0) + ((k + 1) * c) / totalSamples);
  }
}
const inTop = new Map<number, number>();
for (let k = 0; k < PLACES; k++) {
  for (const [id, c] of counts[k]!) inTop.set(id, (inTop.get(id) ?? 0) + c / totalSamples);
}
for (const [id, pTop] of inTop) {
  expectedRank.set(id, (expectedRank.get(id) ?? 0) + 11 * (1 - pTop));
}
const top10 = [...inTop.keys()]
  .sort((a, b) => (expectedRank.get(a) ?? 99) - (expectedRank.get(b) ?? 99))
  .slice(0, PLACES);

writeFileSync(join(OUTPUT_DIR, OUTPUT), `${JSON.stringify({ probabilities, top10 })}\n`);
