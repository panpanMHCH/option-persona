# Option-chain CSV import

Implementation: `src/option_persona/data/{chain_csv,contract_specs,examples}.py`
Tests: `tests/test_chain_csv.py`

## Purpose

This is Phase 2's first building block. It turns a chain snapshot exported from a broker or data vendor into validated `OptionQuote` records. Each quote carries its provenance: source, timezone-aware observation time, and whether it is synthetic and whether it is delayed.

Everything downstream (executable parity, verticals, box spreads) is only as honest as this step. That is why the importer prefers rejecting a row to guessing.

## What the importer will never do

| Situation | Behaviour | Why |
|---|---|---|
| Blank bid | Rejects the row (`missing_value`) | "No bid shown" and "bid of 0" are different facts. Filling in zero would invent a market. |
| Blank or zero ask | Rejects the row (`missing_value` / `no_offer`) | With no offer, there is no executable buy price. |
| No timestamp | Refuses the file, unless the caller supplies a snapshot time | Rule 8: timestamps are never inferred, and "now" is not the observation time. |
| Timestamp without a time-zone offset | Rejects the row, unless the caller declares `assume_timezone` | 15:59 in New York and 15:59 in Bangkok are eleven or twelve hours apart. |
| Delay status unknown | Refuses the file until `is_delayed` is declared | A 15-minute-old quote is not a current market. Only the user knows the feed type. |
| Vendor IV, Greeks, "mark", "theo" | Ignores them, with an info note | These are someone else's model under undisclosed assumptions, not observations. |
| Last-trade price | Ignores it, with an info note | It can be hours old and is not executable. |
| Synthetic rows | Keeps `is_synthetic` sticky | A per-row `false` cannot clear a file-level or source-name flag (`synthetic…`). |

## Schema

There is one row per option quote. Header names are case-insensitive, and spaces and hyphens become underscores. `column_map` renames anything else, for example `{"Strike Price": "strike"}`.

- **Required:** `underlying`, `option_type` (call/put or C/P), `strike`, `expiry` (ISO date), `bid`, `ask`.
- **Underlying reference:** `underlying_price`, or both `underlying_bid` and `underlying_ask`. If only the bid/ask is present, the reference is their mid, and an info issue records that it was derived.
- **Observation time:** `observed_at` (ISO 8601 with an offset), or a file-level `observed_at=` argument.
- **Optional:** `bid_size`, `ask_size`, `exercise_style`, `settlement`, `multiplier`, `currency`, `is_delayed`, `is_synthetic`.

**Number parsing:** `$` and thousands separators are stripped. Decimal commas are not supported.

**Duplicates:** one file is one snapshot. A repeated contract (underlying, type, strike, expiry, multiplier) keeps its first occurrence and rejects the rest, with a pointer to the kept line.

## Contract terms: explicit, registry, or stated assumption

Exercise style and settlement decide which identities are valid. Put-call parity as an equality needs European exercise, and `assess_parity` refuses American contracts.

For each row the importer resolves these terms in order:

1. A column in the row.
2. An importer argument (`exercise_style=`, `settlement=`, `multiplier=`).
3. The registry of well-known index options: SPX, SPXW, XSP, NDX and RUT are European and cash-settled, multiplier 100.
4. The US single-stock/ETF convention: American, physical, 100.

Steps 3 and 4 are written to `ChainImport.assumptions`, which the dashboard displays. Assumptions are recorded only for **accepted** rows.

**What the registry does not know:**

- **AM versus PM settlement timing.** Standard monthly SPX settles on the opening print; SPXW is PM-settled. The expiry instant for AM-settled analysis must be supplied separately.
- **Adjusted contracts.** After a corporate action, deliverables become non-standard, and nothing in a typical quote export reveals this.

## Issue severities

- **error:** the row is rejected. Codes include `missing_value`, `unparseable`, `invalid_value`, `crossed_quote`, `no_offer`, `naive_timestamp`, `missing_timestamp`, `missing_delay_flag`, `expired` and `duplicate_contract`.
- **warning:** the row is kept but needs your attention:
  - `zero_bid`: the mid is not a meaningful value.
  - `locked_market`.
  - `post_expiry_observation`: observed at or after 16:00 ET on expiry day.
  - `underlying_inconsistent`: the reference price lies outside the underlying's own quote.
  - `one_sided_underlying`.
- **info:** how the file was interpreted, e.g. `ignored_column` and `derived_underlying`.

Line numbers assume one header line (header = line 1), so they match what a spreadsheet shows.

## Usage

```python
from option_persona.data import read_option_chain_csv

result = read_option_chain_csv(
    "data/raw/xsp_2026-11-05.csv",
    source="Cboe delayed quotes download",
    is_delayed=True,
    column_map={"Strike Price": "strike"},
)
result.rows_accepted, result.rows_rejected
result.assumptions        # what was assumed, in words
result.issues_frame()     # every rejection and warning, with line and column
result.to_frame()         # accepted quotes, provenance alongside prices
```

`data/raw/` is git-ignored, so personal or licensed data stays out of the repository.

The dashboard's **Chain import** tab does the same thing and offers a downloadable **synthetic** example file. In that file every row is `is_synthetic=true`, and the underlying is named `SYNTHETIC`, so it cannot be mistaken for a market snapshot.

## Validation

`tests/test_chain_csv.py` has 48 tests:

- Each rejection code is exercised, with the exact line and column asserted.
- Each refusal at file level is tested.
- Specification precedence is tested: column, then argument, then registry, then the stated default.
- Naive timestamps are tested with a declared Bangkok zone, including conversion to the correct New York instant.
- Duplicates are tested, and so is the rule that rejected rows contribute no assumptions.
- The synthetic exporter round-trips exactly through CSV.
- A hypothesis property test checks that any well-formed row is accepted with its prices unchanged.

## Limitations and next steps

- **Single-snapshot semantics.** There is no time series of snapshots, and no storage yet. DuckDB/Parquet is the next roadmap item.
- **Quote quality is only partly assessed.** Freshness relative to the underlying, leg synchronisation across call/put/underlying, wide-spread and displayed-size screens are the subject of the freshness item.
- **No broker-specific presets.** Export formats differ and change, and none has been verified against a real file here. Use `column_map` until a format has been checked against a genuine export.
