# jev-km

A two-role decision system for long-horizon game agents: one model family, two
prompts, one strict contract between them.

Both roles run the same model over `chat/completions`. What separates them is
responsibility, not capability:

- the **executor** answers one question every step — which legal control do I press
  now — and reports its reasoning in declared fields;
- the **planner** answers whether the current stage plan still holds, and publishes
  a revision. It never names a control.

```
                    ┌────────────────────────────────────────┐
                    │  planner                               │
                    │  same model, planner prompt            │
                    └────────────────────────────────────────┘
                       ▲  ExecutorStep            │ PlanDirective
                       │  (action, why,           │ (briefing, stage_goal,
                       │   confidence, expect,    │  tactics, situation,
                       │   belief, stuck,         │  avoid, exclude,
                       │   outcome)               │  review_in)
                       │                          ▼
   observe ──► ┌──────────────────────────────────────────┐ ──► execute ──► observe
               │  executor                                 │
               │  same model, executor prompt              │
               └──────────────────────────────────────────┘

   every exchange is recorded in contract form as a distillation sample
```

## Why this replaced a scorer plus a planner

The previous fast tier was a scoring endpoint. It received a map of action id to
hand-written usage condition and returned one probability per candidate. That forced
three things:

- the candidate list had to be fully enumerated and described every step, because the
  scorer could not name anything it was not given;
- the confidence threshold became the only way to express doubt, because a
  probability table has no other signal;
- the planner had to author a per-action condition map to make the list meaningful,
  which is exactly the burden the split was meant to remove.

Worse, on a hard step the planner was asked for a single action, so the two tiers
collapsed into one at the planner's latency and price. A chat model with a role
prompt needs none of that: it reports its own confidence, states what it expects to
happen, and names one action directly.

## The contract

`general_agent/contract.py` is the only channel between the roles, and field names
are training labels. Lifetimes are declared, not implied:

| Field | Lifetime | Read by |
| --- | --- | --- |
| `briefing` | whole episode | executor |
| `stage_goal` | one stage | executor |
| `tactics` | one stage | executor |
| `avoid` | one stage | executor (checked at execution) |
| `exclude` | whole episode | adapter (prunes the candidate list) |
| `situation` | one review | executor |
| `action` | one step | adapter |

A reply that only refreshes `situation` cannot silently invalidate `stage_goal`. The
old design had to guard against that with an "always publish a complete revision"
rule; here the lifetimes are enforced by the merge in `parse_planner_reply`.

Payloads are rendered from declared fields and serialised by a canonical `encode()`,
so two identical situations produce identical bytes. That is what makes a recorded
prompt a usable training artifact rather than a description of one.

## When the planner is consulted

`ExecutorPolicy.choose` reviews on any of the following, in the order they are
checked:

- `first_step` — no plan exists
- `executor_error` — two consecutive failed reviews
- `executor_stuck` — the executor reported the plan cannot be carried out
- `phase_change` — phase, level or the legal set changed
- `no_change` — the same action three times in a row
- `low_confidence` — the executor's own confidence fell below the threshold
- `plan_age` — the interval the planner asked for, floored by configuration

The last two are worth contrasting with the old gate. `low_confidence` is the
model's number about its own choice, so it is comparable across candidate counts;
the old gate was a softmax mass over a list of varying length. And `plan_age` can be
lengthened by the planner when a plan works, but not shortened below the floor,
which is what stops the slow role from turning into a per-step caller.

## A failed review does not stop the episode

A review is an optimisation on top of a working executor, not a precondition of it.
This was measured: with the review as a precondition, one unparseable reply discarded
a whole 37-step trajectory at step 4, because the first-step trigger always fires.

The current rule keeps executing on the previous plan and counts failures, stopping
at six. A rejected reply also leaves the previous directive in place — an
unparseable answer has said nothing about whether the plan is wrong.

## Fallback is not a game heuristic

When the executor's reply is unusable, the policy takes the first action that is
neither `wait` nor forbidden by the plan. `wait` is excluded because in a real game
it is usually offered during a transition, and taking it on failure turns a decision
problem into an indefinite stall. If nothing qualifies, the episode stops: the
harness has no legal way to act and says so rather than guessing.

There is deliberately no hand-coded game fallback. A right-hand rule works for mazes
and is wrong for shooters, card games and platformers.

## Keyboard pruning

Games in the catalogue range from four-arrow mazes to a rhythm game that listens to
one row of letters. If the adapter only offers arrows, that game sits on its
instruction screen for the whole budget.

The adapter publishes `state.input_universe` — every key it can send — and the
planner may return `exclude`, the ids this game demonstrably ignores. The adapter
owns the decision to apply it, and refuses any request that would leave fewer than
four keys or no play key. Excluding an id also excludes its physical alias
(`key_w` and `move_up_alt` send the same key), otherwise the game would still
receive it.

The executor never sees the universe, so it cannot name a key that is not live this
step.

## Rate limits and dead endpoints

A 429 is back-pressure, not a failure: the same question is asked again after a
sleep, because a throttled step still has a valid label. Auth failures (401/403) are
definitive and are not retried. Other retryable statuses back off exponentially.

## Step clock versus game time

Inference takes seconds; game time does not wait. The browser host freezes the
original simulation while models are thinking and advances it at 60 Hz when input is
delivered, so a decision arrives inside the game's own timing windows.

A stepped run therefore proves feasibility of the decision hierarchy, not wall-clock
real-time control. Real-time is a separate opt-in profile.

## Recording for distillation

`ModelClient` keeps every request/response pair in wire form; the policies stamp the
accepted contract object onto the call as `label`; `RoleRecorder` writes those to
`roles.jsonl` per run.

Every exchange is written, including the ones that cannot be used as supervision.
A filtered dataset can be derived from a complete one, never the reverse, so the
recorder marks rows instead of dropping them:

- `error` non-empty with a transport cause: no reply arrived, counted as `failed`;
- `error: "reasoning only"`: the reply is the model's reasoning draft, because
  `content` came back empty. This is a budget problem, not a prompt problem;
- `error: "label rejected"`: an answer arrived but failed contract validation;
- `proven: false`: the step had no readable numeric metric, so the label cannot be
  scored against game state. The row is still written.

`summary.json` reports `failed`, `invalid` and a `reasons` breakdown, so a batch can
be triaged by cause rather than by a single failure count.

Each row carries `schema`, `role`, `game`, `run_id`, `goal`, `step`,
`plan_revision`, `trigger`, `messages`, `label`, `raw_reply`, `frame_sha`,
`from_reasoning`, `proven` and `progress`, so a dataset assembled across a refactor
can be filtered by `schema` instead of silently mixing two prompt shapes.

Frames are stored once per content hash under `frames/`, and `trace.jsonl` refers to
the frame by `image_sha` rather than inlining base64. Inlining cost 340 KB per step
and made a 13-step trace 4 MB; with the reference it is 73 KB.

Both roles are budgeted separately, because a plan revision is far longer than one
action: `JEVGAMER_EXECUTOR_MAX_TOKENS` (default 1024) and
`JEVGAMER_PLANNER_MAX_TOKENS` (default 2048). A reply that stops on the ceiling is
retried once at double the budget, since a truncated JSON object is not a short
answer but an unparseable one.

## Layout

```
general_agent/
  contract.py       the wire contract: dataclasses, parsing, validation, rendering
  prompts.py        executor and planner system prompts
  client.py         one chat client for both roles, with retry and recording
  policy.py         executor: action choice, triggers, plan adoption, fallback
  outer.py          planner: plan revisions, never an action
  runner.py         episode loop: reset -> observe -> choose -> act
  types.py          Action / Observation / Decision / EpisodeResult
  live.py           wiring of both roles + live panel
  sft.py            RoleRecorder: calls -> labelled JSONL
  recurrence.py     repeated-public-state detection
  scene_memory.py   observed-scene snapshots for continuity
  monitor.py        read-only inspection of running episodes
  adapters/         per-family action space: html5, generic_html, gzdoom, ...
agent_harness/      shared model-turn history (Thread / Turn)
webenv/             browser host, DOM projection, step clock
map/                game catalogue metadata (ids, categories, entry paths)
docs/ARCHITECTURE.md  the decision loop in detail
```

The boundary is public observation, then model judgment, then normal input, then a
verified effect. An adapter makes a scene legible; it must not become a second game
engine. Rules, controls and success predicates belong to data, not to code branches.

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest general_agent/tests -q
```

Expected on this snapshot: `225 passed, 2 subtests passed`. The suite covers the
contract (parsing, merging, lifetime rules, canonical encoding), the trigger order,
plan adoption and rejection, the planner's refusal to name an action, keyboard
pruning, output budgets and truncation recovery, the distillation writer, dataset
assembly, game selection for a batch, and the live panel's field names.

## Collecting a batch

```bash
set JEVGAMER_MAP_ROOT=D:\path\to\game\assets
python tools\collect_sft.py --dry-run              # list games and budgets, spend nothing
python tools\collect_sft.py --workers 3 --steps 40 # then actually run them
python tools\build_dataset.py                      # merge into sft.jsonl + manifest.json
```

`--dry-run` prints the resolved game list, the pack root, the map root and both
token budgets before any browser starts. That check exists because an unset map root
previously made every game fail identically after the browser had already launched.

Each game gets its own folder and its own browser, so `--workers` parallelises games
without letting two games share DOM state. `--resume` skips games whose index row
says they completed.

`build_dataset.py` produces three files: `sft.jsonl` (every row, whole
trajectories), `sft_clean.jsonl` (rows with a validated label; add `--proven-only`
to also require a numeric metric) and `manifest.json` (counts by role, game, trigger
and error reason, plus trajectory lengths).

## Configuration

Secrets are read from the environment, never from the tree:

| Variable | Meaning |
| --- | --- |
| `JEVGAMER_TOKEN` | API token for both roles |
| `JEVGAMER_CHAT` | chat endpoint (fallback when the local proxy is down) |
| `JEVGAMER_CHAT_PROXY` | local rate proxy, preferred when listening |
| `JEVGAMER_EXECUTOR_MODEL` | executor model, default `deepseek-v4.1-flash` |
| `JEVGAMER_PLANNER_MODEL` | planner model, default `deepseek-v4.1-flash` |
| `JEVGAMER_EXECUTOR_CONFIDENCE` | self-reported confidence threshold, default 0.45 |
| `JEVGAMER_EXECUTOR_MAX_TOKENS` | executor output budget, default 1024 |
| `JEVGAMER_PLANNER_MAX_TOKENS` | planner output budget, default 2048 |
| `JEVGAMER_MAP_ROOT` | where game assets live, when they are outside `map/` |
| `JEVGAMER_SEED` | deterministic input seed, default 42 |
| `JEVGAMER_SFT_DIR` | root for `roles.jsonl` output |

Thinking is off by default for both roles. Measurements showed a review spending
~4785 characters of hidden reasoning and ~1250 completion tokens for a response that
is a small JSON object: 10.6s per review, and sometimes the whole budget consumed so
`content` came back empty. With thinking off the same review took ~1s and ~142
tokens of valid JSON.

## What this repository is not

It is the decision core, not the whole research tree. The private upstream repository
also holds a 188-game library, knowledge packs, recorded runs, and several hundred
ad-hoc probe scripts. Game assets are absent here on purpose: the catalogue is
included so the adapters import and so the validated action space is documented, but
the games themselves are not.
