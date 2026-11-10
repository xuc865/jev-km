# jev-km

A two-tier (fast/slow) decision system for long-horizon game agents.

The design premise is that a single model cannot be both cheap enough to act every
step and wise enough to plan across a long episode. So decisions are split by
latency budget rather than by task: a small multimodal scorer picks the actual
action at every step, and a large reasoning model intervenes only when the fast
tier is unsure or when the situation has changed enough that the plan is stale.

```
                          ┌──────────────────────────────────┐
                          │  Slow tier: planner / reviewer   │
                          │  (large reasoning model)         │
                          └──────────────────────────────────┘
                             ▲ publishes          │ action_criteria
                             │ context revision   │ control_exclude (keyboard prune)
                             │                    ▼
   observe ──► ┌──────────────────────────────────────────┐ ──► execute ──► observe
               │  Fast tier: candidate scorer              │
               │  ranks the legal actions for THIS step    │
               └──────────────────────────────────────────┘

   every exchange of both tiers is recorded as a training sample
```

## The two tiers

Fast tier — `general_agent/policy.py`

`JevPolicy` sends the current observation, the screenshot and a usage sentence for
each legal action to a small multimodal scorer, which returns one probability per
candidate. It takes the argmax, subject to a confidence threshold. It cannot
invent an action: the candidate list comes from the adapter, which owns the game's
actual action space.

Slow tier — `general_agent/outer.py`

`OuterEscalation` is a larger reasoning model. It never plays step by step. It emits
a layered plan that the fast tier then executes:

| Field | Lifetime | Purpose |
| --- | --- | --- |
| `briefing` | stable across reviews | what the game is, controls, win condition |
| `stage_goal` | until the phase changes | the objective of the current phase or level |
| `play_instruction` | until revised | concrete steps for the fast tier this stage |
| `situation` | one review | one sentence on current state and risk |
| `action` | one step | an optional exact action id to execute |

The split matters because the two fields age differently. `stage_goal` and
`play_instruction` are conditions written against observation fields, so they stay
valid for many fast steps; a coordinate copied into a persistent instruction goes
stale within a few frames. The prompt therefore asks for conditions that reference
observation fields, and one-off decisions belong in `action`.

## When the slow tier is consulted

Calling the planner every step would erase the entire point of having a fast tier.
`JevPolicy.choose` triggers a review on any of:

- the first step of the episode
- the strict confidence threshold is not met (`top < confidence`)
- the fast scorer returned an error or an empty distribution
- the recurrence tracker sees the same public state three times in its window
- the phase, level or headings changed
- the player moved to a new cell and at least two steps have passed since review
- the last three actions were identical
- no plan is cached yet, or the periodic interval elapsed (`call_outer_every`)

A review is an optimisation, not a precondition. A failed review increments a
counter and the fast tier keeps playing; only six consecutive failures stop the
episode. This was a correction from measurement: an earlier version ended the run
on the first empty response, which turned a 37-step trajectory into a 4-step one
because the very first review is unskippable.

## Confidence and escalation

The fast tier's answer is a distribution over legal actions, which makes the
"should I escalate" question a threshold on the top probability rather than a
separate classifier. Below `JEVGAMER_JEV_CONFIDENCE` (default 0.6) the decision is
escalated. If the slow tier also declines, the policy falls back to the fast
tier's best non-wait candidate and marks the decision `jev-low`; if there is no
such candidate it stops. There is deliberately no hand-coded game fallback — a
right-hand rule only works for mazes and is wrong for shooters, card games and
platformers.

## Rate limits and dead endpoints

Both failure modes are about the fast endpoint, and they are not the same problem:

A 429 produces an empty probability table during the vendor's cooldown window.
Waiting is correct, so the call retries up to six times with a twelve-second
pause. Treating this as a game failure once collapsed 30-sample episodes into
3-sample ones across a 38-way sweep.

An endpoint that accepts the TCP connection but never answers will not clear by
waiting. After two consecutive failures the scorer switches to a chat transport
that answers the same choice question as text classification, and the switch is
sticky for the episode so a dead endpoint is probed once rather than every step.

## Step clock versus game time

Inference takes seconds; game time does not wait. The browser host freezes the
original simulation while models are thinking and advances it at 60 Hz when input
is delivered, so a decision arrives inside the game's own timing windows instead of
after them. This keeps original timers, stats and difficulty untouched.

The distinction that follows is important and is why the recordings are labelled:
a stepped run proves feasibility of the decision hierarchy, not wall-clock
real-time control. Real-time is a separate opt-in profile where timers keep
running during inference, and a stepped recording cannot stand in for it.

## Recording both tiers for training

`sft.py` writes two append-only JSONL streams per episode:

`fast.jsonl` — the exact scorer input (world text, every per-action criterion, the
attached images, the rebuilt instruction prefix) plus the returned distribution.
This is the distillation target for a smaller student: it stores the rendered
sequences rather than the parts, so a student trains on the same tokens the
teacher saw.

`slow.jsonl` — the planner's prompt, response, chosen action, usage and latency.

Samples carry the game's own progress metric at that step, so a run can be split
anywhere and still know whether the state was advancing. Two cases are counted
rather than written, because writing them would teach the student from noise: a
step where the scorer returned no distribution has no label, and a step before the
game published its metric is the title screen, not progress.

## Layout

```
general_agent/
  policy.py         fast tier: JevPolicy, confidence gate, escalation triggers
  outer.py          slow tier: OuterEscalation, layered plan, keyboard pruning
  runner.py         episode loop: reset -> observe -> choose -> act
  types.py          Action / Observation / Decision / EpisodeResult
  live.py           wiring of both tiers + live panel
  sft.py            FastRecorder / SlowRecorder / SampleWriter
  recurrence.py     repeated-public-state detection
  scene_memory.py   observed-scene snapshots for continuity
  monitor.py        read-only inspection of running episodes
  adapters/         per-family action space: html5, generic_html, gzdoom, ...
agent_harness/      shared model-turn history (Thread / Turn)
webenv/             browser host, DOM projection, step clock
map/                game catalogue metadata (ids, categories, entry paths)
```

The boundary is public observation, then model judgment, then normal input, then a
verified effect. An adapter makes a scene legible; it must not become a second game
engine. Rules, controls and success predicates belong to data, not to code branches.

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest general_agent/tests -q
```

Expected on this snapshot: `91 passed, 7 failed`. The seven are pre-existing
failures in the upstream private repository, not artefacts of this extract — they
were verified to fail identically there. They are left visible rather than skipped
because two of them encode a behaviour that is currently wrong: `test_runner` and
`test_panel_dom` both expect a slow-tier transport exception to stop the episode
before any further input, while the current policy deliberately continues on the
fast tier and stops only after six consecutive failures. That is an open
inconsistency between the intended contract and the implemented one, not a missing
file.

## Configuration

Secrets are read from the environment, never from the tree:

| Variable | Meaning |
| --- | --- |
| `JEVGAMER_TOKEN` | API token for both tiers |
| `JEVGAMER_CHAT` | chat endpoint for the slow tier |
| `JEVGAMER_SUPERVISOR` | slow-tier model name |
| `JEVGAMER_JEV_CONFIDENCE` | confidence gate, default 0.6 |
| `JEVGAMER_JEV_QPM` | fast-tier call budget per minute, default 600 |
| `JEVGAMER_SLOW_THINKING` | `1` restores reasoning tokens for the slow tier |

Slow-tier thinking defaults to off. Measurements on this system showed a review
spending ~4785 characters of hidden reasoning and ~1250 completion tokens for a
response that is a small JSON object, costing 10.6s per review and sometimes
exhausting the budget so the content came back empty. With thinking off the same
review took ~1s and ~142 tokens with valid JSON.

## What this repository is not

It is the decision core, not the whole research tree. The private upstream
repository also holds a 188-game library, knowledge packs, recorded runs, and
several hundred ad-hoc probe scripts. Game assets are absent here on purpose: the
catalogue is included so the adapters import and so the validated action space is
documented, but the games themselves are not.
