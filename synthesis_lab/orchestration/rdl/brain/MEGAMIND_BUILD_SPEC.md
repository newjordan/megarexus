# MEGAMIND Build Spec — orchestrator contract

**Goal:** Bi-hemispherical FULL physics + mapping + conceptual operators under
`synthesis_lab/orchestration/rdl/brain/`.

**Hard bans:** no `dropletsim`, `native_scene` as solver, `sp80_spill_model`,
`_invalid/`, human-route train labels, fixture win banks as labels.
Env/frame deltas only for training signals.

**Existing scaffold (do not rewrite blindly):**  
`hemispheres.py`, `central.py`, `oscillate.py`, `smoke_oscillate.py`, `DESIGN.md`

**Python:** `python3`  
**GPU:** often OOM (vLLM); prefer CPU for smokes unless free.

---

## Package layout (target)

```text
rdl/brain/
  DESIGN.md
  MEGAMIND_BUILD_SPEC.md
  __init__.py
  types.py              # KnownMapView, OperatorResult, BrainState
  operators/
    __init__.py
    base.py             # Operator protocol
    body.py             # stream/sphere/learned plugs (BodyOp)
    ray_ops.py          # cast_rays, ray_aabb wrappers (RayOp)
    field_ops.py        # signed potential attract/repel (FieldOp) — pure numpy OK
    conceptual.py       # PortBand/components conceptual proposals
  map_trust.py          # frame → scene → gate → trusted map (fail closed)
  hemisphere_i.py       # Internal full: map+ops+propose+dream
  hemisphere_e.py       # External full: world pred + judge I
  central.py            # keep + extend fuse
  oscillate.py          # keep 3-phase; accept real feature bundles
  megamind.py           # facade: step(frame, actions) → BrainState
  collect_real.py       # tiny Arcade transition collector (legal)
  smoke_megamind.py     # end-to-end smoke
```

---

## Interfaces (required)

### types.BrainState
```python
frame, trusted: bool, gate_failures: list
h_I, h_E: Tensor | None
dream_summary: dict   # body/ray/field outputs (JSON-serializable scalars/lists)
world_summary: dict
disagreement: float
judge_I: float
action_logits: Tensor | None
trust: Tensor | None   # [α_I, α_E]
mode_logits: Tensor | None
truth_flags: all False for control/goal/action/solve admitted
```

### operators.base
```python
class Operator(Protocol):
    name: str
    def run(self, map_view, context: dict) -> OperatorResult
```

### Hemisphere contracts
- **I.run(frame, available_actions)** → h_I, plan candidates, dream_next emb, operator packet
- **E.run(frame, action, h_I?, dream?)** → h_E, pred, judge_I, disagreement
- **E.learn_step** only on real (s,a,s′)

### megamind.MegaMind
```python
observe(frame) -> BrainState          # map+I propose, no act
judge_plan(frame, action, plan) -> …  # E judges I dream
act_logits(state) -> logits           # central only
train_transition(s,a,s')              # oscillate one cycle
```

---

## Work packages

### WP-A — Physics (Body + Ray)
- `operators/body.py`: wrap sim3d HydraulicSim / imagine_pour IF importable;
  fallback pure-numpy stub that still returns structured scores.
- Prefer: add sim3d to sys.path via existing `_paths.py` pattern.
- `operators/ray_ops.py`: wrap imagination_kernel.cast_rays OR local 8-dir
  ray on solid mask; polarity mouth+/floor− generic.
- Smoke: one real SP80 reset frame if arc_agi available, else synthetic.

### WP-B — Mapping + Conceptual
- `map_trust.py`: try sim3d scene+perception_gate; on failure return
  trusted=False and still extract components via simple connected components
  (no game source).
- `operators/field_ops.py`: signed goal-attract / wall-repel on component
  centroids (megarexus-native pure numpy; no external project deps).
- `operators/conceptual.py`: legal_search components/ports if importable;
  else component centroids + symmetry score as proposals.
- All outputs hypothesis-only flags.

### WP-C — Hemispheres + Central + Real train
- `hemisphere_i.py` / `hemisphere_e.py` compose operators + stub nets from
  existing hemispheres.py (reuse encoders).
- Extend central fuse with operator summary vector (fixed dim, padded).
- `collect_real.py`: few SP80 random steps → tensors.
- `oscillate.py`: keep phases; batch can include extra feature vectors.
- `megamind.py` + `smoke_megamind.py`.

---

## Success gates (orchestrator checks)

1. `smoke_oscillate.py` still PASS (CPU).
2. `smoke_megamind.py` PASS: observe + train_transition cycle + no banned imports.
3. Import graph: `rg -n 'dropletsim|spill_model|native_scene' rdl/brain` → empty
   (except comments banning them).
4. truth_flags all false in BrainState defaults.
5. DESIGN.md not contradicted (I dream, E judge, C commits).

---

## Agent rules

- Small focused files; no giant rewrites of sim3d.
- Prefer adapters/wrappers over copy-paste of physics engines.
- If import fails, degrade gracefully with stubs + explicit `status: degraded`.
- Report: files written, smoke command+exit, remaining gaps.
