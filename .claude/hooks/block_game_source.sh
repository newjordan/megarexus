#!/usr/bin/env bash
# Lab honesty guard (Phase 0, DREAMWEAVER_INTEGRATION_PLAN_20260706.md).
# Game engine source under environment_files/ is an ANSWER KEY: reading it to
# build solvers/labels was ruled cheating three times (sp80_spill_model,
# route-replay, dropletsim 2026-07-06). The env may EXECUTE these files via the
# harness; agents may not READ (or tamper with) them.
# Only the tool's TARGET fields are inspected — file_path / command / search
# path — so notes and docs may freely MENTION the banned paths.
input=$(cat)
tool=$(printf '%s' "$input" | jq -r '.tool_name // empty')
case "$tool" in
  Read|Edit|Write|NotebookEdit)
    target=$(printf '%s' "$input" | jq -r '(.tool_input.file_path // "") + " " + (.tool_input.notebook_path // "")')
    pattern='environment_files/[^"'"'"' ]*\.py'
    ;;
  Grep|Glob)
    target=$(printf '%s' "$input" | jq -r '(.tool_input.path // "") + " " + (.tool_input.glob // "")')
    pattern='environment_files'   # searching inside the env dir can dump source content
    ;;
  Bash)
    target=$(printf '%s' "$input" | jq -r '.tool_input.command // ""')
    pattern='environment_files/[^"'"'"' ]*\.py'
    ;;
  *)
    exit 0
    ;;
esac
if printf '%s' "$target" | grep -Eq "$pattern"; then
  cat <<'EOF'
{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"BLOCKED — lab honesty ruling (answer-key ban 2026-07-01; third violation retracted 2026-07-06): game engine source under environment_files/ must never be read or modified. Learn win/flow semantics from observed frames and env interaction only. See synthesis_lab/notes/INVALID_DROPLETSIM_ANSWERKEY_20260706.md and DREAMWEAVER_INTEGRATION_PLAN_20260706.md section 1."}}
EOF
fi
exit 0
