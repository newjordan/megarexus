#!/usr/bin/env bash
# Source this before PufferLib-backed Rexus RL work:
#   source codexlab/pufferlib_env.sh
#
# The script only prepares local process environment variables. It does not
# install packages or mutate the external checkout.

if [ -n "${BASH_SOURCE[0]:-}" ]; then
  _PUFFER_ENV_FILE="${BASH_SOURCE[0]}"
else
  _PUFFER_ENV_FILE="$0"
fi

REXUS_ROOT="$(cd "$(dirname "$_PUFFER_ENV_FILE")/.." && pwd)"

if [ -z "${PUFFERLIB_PATH:-}" ]; then
  if [ -d "$REXUS_ROOT/.external/pufferlib-4.0-newjordan" ]; then
    export PUFFERLIB_PATH="$REXUS_ROOT/.external/pufferlib-4.0-newjordan"
  elif [ -d /tmp/pufferlib-4.0-newjordan ]; then
    export PUFFERLIB_PATH="/tmp/pufferlib-4.0-newjordan"
  else
    export PUFFERLIB_PATH="$REXUS_ROOT/.external/pufferlib-4.0-newjordan"
  fi
fi

case ":${PYTHONPATH:-}:" in
  *":$REXUS_ROOT/codexlab:"*) ;;
  *) export PYTHONPATH="$REXUS_ROOT/codexlab${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

case ":${PYTHONPATH:-}:" in
  *":$PUFFERLIB_PATH:"*) ;;
  *) export PYTHONPATH="$PUFFERLIB_PATH${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

echo "PUFFERLIB_PATH=$PUFFERLIB_PATH"
echo "PYTHONPATH=$PYTHONPATH"
