AGENTS := ARC-AGI-3-Agents

action:
	@test -d $(AGENTS) || { echo "missing $(AGENTS). This checkout does not include that submodule."; exit 1; }
	uv run $(AGENTS)/main.py --agent=action

install:
	@test -d $(AGENTS) || { echo "missing $(AGENTS). This checkout does not include that submodule."; exit 1; }
	uv venv
	cd $(AGENTS) && UV_PROJECT_ENVIRONMENT=../.venv uv sync --all-extras
	uv pip install -r requirements.txt

tensorboard:
	.venv/bin/tensorboard --logdir=runs --port=6006

clean:
	rm -r ./runs