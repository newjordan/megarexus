#!/usr/bin/env python3
"""Launch the CONTRA informer-DAG HTTP service.

  python -m contra_service.serve --host 127.0.0.1 --port 8973 [--fastapi]

The stdlib server is the default (no extra deps).  ``--fastapi`` uses the
optional FastAPI/uvicorn app for OpenAPI/Swagger discovery when available, and
falls back to the stdlib server if not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODEXLAB = ROOT / "codexlab"
if str(CODEXLAB) not in sys.path:
    sys.path.insert(0, str(CODEXLAB))

from contra_service import app  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8973)
    parser.add_argument("--root", default=str(ROOT), help="repo root for raw_frame_ref resolution")
    parser.add_argument("--fastapi", action="store_true", help="use the optional FastAPI app (OpenAPI) if available")
    args = parser.parse_args(argv)

    if args.fastapi:
        try:
            import uvicorn  # noqa: PLC0415

            application = app.build_fastapi_app(root=args.root)
            print(f"contra_service FastAPI (uvicorn) on http://{args.host}:{args.port} (root={args.root})")
            uvicorn.run(application, host=args.host, port=int(args.port), log_level="warning")
            return 0
        except ImportError as exc:
            print(f"FastAPI/uvicorn unavailable ({exc}); falling back to stdlib server", file=sys.stderr)

    app.serve(host=args.host, port=int(args.port), root=args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
