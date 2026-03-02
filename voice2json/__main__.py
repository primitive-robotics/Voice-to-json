"""
Entry point for `python -m voice2json`.

Loads .env (if present), then starts the main loop.
"""

import sys
import os


def _load_dotenv() -> None:
    """Best-effort load of .env file without requiring python-dotenv."""
    env_file = ".env"
    if not os.path.exists(env_file):
        return
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            # Strip inline comments (e.g. VALUE=foo  # comment)
            value = value.split("#")[0].strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def main() -> None:
    _load_dotenv()

    # Deferred import so env vars are set before modules read them
    from voice2json.app import run_loop

    try:
        run_loop()
    except KeyboardInterrupt:
        print("\n[app] Interrupted by user. Goodbye.", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
