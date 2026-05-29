"""Compatibility shim for the reorganized model comparison entrypoint."""

from models.compare_models import main


if __name__ == "__main__":
    raise SystemExit(main())
