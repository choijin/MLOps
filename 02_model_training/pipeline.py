"""Compatibility shim for the reorganized training pipeline entrypoint."""

from pipeline.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
