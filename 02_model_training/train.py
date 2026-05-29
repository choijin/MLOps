"""Compatibility shim for the reorganized training entrypoint."""

from models.train import main


if __name__ == "__main__":
    raise SystemExit(main())
