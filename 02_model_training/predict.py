"""Compatibility shim for the reorganized local prediction entrypoint."""

from models.predict import main


if __name__ == "__main__":
    raise SystemExit(main())
