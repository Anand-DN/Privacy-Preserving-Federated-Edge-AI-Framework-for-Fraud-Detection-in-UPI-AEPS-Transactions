"""Compatibility entry point for the federated experiment pipeline.

The full implementation now lives in main.py so there is one authoritative
training/evaluation path.
"""

from main import main


if __name__ == "__main__":
    main()
