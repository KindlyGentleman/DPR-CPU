"""Per-case frozen-model parity and physical-scale scientific metrics, without timing."""

from .compare import main
from .common import cli

if __name__ == "__main__":
    cli(lambda: main(validate_only=True))
