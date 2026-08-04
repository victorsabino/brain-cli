"""Error types raised by brain_mcp.adapter and surfaced through FastMCP tools."""


class BrainError(Exception):
    """Generic non-zero exit from a brain.py cmd_* call that isn't one of the
    more specific cases below."""


class BrainUnavailable(BrainError):
    """brain.db is missing, or connect() otherwise sys.exit()'d."""


class BrainNotFound(BrainError):
    """A uid referenced by get/link/feedback does not exist. Prefer returning
    the structured {"found": false, ...} shape from the tool instead of
    raising this where the spec calls for a non-exception failure path."""


class BrainInvalidInput(BrainError):
    """Bad type/kind/signal, or a validation failure surfaced via _err()."""
