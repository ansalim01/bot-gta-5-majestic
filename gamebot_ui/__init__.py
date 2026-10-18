"""Desktop control panel for the game bot collection."""


def run(*args, **kwargs):
    """Import Qt lazily so non-GUI tools can still use package metadata."""
    from .app import run as run_application

    return run_application(*args, **kwargs)


__all__ = ["run"]
