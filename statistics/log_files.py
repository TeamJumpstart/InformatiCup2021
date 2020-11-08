from pathlib import Path


def get_log_files(log_dir, prefix=""):
    """Get a list of all logfiles in `log_dir`.

    Args:
        prefix: Can be used to filter by date, e.g. use `"20201106-10"` to only get logfiles from the 6th November
                from 10:00 til 10:59.
    """
    return [f for f in Path(log_dir).iterdir() if f.name.endswith(".json") and f.name.startswith(prefix)]
