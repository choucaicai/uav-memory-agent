"""Action normalization keys shared by raw training and evaluation."""

RAW_DATASET_NAME = "uavon_raw_json"


def resolve_action_stats_key(norm_stats, requested=None):
    """Prefer raw-run statistics; retain support for older merged checkpoints."""
    available = {
        key for key, value in norm_stats.items()
        if isinstance(value, dict) and "action" in value
    }
    if requested is not None:
        if requested not in available:
            raise ValueError(f"Unknown action statistics key {requested!r}; available: {sorted(available)}")
        return requested
    for key in (RAW_DATASET_NAME, "oct_v1_target"):
        if key in available:
            return key
    if len(available) == 1:
        return next(iter(available))
    raise ValueError(f"Set --unnorm_key explicitly; available action statistics: {sorted(available)}")
