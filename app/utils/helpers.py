from datetime import datetime


def get_current_timestamp():
    return datetime.utcnow().isoformat()


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default