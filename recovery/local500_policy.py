"""Local reproduction control only; no model, sampler or optimizer changes."""
POLICY = 'local_reproduction500'


def cycle_state(seconds, consecutive=0):
    return dict(consecutive=0, warning=seconds > 3, stop=seconds > 60)


def wait_expired(heartbeat, now):
    return (heartbeat.get('state') in ('DATA_WAIT', 'ACTIVE_STEP') and
            now - heartbeat['started_monotonic'] > 60)
