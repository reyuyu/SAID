"""Control-only policy for the canonical NFS reproduction; no training math."""
POLICY = 'nfs_reproduction500'


def cycle_state(seconds, consecutive):
    consecutive = consecutive + 1 if seconds > 30 else 0
    return dict(consecutive=consecutive, warning=seconds > 3,
                stop=consecutive >= 5)


def wait_expired(heartbeat, now):
    return heartbeat.get('state') == 'DATA_WAIT' and now - heartbeat['started_monotonic'] > 60


def source_changes(actual, expected, authorized):
    if actual.keys() != expected.keys():
        raise RuntimeError('Code manifest keys changed')
    changes = {}
    for name in actual:
        if actual[name] != expected[name]:
            if authorized.get(name) != (expected[name], actual[name]):
                raise RuntimeError('Unauthorized code change: ' + name)
            changes[name] = dict(historical=expected[name], current=actual[name],
                                 reason='User-authorized NFS resource control and finite-parameter checking')
    return changes
