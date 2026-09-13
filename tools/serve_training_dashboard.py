"""Read-only training dashboard for S0-TriMask / S0-TriMask-HS runs (Python standard library only).

    python tools/serve_training_dashboard.py \
        --run s0_trimask_hs_v02=runs_salu/said_s0_trimask_v01_hs/step500 \
        --run s0_trimask_v01=/root/SAID-s0-trimask-v01/runs_salu/said_s0_trimask_v01/step500

Then, from your own machine::

    ssh -N -L 8765:127.0.0.1:8765 <your-ssh-target>
    # and open http://127.0.0.1:8765

Design rules (task section 9):

* **read only** -- the only endpoints are the five documented GETs plus static files and a health
  check. There is no way to start a run, stop a process, change a parameter, delete a file or
  execute a command, and the service never writes inside a run directory;
* **no training coupling** -- it does not import torch, does not open a checkpoint, does not run a
  forward pass and does not touch a GPU. A broken page, a closed browser or a killed service cannot
  affect training;
* **loopback by default** -- binds ``127.0.0.1:8765``; if the port is taken it reports the actual
  port it bound instead of silently moving;
* **no client paths** -- a run id is resolved through the registry built at start-up, so path
  traversal is not representable; static files are additionally confined to the web directory.

The repository already ships a Streamlit dashboard (``tools/said_dashboard``) for pre-exported
attention artifacts. It is left untouched: it has a different artifact model and no HTTP API to
test, while this panel must serve a live run's JSONL incrementally. The data layer
(:mod:`tools.dashboard_data`) is transport independent, so a Streamlit page could import it later
without changing anything here.
"""
import argparse
import json
import os
import posixpath
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS_DIR)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from dashboard_data import DashboardData, RunNotFound, RunRegistry  # noqa: E402

WEB_DIR = os.path.join(REPO_ROOT, 'web', 'training_dashboard')
STATIC_TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'application/javascript; charset=utf-8',
                '.css': 'text/css; charset=utf-8'}
DEFAULT_PORT = 8765
DEFAULT_HOST = '127.0.0.1'


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = 'trimask-dashboard/1.0'
    data = None                       # injected by the server
    web_dir = WEB_DIR

    # -- helpers ------------------------------------------------------------ #
    def _send(self, status, body, content_type='application/json; charset=utf-8'):
        payload = body if isinstance(body, bytes) else body.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(payload)

    def _json(self, status, value):
        self._send(status, json.dumps(value, ensure_ascii=False, allow_nan=False))

    def _error(self, status, message):
        self._json(status, {'error': message, 'status': status})

    def log_message(self, fmt, *args):      # keep the service log quiet and one line per request
        sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % args))

    # -- routing ------------------------------------------------------------ #
    def do_GET(self):
        self._dispatch('GET')

    def do_HEAD(self):
        self._dispatch('HEAD')

    def _dispatch(self, method):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        try:
            if path in ('/health', '/api/health'):
                return self._json(200, {'status': 'ok', 'service': 'trimask-dashboard',
                                        'runs': self.data.registry.ids(),
                                        'read_only': True,
                                        'gpu_used': False, 'checkpoints_loaded': False})
            if path == '/api/runs':
                return self._json(200, {'runs': self.data.list_runs()})
            if path.startswith('/api/run/'):
                parts = [part for part in path[len('/api/run/'):].split('/') if part != '']
                if not parts:
                    return self._error(404, 'missing run id')
                run_id = parts[0]
                leaf = parts[1] if len(parts) > 1 else 'status'
                if len(parts) > 2:
                    return self._error(404, 'unknown endpoint')
                self.data.registry.resolve(run_id)          # raises for unknown/traversal ids
                if leaf == 'status':
                    return self._json(200, self.data.status(run_id))
                if leaf == 'metrics':
                    after = query.get('after', ['0'])[0]
                    limit = query.get('limit', [None])[0]
                    try:
                        after = int(after)
                        limit = int(limit) if limit is not None else None
                    except ValueError:
                        return self._error(400, 'after/limit must be integers')
                    block = self.data.metrics(run_id, after=after,
                                              **({'limit': limit} if limit else {}))
                    return self._json(200, block)
                if leaf == 'masks':
                    return self._json(200, self.data.masks(run_id))
                if leaf == 'evaluation':
                    return self._json(200, self.data.evaluation(run_id))
                if leaf == 'diagnostics':
                    # read-only offline probe results; when the probe has not been run this returns
                    # an explicit 未诊断 payload and never a fabricated zero
                    return self._json(200, self.data.diagnostics(run_id))
                if leaf == 'logs':
                    limit = int(query.get('limit', ['200'])[0])
                    return self._json(200, self.data.logs(run_id, limit=max(1, min(limit, 1000))))
                if leaf == 'metrics.csv':
                    csv_text = self.data.metrics_csv(run_id)
                    return self._send(200, csv_text, 'text/csv; charset=utf-8')
                return self._error(404, 'unknown endpoint')
            if path in ('/', '/index.html'):
                return self._static('index.html')
            if path.startswith('/static/'):
                return self._static(path[len('/static/'):])
            if path == '/favicon.ico':
                return self._send(204, b'', 'image/x-icon')
            return self._error(404, 'not found')
        except RunNotFound:
            # an unknown id, an invalid id and a traversal attempt are all indistinguishable here on
            # purpose: the client never learns whether a path outside the registry exists
            return self._error(404, 'unknown run')
        except Exception as error:                      # pragma: no cover - defensive
            return self._error(500, 'internal error: %s' % error)

    def _static(self, relative):
        candidate = posixpath.normpath('/' + relative).lstrip('/')
        if candidate.startswith('..') or os.path.isabs(candidate):
            return self._error(403, 'path traversal rejected')
        full = os.path.realpath(os.path.join(self.web_dir, candidate))
        root = os.path.realpath(self.web_dir)
        if full != root and not full.startswith(root + os.sep):
            return self._error(403, 'path traversal rejected')
        if not os.path.isfile(full):
            return self._error(404, 'not found')
        extension = os.path.splitext(full)[1].lower()
        if extension not in STATIC_TYPES:
            return self._error(403, 'unsupported static type')
        with open(full, 'rb') as handle:
            return self._send(200, handle.read(), STATIC_TYPES[extension])


def build_registry(entries, config_path=None):
    """``--run id=dir`` entries plus an optional JSON config file.

    A ``--run`` value may carry extra ``|key=value`` fields, e.g.
    ``--run "hs=dir|prefix=S0_TriMask_HS_step000500|label=HS v0.2"``; the prefix only overrides which
    evaluation file is preferred, and every path still has to live inside the registered directory.
    """
    registry = RunRegistry()
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as handle:
            for item in json.load(handle):
                registry.register(item['run_id'], item['directory'], label=item.get('label'),
                                  demo=item.get('demo', False),
                                  objective=item.get('objective'), arm=item.get('arm'),
                                  evaluation_prefix=item.get('evaluation_prefix'))
    for entry in entries or []:
        if '=' not in entry:
            raise SystemExit('--run expects run_id=directory, got %r' % entry)
        run_id, rest = entry.split('=', 1)
        fields = rest.split('|')
        directory = fields[0]
        options = {}
        for field in fields[1:]:
            if '=' not in field:
                raise SystemExit('--run option %r must be key=value' % field)
            key, value = field.split('=', 1)
            if key in ('prefix', 'evaluation_prefix'):
                options['evaluation_prefix'] = value
            elif key == 'label':
                options['label'] = value
            elif key == 'objective':
                options['objective'] = value
            elif key == 'arm':
                options['arm'] = value
            elif key == 'demo':
                options['demo'] = value.lower() in ('1', 'true', 'yes')
            else:
                raise SystemExit('unknown --run option %r' % key)
        registry.register(run_id, directory, **options)
    return registry


def build_server(registry, host=DEFAULT_HOST, port=DEFAULT_PORT, web_dir=WEB_DIR,
                 port_attempts=10):
    """Bind the server, reporting the port actually used.

    A busy port is never silently ignored: the next free port is tried and the fallback is both
    returned and printed, so the SSH tunnel command the user copies is the correct one.
    """
    data = DashboardData(registry)
    handler = type('BoundDashboardHandler', (DashboardHandler,),
                   {'data': data, 'web_dir': web_dir})
    last_error = None
    for candidate in range(port, port + max(1, port_attempts)):
        try:
            server = ThreadingHTTPServer((host, candidate), handler)
        except OSError as error:
            last_error = error
            continue
        server.daemon_threads = True
        # the bound port, not the requested one: with --port 0 the OS picks an ephemeral port and
        # the caller (and the printed URL) must see the real value
        return server, data, server.server_address[1]
    raise SystemExit('cannot bind %s:%d-%d: %s'
                     % (host, port, port + port_attempts - 1, last_error))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', default=[],
                        help='run_id=directory (repeatable); the directory is registered at '
                             'start-up and no request can reach outside it')
    parser.add_argument('--runs-config', default=None, help='optional JSON list of run entries')
    parser.add_argument('--host', default=os.environ.get('TRIMASK_DASHBOARD_HOST', DEFAULT_HOST))
    parser.add_argument('--port', type=int,
                        default=int(os.environ.get('TRIMASK_DASHBOARD_PORT', DEFAULT_PORT)))
    parser.add_argument('--web-dir', default=WEB_DIR)
    args = parser.parse_args()

    registry = build_registry(args.run, args.runs_config)
    if not registry.ids():
        print('WARNING: no run registered; the page will show an empty run list', flush=True)
    server, _, actual_port = build_server(registry, args.host, args.port, args.web_dir)
    actual_host = server.server_address[0]
    print(json.dumps({'service': 'trimask-dashboard', 'pid': os.getpid(),
                      'host': actual_host, 'port': actual_port,
                      'url': 'http://%s:%d' % (actual_host, actual_port),
                      'requested_port': args.port, 'port_fallback': actual_port != args.port,
                      'runs': registry.ids(), 'web_dir': os.path.realpath(args.web_dir),
                      'read_only': True, 'gpu_used': False}, ensure_ascii=False), flush=True)
    if actual_port != args.port:
        print('WARNING: requested port %d was busy; bound %d instead' % (args.port, actual_port),
              flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
