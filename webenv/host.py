"""Private browser host, source observation hooks, and continuous raw recording.

This module implements transport only. State projections and input definitions
belong to knowledge packs; it never chooses a game action.
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, unquote, urlsplit
from .timing import timing_contract

ROOT = Path(__file__).resolve().parents[1]
MAP_ROOT = ROOT / "map"
CLOCK = Path(__file__).with_name("clock.js")

# Explicit map: Windows reads MIME types from the registry, so .js can come back
# as text/plain and the browser then refuses to execute <script type="module">.
CONTENT_TYPES = {
    '.html': 'text/html; charset=utf-8', '.htm': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8',
    '.webmanifest': 'application/manifest+json; charset=utf-8',
    '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.webp': 'image/webp',
    '.ico': 'image/x-icon', '.bmp': 'image/bmp',
    '.woff': 'font/woff', '.woff2': 'font/woff2', '.ttf': 'font/ttf', '.otf': 'font/otf',
    '.mp3': 'audio/mpeg', '.ogg': 'audio/ogg', '.wav': 'audio/wav', '.m4a': 'audio/mp4',
    '.mp4': 'video/mp4', '.webm': 'video/webm', '.wasm': 'application/wasm',
    '.txt': 'text/plain; charset=utf-8',
}


def content_type_for(path: Path) -> str:
    known = CONTENT_TYPES.get(path.suffix.lower())
    if known:
        return known
    return mimetypes.guess_type(path.name)[0] or 'application/octet-stream'

SESSION = '''<!doctype html><html><head><meta charset="utf-8"><title>JevGamer Runtime</title>
<style>*{box-sizing:border-box}html,body{margin:0;height:100%;background:#101622;color:#e8eef7;font:15px system-ui}header{height:44px;padding:10px 18px;border-bottom:1px solid #344155}main{display:grid;grid-template-columns:minmax(0,1fr) 360px;height:calc(100% - 44px)}iframe{width:100%;height:100%;border:0;background:#000}aside{padding:18px;overflow:auto}h3{font-size:15px;color:#64cffa}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.55 system-ui}.tag{color:#8cddad}</style></head>
<body><header><b>JEV 快思考 + DeepSeek 慢思考</b> · <span id="run">独立模型闭环</span> · <span id="timing">__TIMING_LABEL__</span></header><main><iframe id="game" name="game" src="__GAME_URL__"></iframe><aside><h3>运行状态</h3><pre id="status">正在初始化环境…</pre><h3>DeepSeek 阶段策略</h3><pre id="strategy"></pre><h3>JEV 候选评分</h3><pre id="scores"></pre><h3>执行反馈</h3><pre id="execution"></pre><p class="tag">仅模型选择动作 · 原始连续录像</p></aside></main>
<script>window.__JEV_DISPLAY__=data=>{for(const id of ['run','status','strategy','scores','execution'])if(data[id]!==undefined)document.getElementById(id).textContent=String(data[id]);};</script></body></html>'''


class BrowserHost:
    def __init__(self, entry_path: str, config: dict, pack_dir: Path, artifacts: Path,
                 seed: int = 42, record: bool = True, asset_root: Path = MAP_ROOT):
        self.entry_path = entry_path
        self.asset_root = Path(asset_root).resolve()
        self.config = config
        self.timing = timing_contract(config)
        self.pack_dir = pack_dir
        self.artifacts = artifacts
        self.seed = int(seed)
        self.record = bool(record)
        self.page = self.frame = self.context = self.browser = None
        self._playwright = None
        self.server = None
        self._server_thread = None
        self.errors = []
        self._hook_hashes = {}

    def _asset(self, relative: str) -> tuple[bytes, str]:
        target = (self.asset_root / relative).resolve()
        if self.asset_root not in target.parents or not target.is_file():
            raise FileNotFoundError(relative)
        content = target.read_bytes()
        content_type = content_type_for(target)
        for hook in self.config.get('source_hooks', []):
            hook_target = (self.asset_root / Path(self.entry_path).parent / hook['file']).resolve()
            if target != hook_target:
                continue
            text = content.decode('utf-8')
            anchor = hook['anchor']
            if text.count(anchor) != 1:
                # Windows checkouts hold CRLF line endings while packs declare
                # anchors with LF. Retry on the normalized line endings.
                if '\r\n' in text:
                    text = text.replace('\r\n', '\n')
                    anchor = anchor.replace('\r\n', '\n')
                if text.count(anchor) != 1:
                    raise ValueError(f"Readonly hook anchor must match once: {hook['file']}")
            hook_path = (self.pack_dir / hook['script']).resolve()
            if self.pack_dir.resolve() not in hook_path.parents:
                raise ValueError('Hook script outside pack')
            insert = hook_path.read_text(encoding='utf-8')
            self._hook_hashes[hook['file']] = hashlib.sha256(insert.encode()).hexdigest()
            replacement = insert + '\n' + anchor if hook.get('position', 'before') == 'before' else anchor + '\n' + insert
            content = text.replace(anchor, replacement, 1).encode('utf-8')
        return content, content_type

    def start(self):
        from playwright.sync_api import sync_playwright
        self.artifacts.mkdir(parents=True, exist_ok=True)
        host = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                path = unquote(urlsplit(self.path).path)
                try:
                    if path == '/__jev/session':
                        body = SESSION.replace('__GAME_URL__', host.game_url).replace('__TIMING_LABEL__',host.timing['label']).encode()
                        kind = 'text/html; charset=utf-8'
                    elif path.startswith('/map/'):
                        body, kind = host._asset(path[len('/map/'):])
                    else:
                        raise FileNotFoundError(path)
                    self.send_response(200)
                except FileNotFoundError:
                    body, kind = b'not found', 'text/plain'; self.send_response(404)
                except Exception as exc:
                    host.errors.append(str(exc)); body, kind = b'runtime hook error', 'text/plain'; self.send_response(500)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                # Godot 4 exports ship `ensureCrossOriginIsolationHeaders: true`
                # and their wasm build needs SharedArrayBuffer, which the browser
                # only grants to a cross-origin-isolated document. Without these
                # two headers the engine loads its canvas and then silently never
                # boots: a flat grey frame, no console error, and every run of
                # those games recorded "never started".
                self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
                self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
                self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
                self.end_headers()
                try: self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError): pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.base_url = f'http://127.0.0.1:{self.server.server_port}'
        self.game_url = self.base_url + '/map/' + quote(self.entry_path, safe='/') + f'?jev_seed={self.seed}'
        self._server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._server_thread.start()
        self._playwright = sync_playwright().start()
        executable = os.environ.get('JEVGAMER_CHROME')
        if not executable and os.name != 'nt':
            candidate = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
            if candidate.is_file(): executable = str(candidate)
        kwargs = {'headless': True, 'args': ['--disable-background-timer-throttling',
                  '--disable-renderer-backgrounding', '--autoplay-policy=no-user-gesture-required']}
        if executable: kwargs['executable_path'] = executable
        self.browser = self._playwright.chromium.launch(**kwargs)
        viewport = self.config.get('viewport') or {'width': 1920, 'height': 1080}
        options = {'viewport': viewport, 'device_scale_factor': 1, 'service_workers': 'block'}
        if self.record:
            options.update(record_video_dir=str(self.artifacts / 'raw-video'), record_video_size=viewport)
        self.context = self.browser.new_context(**options)
        boot = {'seed': self.seed, 'time_mode': self.timing['mode'],
                'dt_ms': self.timing['action_quantum_ms'],
                'dom_semantics': self.config.get('dom_semantics', []),
                'dom_fields': self.config.get('dom_fields', {})}
        self.context.add_init_script(script='window.__JEV_BOOT__=' + json.dumps(boot) + ';\n' + CLOCK.read_text(encoding='utf-8'))
        self.context.add_init_script(script=CLOCK.with_name('dom.js').read_text(encoding='utf-8'))
        extra_init = self.config.get('init_script')
        if extra_init:
            self.context.add_init_script(script=(self.pack_dir / extra_init).read_text(encoding='utf-8'))
        def route(request_route):
            url = request_route.request.url
            if url.startswith(self.base_url + '/') or url.startswith(('data:', 'blob:', 'about:')):
                request_route.continue_()
            elif request_route.request.resource_type == 'script':
                request_route.fulfill(status=200, content_type='application/javascript', body='/* offline external SDK */')
            else:
                request_route.abort()
        self.context.route('**/*', route)
        self.page = self.context.new_page()
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))
        self.page.goto(self.base_url + '/__jev/session', wait_until='load', timeout=60000)
        self.frame = self.page.frame(name='game')
        if not self.frame: raise RuntimeError('Game frame not loaded')
        self._focus_surface()
        if self.config.get('time', {}).get('mode', 'step') == 'step': self.advance(int(self.config.get('time',{}).get('initial_frames',1)))
        manifest = {'entry_path': self.entry_path, 'seed': self.seed, 'clock': boot,
                    'recording': self.record, 'timing_contract':self.timing, 'viewport': viewport, 'hook_sha256': self._hook_hashes,
                    'source_sha256': hashlib.sha256((self.asset_root / self.entry_path).read_bytes()).hexdigest()}
        (self.artifacts / 'host.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        return self

    def snapshot(self):
        if self.errors: raise RuntimeError('Browser runtime error: ' + self.errors[-1])
        expression = self.config.get('snapshot', 'window.__JEV_STATE__()')
        return self.frame.evaluate('() => (' + expression + ')')

    def advance(self, frames: int):
        if self.config.get('time', {}).get('mode', 'step') == 'step':
            return self.frame.evaluate('([frames,paced]) => window.__JEV_CLOCK__.advanceFrames(frames,paced)', [int(frames), self.record])
        self.page.wait_for_timeout(int(frames) * self.timing['action_quantum_ms'])

    def _focus_surface(self) -> None:
        """Give the game frame keyboard focus, whatever element receives it.

        The host used to call `locator('#game').focus()` unconditionally. A bare
        `<canvas id="game">` is not focusable without a tabindex, so Playwright
        waited 30s and threw "Locator.focus: Timeout" before the game ever ran:
        the episode recorded zero steps and looked like a game that refuses to
        start. Keyboard events only need the document to be focused, so a canvas
        is attempted first and the body is the fallback.
        """
        if self.page is None:
            return
        try:
            self.page.locator('#game').focus(timeout=3000)
            return
        except Exception:
            pass
        for target in ('body', 'canvas'):
            try:
                self.page.locator(target).first.focus(timeout=2000)
                return
            except Exception:
                continue

    def keys(self, keys, frames):
        self._focus_surface()
        try:
            for key in keys: self.page.keyboard.down(key)
            self.advance(int(frames))
        finally:
            for key in reversed(keys): self.page.keyboard.up(key)

    def click(self, selector, nth=None):
        locator = self.frame.locator(selector)
        if nth is not None: locator = locator.nth(nth)
        locator.click(timeout=10000)

    def click_at(self, x_ratio: float, y_ratio: float, frames: int = 1):
        """Click a point given as a fraction of the canvas box.

        Tower defence, card and puzzle games take no meaningful keyboard input:
        the only way to make anything happen is to click a tile. Without a
        coordinate click those games were driven by arrow keys for the whole
        episode and the metric never moved, which is what kept them stalled.
        The event is synthesised on the element under the point rather than
        using Playwright's mouse, so it works inside the game iframe and does
        not depend on the outer page's scroll position.
        """
        self.frame.evaluate(
            """([xr, yr]) => {
              var c = document.querySelector('canvas');
              var r = c && c.getBoundingClientRect && c.getBoundingClientRect();
              var left = r && r.width ? r.left : 0;
              var top = r && r.height ? r.top : 0;
              var w = r && r.width ? r.width : window.innerWidth;
              var h = r && r.height ? r.height : window.innerHeight;
              var x = left + w * xr;
              var y = top + h * yr;
              var el = document.elementFromPoint(x, y) || c || document.body;
              var opts = {bubbles: true, cancelable: true, clientX: x, clientY: y,
                          view: window, button: 0, buttons: 1};
              ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click'].forEach(function (t) {
                try {
                  var ev = (t.indexOf('pointer') === 0 && window.PointerEvent)
                    ? new PointerEvent(t, opts) : new MouseEvent(t, opts);
                  el.dispatchEvent(ev);
                } catch (e) {}
              });
              try { if (el.focus) { el.focus(); } } catch (e) {}
              return {x: x, y: y};
            }""",
            [float(x_ratio), float(y_ratio)],
        )
        self.advance(int(frames))


    def scroll(self, selector, direction):
        self.frame.locator(selector).evaluate("(element,direction) => element.scrollBy({top:(direction==='up'?-1:1)*Math.max(180,element.clientHeight*0.75),behavior:'instant'})", direction)

    def display(self, data):
        self.page.evaluate('data => window.__JEV_DISPLAY__(data)', data)

    def screenshot(self):
        # Capture the game frame only. The session page reserves a 360px status
        # sidebar, so a full-page shot spends a fifth of the pixels on text the
        # model already receives as `world` and squeezes the actual game.
        try:
            return self.frame.locator('body').screenshot(type='png', timeout=15000)
        except Exception:
            return self.page.screenshot(type='png')

    def thumb(self, width: int = 160):
        """Cheap frame fingerprint.

        `screenshot()` returns a full 1920x1080 PNG, and the change detector only
        needs to know whether the picture moved. Encoding that PNG twice per step
        (once in observe, once in act, for the before/after pair) was the single
        largest CPU cost: at 26 concurrent workers the box sat at 94% CPU and each
        step stretched from 16 s to 48 s, so throughput fell instead of rising.
        Playwright can encode a downscaled JPEG in the browser process, which
        keeps the comparison just as reliable (a moved sprite survives 160 px)
        and costs a fraction of the bytes.
        """
        try:
            frame = self.frame.locator('body')
            return frame.screenshot(type='jpeg', quality=45, timeout=15000,
                                    scale='css', animations='disabled')
        except TypeError:
            try:
                return self.frame.locator('body').screenshot(type='jpeg', quality=45, timeout=15000)
            except Exception:
                return b''
        except Exception:
            return b''

    def close(self):
        video = self.page.video if self.page is not None and self.record else None
        try:
            if self.context: self.context.close()
            if video: video.save_as(self.artifacts / 'full-run.webm')
        finally:
            if self.browser: self.browser.close()
            if self._playwright: self._playwright.stop()
            if self.server: self.server.shutdown(); self.server.server_close()
            if self._server_thread: self._server_thread.join(timeout=2)
