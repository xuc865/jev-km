/* Seeded browser boot with an optional deterministic step clock. No game policy or game state writes. */
(() => {
  'use strict';
  const boot = window.__JEV_BOOT__ || {};
  if (!location.pathname.startsWith('/map/') && !location.pathname.startsWith('/fixture/')) return;
  const nativeRAF = window.requestAnimationFrame.bind(window);
  const nativeSetTimeout = window.setTimeout.bind(window);
  let randomSeed = Number(boot.seed ?? 42) >>> 0;
  Math.random = () => {
    randomSeed = (randomSeed + 0x6D2B79F5) | 0;
    let n = Math.imul(randomSeed ^ (randomSeed >>> 15), 1 | randomSeed);
    n ^= n + Math.imul(n ^ (n >>> 7), 61 | n);
    return ((n ^ (n >>> 14)) >>> 0) / 4294967296;
  };
  if (boot.time_mode !== 'step') return;
  const NativeDate = Date, epoch = 1704067200000;
  let now = 0, nextId = 1, advancing = false;
  const raf = new Map(), timers = new Map();
  function VirtualDate(...args) {
    if (!new.target) return new NativeDate(epoch + now).toString();
    return args.length ? new NativeDate(...args) : new NativeDate(epoch + now);
  }
  Object.setPrototypeOf(VirtualDate, NativeDate);
  VirtualDate.prototype = NativeDate.prototype;
  VirtualDate.now = () => epoch + now;
  window.Date = VirtualDate;
  Object.defineProperty(performance, 'now', {configurable: true, value: () => now});
  window.requestAnimationFrame = fn => { const id = nextId++; raf.set(id, fn); return id; };
  window.cancelAnimationFrame = id => raf.delete(id);
  function schedule(fn, delay, repeat, args) {
    if (typeof fn !== 'function') throw new TypeError('Runtime clock requires function timers');
    const id = nextId++, ms = Math.max(1, Number(delay) || 0);
    timers.set(id, {fn, args, due: now + ms, repeat: repeat ? ms : 0});
    return id;
  }
  window.setTimeout = (fn, delay, ...args) => schedule(fn, delay, false, args);
  window.setInterval = (fn, delay, ...args) => schedule(fn, delay, true, args);
  window.clearTimeout = window.clearInterval = id => timers.delete(id);
  async function advanceFrames(count, paced = false) {
    if (advancing) throw new Error('Concurrent frame stepping is not allowed');
    if (!Number.isInteger(count) || count < 0 || count > 3600) throw new RangeError('Invalid frame count');
    advancing = true;
    try {
      for (let frame = 0; frame < count; frame++) {
        now += Number(boot.dt_ms || 1000 / 60);
        for (let calls = 0; ; calls++) {
          const due = [...timers.entries()].filter(([, timer]) => timer.due <= now)
            .sort((a, b) => a[1].due - b[1].due || a[0] - b[0])[0];
          if (!due) break;
          if (calls >= 1000) throw new Error('Timer loop exceeded frame budget');
          const [id, timer] = due;
          if (timer.repeat) timer.due += timer.repeat; else timers.delete(id);
          timer.fn(...timer.args);
          await Promise.resolve();
        }
        const callbacks = [...raf.values()]; raf.clear();
        for (const callback of callbacks) callback(now);
        await Promise.resolve();
        if (paced) await new Promise(resolve => nativeRAF(resolve));
        else await new Promise(resolve => nativeSetTimeout(resolve,0));
      }
      return {now_ms: now, frames: count};
    } finally { advancing = false; }
  }
  Object.defineProperty(window, '__JEV_CLOCK__', {value: Object.freeze({
    advanceFrames, now: () => now,
    info: () => ({now_ms: now, raf_callbacks: raf.size, timers: timers.size})
  }), writable: false, configurable: false});
})();
