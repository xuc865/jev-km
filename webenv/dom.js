/* Shared public DOM observation; no game rules. */
(() => {
let actionCounter=0;
function domSnapshot(doc) {
  const win = doc.defaultView;
  // Declarative accessibility annotations describe existing controls only.
  // Input handlers and game state stay owned by the original page.
  for (const rule of win.__JEV_BOOT__?.dom_semantics || []) {
    for (const el of doc.querySelectorAll(rule.selector)) {
      if (rule.role) el.setAttribute('role',rule.role);
      if (rule.label) el.setAttribute('aria-label',rule.label);
    }
  }
  const visible = (el) => {
    const r = el.getBoundingClientRect(), st = win.getComputedStyle(el);
    return r.width >= 4 && r.height >= 4 && r.bottom > 0 && r.right > 0 &&
      r.top < win.innerHeight && r.left < win.innerWidth &&
      st.display !== 'none' && st.visibility !== 'hidden' && st.opacity !== '0';
  };
  const fields={};
  for (const [name,rule] of Object.entries(win.__JEV_BOOT__?.dom_fields || {})) {
    const el=[...doc.querySelectorAll(rule.selector)].find(visible);
    if (!el) continue;
    let raw=String(rule.attribute ? el.getAttribute(rule.attribute) ?? '' : el.innerText || el.textContent || '').trim();
    if (rule.pattern) {
      const match=raw.match(new RegExp(rule.pattern,rule.flags || ''));
      if (!match) continue;
      raw=match[1] ?? match[0];
    }
    if (!raw.length) continue;
    const value=rule.type==='number' ? Number(raw) : raw;
    if (rule.type==='number' && !Number.isFinite(value)) continue;
    fields[name]=value;
  }
  const nodes = doc.querySelectorAll('button,[role="button"],a[href],[onclick],[tabindex],[class*="cursor-"],[draggable="true"]');
  const candidates = [];
  for (const el of nodes) {
    if (!visible(el) || el.disabled || el.getAttribute('aria-disabled') === 'true') continue;
    if (win.getComputedStyle(el).pointerEvents === 'none' || win.getComputedStyle(el).cursor === 'not-allowed') continue;
    if (!el.matches('button,[role="button"],a[href],[onclick],[tabindex]') && !['pointer','crosshair'].includes(win.getComputedStyle(el).cursor)) continue;
    if (el.tagName === 'A' && el.href && new URL(el.href,doc.URL).origin !== new URL(doc.URL).origin) continue;
    const r = el.getBoundingClientRect();
    const x = Math.max(1, Math.min(win.innerWidth - 1, r.left + r.width / 2));
    const y = Math.max(1, Math.min(win.innerHeight - 1, r.top + r.height / 2));
    const hit = doc.elementFromPoint(x, y);
    if (hit && hit !== el && !el.contains(hit)) continue;
    let label = (el.getAttribute('aria-label') || [el.innerText || el.textContent || '',
      el.getAttribute('title') || ''].join(' ')).replace(/\s+/g, ' ').trim().slice(0, 700);
    if (label.length<=8) {
      const alternatives=[...el.querySelectorAll('img[alt],svg title')]
        .map(node=>(node.getAttribute('alt')||node.textContent||'').trim()).filter(Boolean);
      label=[...new Set([label,...alternatives])].filter(Boolean).join(' ').slice(0,700);
    }
    let context='';
    const needsContext=label.length<=8 && (!/[\p{L}]/u.test(label) || /^[A-Za-z]$/.test(label));
    for (let parent=el.parentElement,depth=0;needsContext && parent && depth<3;parent=parent.parentElement,depth++) {
      const text=(parent.innerText||'').replace(/\s+/g,' ').trim();
      if (text.length>360) break;
      if (text.length>label.length) {context=text;break;}
    }
    if (!/[\p{L}\p{N}]/u.test(label)) {
      label=context && context.length<=260 ? context : '';
    } else if (label.length<=8 && context.length>label.length) {
      label=label+' | '+context;
    }
    if (!label) continue;
    if (!el.dataset.jgAction) el.dataset.jgAction = 'jg_' + (++actionCounter);
    candidates.push({id:el.dataset.jgAction, label,
      selector:'[data-jg-action="' + el.dataset.jgAction + '"]'});
  }
  const text = (doc.body?.innerText || '').slice(0, 22000);
  const headings = [...doc.querySelectorAll('h1,h2,h3,[role="heading"]')]
    .filter(visible).map(el => (el.innerText || '').trim()).slice(0, 40);
  const scrollCandidates = [];
  for (const el of doc.querySelectorAll('body *')) {
    if (!visible(el) || el.clientHeight < 80 || el.scrollHeight <= el.clientHeight + 12) continue;
    if (!/auto|scroll/.test(win.getComputedStyle(el).overflowY)) continue;
    const r=el.getBoundingClientRect();
    const x=Math.max(1,Math.min(win.innerWidth-1,r.left+r.width/2));
    const y=Math.max(1,Math.min(win.innerHeight-1,r.top+Math.min(r.height,win.innerHeight-r.top)/2));
    const hit=doc.elementFromPoint(x,y);
    if (hit && !el.contains(hit)) continue;
    if (!el.dataset.jgAction) el.dataset.jgAction='jg_' + (++actionCounter);
    const selector='[data-jg-action="' + el.dataset.jgAction + '"]';
    const region=(el.innerText||'').replace(/\s+/g,' ').trim().slice(0,45);
    if (el.scrollTop + el.clientHeight < el.scrollHeight-5) scrollCandidates.push({
      id:el.dataset.jgAction+'_down',label:'向下滚动，查看下方控件：'+region,selector,kind:'scroll',direction:'down'});
    if (el.scrollTop > 5) scrollCandidates.push({
      id:el.dataset.jgAction+'_up',label:'向上滚动，查看上方控件：'+region,selector,kind:'scroll',direction:'up'});
  }
  const available=[...scrollCandidates.slice(0,4),...candidates].slice(0,31);
  const body = JSON.stringify([text, headings, available, fields]);
  let hash = 2166136261;
  for (let i=0;i<body.length;i++) hash = Math.imul(hash ^ body.charCodeAt(i), 16777619);
  return {text, headings, fields, candidates:available, signature:String(hash >>> 0), url:doc.URL};
}
Object.defineProperty(window,'__JEV_DOM__',{value:Object.freeze({snapshot:domSnapshot}),writable:false});
})();
