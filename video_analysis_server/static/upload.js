/* Resumable chunked upload + job management UI.
 *
 * The file is never read into memory: File.slice() hands one chunk at a time to
 * fetch(), so a 35 GB video costs one chunk of RAM. If the network drops, the
 * loop retries the same chunk; if the browser is closed, /api/uploads/init with
 * the same filename+size returns the existing upload_id and the list of chunks
 * the server already has, and the loop picks up from there.
 */
'use strict';

const $ = (id) => document.getElementById(id);
/* Every request gets a deadline.
 *
 * fetch() has NO timeout of its own. When a phone hands off between cells the
 * TCP connection dies without an RST ever reaching us, and the promise then
 * stays pending forever - the upload loop simply stops, with no error, no
 * retry, and nothing in the server log. That is exactly how a 26%-complete
 * 29.6 GB upload got stuck: the server never saw another byte, and the page
 * sat waiting on a socket that was never coming back.
 */
const API_TIMEOUT = 60000;

const api = async (url, opts) => {
  const { timeout = API_TIMEOUT, ...rest } = opts || {};
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeout);
  try {
    const res = await fetch(url, Object.assign({}, rest, { signal: ctl.signal }));
    const text = await res.text();
    let body = null;
    try { body = text ? JSON.parse(text) : null; } catch (_) { body = { error: text }; }
    if (!res.ok) {
      const err = new Error((body && (body.detail || body.error))
        ? JSON.stringify(body.detail || body.error) : res.statusText);
      err.status = res.status;
      throw err;
    }
    return body;
  } catch (err) {
    if (err && err.name === 'AbortError') {
      throw new Error(`서버 응답 없음 (${Math.round(timeout / 1000)}초 초과)`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
};

/** Resolve to false instead of hanging forever. */
function withTimeout(promise, ms) {
  return new Promise((resolve) => {
    let settled = false;
    const t = setTimeout(() => { if (!settled) { settled = true; resolve(false); } }, ms);
    Promise.resolve(promise).then(
      () => { if (!settled) { settled = true; clearTimeout(t); resolve(true); } },
      () => { if (!settled) { settled = true; clearTimeout(t); resolve(false); } });
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Can the browser still read this file? Android hands out a cached copy of a
 *  picked file, and when that copy is refreshed or evicted every slice of it
 *  becomes unreadable - the upload then fails with a generic network error that
 *  no amount of retrying will fix. */
async function fileReadable(file) {
  try {
    await file.slice(0, 1).arrayBuffer();
    return true;
  } catch (_) {
    return false;
  }
}

/* Send a diagnostic line to the server.
 *
 * A phone browser's console cannot be read from here, so anything that goes
 * wrong inside the page is invisible unless the page says so out loud. This is
 * strictly best-effort: it must never throw, never block, and never be the
 * reason an upload fails.
 */
function report(event, detail, uploadId) {
  try {
    fetch('/api/client-log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        event,
        detail: String(detail === undefined ? '' : detail).slice(0, 400),
        upload_id: uploadId || null,
        version: typeof CLIENT_VERSION === 'string' ? CLIENT_VERSION : null,
      }),
      keepalive: true,        // still goes out if the page is being suspended
    }).catch(() => {});
  } catch (_) { /* diagnostics never break the upload */ }
}

const fmtBytes = (n) => {
  if (n === null || n === undefined) return '-';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0; let v = Number(n);
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i += 1; }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${u[i]}`;
};
const fmtTime = (s) => {
  if (s === null || s === undefined) return '-';
  const t = Math.max(0, Math.round(s));
  return `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}`;
};

/* ------------------------------------------------------------------ upload */

/* ------------------------------------------------------------------ upload */

/* A phone screen goes dark after a few minutes and the browser then freezes or
 * kills the page, which used to end a multi-hour upload for good. Three things
 * keep a 35 GB upload alive now:
 *
 *   1. a screen Wake Lock while an upload is running, so the screen does not
 *      turn itself off in the first place (re-acquired every time the page
 *      comes back, because the browser drops it whenever the page is hidden);
 *   2. retries that do NOT burn attempts while the page is hidden or offline -
 *      the loop parks until the page is visible and the network is back, then
 *      re-syncs with the server and carries on;
 *   3. the upload id in localStorage, so even a full reload can continue from
 *      wherever the server actually got to.
 */

/* A chunk goes out over XMLHttpRequest, not fetch, for one reason: xhr gives
 * upload.onprogress, so we can see whether BYTES ARE ACTUALLY MOVING. fetch
 * cannot tell a slow link from a dead one, and on a phone that difference is
 * the whole problem - a dead link has to be abandoned and retried, a slow one
 * has to be left alone. If no progress event arrives for chunkStallMs() the
 * request is aborted and retried from the same chunk; nothing is lost, because
 * the server tracks chunks individually.
 */
function chunkStallMs() {
  // a backgrounded page gets its timers and progress events throttled, so
  // judging it by the same stopwatch would abort perfectly healthy transfers
  return document.visibilityState === 'visible' ? 45000 : 180000;
}

function sendChunk(uploadId, index, blob, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    let lastMove = Date.now();
    let lastLoaded = 0;          // how many bytes made it out before the failure
    let watchdog = setInterval(() => {
      const idle = Date.now() - lastMove;
      if (idle > chunkStallMs()) {
        // leave a trace: when a phone upload misbehaves in the field, this line
        // in the console is the difference between guessing and knowing
        console.warn(`chunk ${index}: ${Math.round(idle / 1000)}초간 전송 정지 - 끊고 재시도`);
        try { xhr.abort(); } catch (_) { /* already gone */ }
      }
    }, 5000);
    const finish = (fn, arg) => {
      if (watchdog) { clearInterval(watchdog); watchdog = null; }
      fn(arg);
    };

    xhr.open('PUT', `/api/uploads/${uploadId}/chunk?index=${index}`, true);
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.upload.onprogress = (ev) => {
      lastMove = Date.now();
      lastLoaded = ev.loaded;
      if (onProgress) onProgress(ev.loaded);
    };
    xhr.onload = () => {
      lastMove = Date.now();
      if (xhr.status >= 200 && xhr.status < 300) { finish(resolve, null); return; }
      let msg = xhr.statusText || `HTTP ${xhr.status}`;
      try {
        const b = JSON.parse(xhr.responseText);
        msg = b.detail || b.error || msg;
      } catch (_) { /* not json */ }
      const err = new Error(msg);
      err.status = xhr.status;
      finish(reject, err);
    };
    const fail = (msg) => {
      const err = new Error(msg);
      // 0 means it never left the phone; a partial count means the path cut it
      err.sentBytes = lastLoaded;
      finish(reject, err);
    };
    xhr.onerror = () => fail('네트워크 연결 끊김');
    xhr.onabort = () => fail('전송이 멈춰서 끊고 다시 시도');
    xhr.ontimeout = () => fail('전송 시간 초과');
    xhr.send(blob);
  });
}

/* Bumped whenever this file changes in a way that matters. It is shown in the
 * header and sent with every init, because "is the phone even running the new
 * code?" turned out to be the single hardest question to answer during a
 * stuck upload - the browser had simply never reloaded the page.
 */
const CLIENT_VERSION = '2026-09-19c';

const MAX_RETRY = 8;             // past this the message gets louder - but it never gives up
const RESUME_KEY = 'ai-counter.upload';
const IS_MOBILE = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
// smaller chunks on a phone: whatever is in flight when the browser suspends
// the page has to be sent again, so a smaller unit wastes less
const MOBILE_CHUNK = 8 * 1024 * 1024;

let uploading = null;   // the run that currently owns the UI
let runToken = 0;       // bumped per start; an older loop sees it change and stops

/* ---- "upload then analyse straight away" preset ---- */

const PRESET_KEY = 'ai-counter.preset';

function presetChoice() {
  const el = document.querySelector('input[name="preset"]:checked');
  return el ? el.value : '';
}

function presetVideo() {
  const el = $('preset-video');
  return !!(el && el.checked);
}

function describePreset() {
  const hint = $('preset-hint');
  if (!hint) return;
  const type = presetChoice();
  hint.textContent = type
    ? `업로드가 끝나면 ${type === 'person' ? '사람' : '차량'} 분석을 바로 대기열에 넣습니다`
      + `${presetVideo() ? ' (결과 영상 포함)' : ''}.`
    : '업로드가 끝나면 분석 종류를 물어봅니다.';
}

function loadPreset() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(PRESET_KEY) || 'null'); } catch (_) { /* ignore */ }
  if (saved) {
    const radio = document.querySelector(`input[name="preset"][value="${saved.type || ''}"]`);
    if (radio) radio.checked = true;
    if ($('preset-video')) $('preset-video').checked = !!saved.video;
  }
  describePreset();
}

function savePreset() {
  try {
    localStorage.setItem(PRESET_KEY,
      JSON.stringify({ type: presetChoice(), video: presetVideo() }));
  } catch (_) { /* private mode */ }
  describePreset();
}

document.querySelectorAll('input[name="preset"]').forEach(
  (el) => el.addEventListener('change', savePreset));
if ($('preset-video')) $('preset-video').addEventListener('change', savePreset);
loadPreset();

/* ---- keeping the screen awake ---- */

let wakeLock = null;
let wakeMode = 'none';
let noSleepVideo = null;

function makeNoSleepVideo() {
  // Fallback for browsers without the Wake Lock API (iOS before 16.4): a muted
  // looping inline video counts as playback and holds the screen on.
  if (noSleepVideo) return noSleepVideo;
  const v = document.createElement('video');
  v.setAttribute('playsinline', '');
  v.setAttribute('muted', '');
  v.setAttribute('loop', '');
  v.muted = true;
  v.style.cssText = 'position:fixed;width:1px;height:1px;opacity:0;pointer-events:none';
  // 1-frame silent mp4, inline so nothing is fetched from the network
  v.src = 'data:video/mp4;base64,AAAAIGZ0eXBtcDQyAAACAG1wNDJpc29taXNvMmF2YzEAAAAIZnJlZQAAAr1tZGF0AAACrgYF//+q3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE0MiByMjQ3OSBkZDc5YTYxIC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMtMjAxNCAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTEgZGVibG9jaz0xOjA6MCBhbmFseXNlPTB4MToweDExMSBtZT1oZXggc3VibWU9MiBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVkX3JlZj0wIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MCA4eDhkY3Q9MCBjcW09MCBkZWFkem9uZT0yMSwxMSBmYXN0X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0wIHRocmVhZHM9NiBsb29rYWhlYWRfdGhyZWFkcz0xIHNsaWNlZF90aHJlYWRzPTAgbnI9MCBkZWNpbWF0ZT0xIGludGVybGFjZWQ9MCBibHVyYXlfY29tcGF0PTAgY29uc3RyYWluZWRfaW50cmE9MCBiZnJhbWVzPTMgYl9weXJhbWlkPTIgYl9hZGFwdD0xIGJfYmlhcz0wIGRpcmVjdD0xIHdlaWdodGI9MSBvcGVuX2dvcD0wIHdlaWdodHA9MSBrZXlpbnQ9MjUwIGtleWludF9taW49MjUgc2NlbmVjdXQ9NDAgaW50cmFfcmVmcmVzaD0wIHJjX2xvb2thaGVhZD0xMCByYz1jcmYgbWJ0cmVlPTEgY3JmPTIzLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAAwZYiEAD//8m+P5OXfBeLGOfKE3xQxHvXSBpSnhdiadyBhJAAAAwAAAwAAFgn0I7DkqgAAAAlBmiRsQ3/+p4QAAAAJQZ5CeIV/AAAJAAAACQGeYXRBXwAACQAAAAkBnmNqQV8AAAkAAAANQZpoSahBaJlMCG///qeEAAAACUGehkURLCv/AAAJAAAACQGepXRBXwAACQAAAAkBnqdqQV8AAAkAAAANQZqsSahBbJlMCG///qeEAAAACUGeykUVLCv/AAAJAAAACQGe6XRBXwAACQAAAAkBnutqQV8AAAkAAAANQZrwSahBbJlMCG///qeEAAAACUGfDkUVLCv/AAAJAAAACQGfLXRBXwAACQAAAAkBny9qQV8AAAkAAAANQZs0SahBbJlMCG///qeEAAAACUGfUkUVLCv/AAAJAAAACQGfcXRBXwAACQAAAAkBn3NqQV8AAAkAAAANQZt4SahBbJlMCG///qeEAAAACUGflkUVLCv/AAAJAAAACQGftXRBXwAACQAAAAkBn7dqQV8AAAkAAAANQZu8SahBbJlMCG///qeEAAAACUGf2kUVLCv/AAAJAAAACQGf+XRBXwAACQAAAAkBn/tqQV8AAAkAAAANQZvgSahBbJlMCG///qeE';
  document.body.appendChild(v);
  noSleepVideo = v;
  return v;
}

async function keepAwake() {
  if (document.visibilityState !== 'visible') return;
  try {
    if ('wakeLock' in navigator) {
      wakeLock = await navigator.wakeLock.request('screen');
      wakeMode = 'lock';
      wakeLock.addEventListener('release', () => { wakeLock = null; });
      return;
    }
  } catch (_) { /* denied (low battery, permissions) - fall through */ }
  try {
    await makeNoSleepVideo().play();
    wakeMode = 'video';
  } catch (_) {
    wakeMode = 'none';
  }
}

function releaseAwake() {
  try { if (wakeLock) wakeLock.release(); } catch (_) { /* already gone */ }
  wakeLock = null;
  if (noSleepVideo) { noSleepVideo.pause(); }
  wakeMode = 'none';
}

const wakeLabel = () => {
  if (wakeMode === 'lock') return '화면 꺼짐 방지 켜짐';
  if (wakeMode === 'video') return '화면 꺼짐 방지 켜짐 (대체 방식)';
  return window.isSecureContext
    ? '화면 꺼짐 방지 불가 — 꺼져도 자동으로 이어집니다'
    : '⚠ http 접속이라 화면 꺼짐을 막을 수 없습니다 — https 주소로 접속하세요';
};

/* ---- parking while the page is hidden or the network is down ---- */

function onceVisible() {
  if (document.visibilityState === 'visible') return Promise.resolve();
  return new Promise((res) => {
    const h = () => {
      if (document.visibilityState !== 'visible') return;
      document.removeEventListener('visibilitychange', h);
      res();
    };
    document.addEventListener('visibilitychange', h);
  });
}

function onceOnline() {
  if (navigator.onLine) return Promise.resolve();
  return new Promise((res) => {
    const h = () => { window.removeEventListener('online', h); res(); };
    window.addEventListener('online', h);
  });
}

/** Park (without spending a retry) until the page can actually make progress. */
async function waitResumable() {
  // Park ONLY when the network is genuinely gone.
  //
  // A hidden page is NOT that. Waiting for the screen to come back meant a
  // 30 GB upload stopped dead the moment the phone blanked and sat there for
  // hours - which is the exact failure this is supposed to prevent. Android
  // keeps a background XHR running for a good while, so the right move is to
  // keep pushing: if the browser really does suspend us the chunk fails, and
  // the retry path picks it up when the page wakes. Trying and failing is
  // strictly better than refusing to try.
  if (navigator.onLine) return false;
  // navigator.onLine is unreliable on phones: it can stay false long after the
  // radio is back, and the 'online' event then never fires. Give it 30 seconds
  // and try the network anyway.
  await withTimeout(onceOnline(), 30000);
  await withTimeout(keepAwake(), 3000);   // the browser dropped the lock while hidden
  return true;
}

/* ---- the upload itself ---- */

function rememberUpload(info) {
  try {
    if (info) localStorage.setItem(RESUME_KEY, JSON.stringify(info));
    else localStorage.removeItem(RESUME_KEY);
  } catch (_) { /* private mode - resume-after-reload just will not be offered */ }
}

async function startUpload(file) {
  // Picking a file again while an earlier run is still alive used to leave two
  // loops sharing one `uploading` global: whichever finished first set it to
  // null and the other died on `uploading.cancel`. Each run now owns its own
  // state object and checks a token, so the older one retires quietly.
  const token = ++runToken;
  if (uploading) uploading.cancel = true;
  const me = { file, uploadId: null, cancel: false, token };
  uploading = me;
  const stale = () => me.cancel || runToken !== token;

  const body = { filename: file.name, file_size: file.size,
                 client_version: CLIENT_VERSION };
  if (IS_MOBILE) body.chunk_size = MOBILE_CHUNK;
  const init = await api('/api/uploads/init', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (stale()) {
    console.warn('upload run superseded before it started');
    report('superseded', `${file.name}`);
    return;
  }
  me.uploadId = init.upload_id;
  showRepick(false);
  report('init-ok', `resumed=${init.resumed} next=${init.next_chunk} `
                    + `total=${init.total_chunks} missing=${(init.missing_chunks || []).length} `
                    + `chunk=${init.chunk_size} fileSize=${file.size} `
                    + `lastMod=${file.lastModified || 0} type=${file.type || '?'}`,
         init.upload_id);
  rememberUpload({ uploadId: init.upload_id, name: file.name, size: file.size,
                   chunkSize: init.chunk_size });
  // bounded: a wake-lock request that never settles must not hold up the upload
  if (wakeMode === 'none') await withTimeout(keepAwake(), 3000);

  const total = init.total_chunks;
  const chunkSize = init.chunk_size;
  let missing = new Set(init.missing_chunks);
  let sent = init.received_bytes;
  let windowBytes = 0;
  let windowStart = performance.now();
  let speed = 0;

  const show = (extra) => renderCurrent(Object.assign({
    name: file.name, size: file.size, sent,
    pct: Math.min(100, sent / file.size * 100), speed, wake: wakeLabel(),
  }, extra));

  show({ note: init.resumed ? `${fmtBytes(sent)} 지점부터 이어서` : '시작' });

  /** Ask the server what it actually has - after a long sleep our idea of it
   *  may be stale, and the server is the authority. */
  let reportedFirst = false;
  let reportedOk = false;

  const resync = async () => {
    const st = await api(`/api/uploads/${init.upload_id}/status`);
    missing = new Set(st.missing_chunks);
    sent = st.received_bytes;
    return st;
  };

  for (let i = 0; i < total; i += 1) {
    if (stale()) { if (runToken === token) { renderCurrent(null); releaseAwake(); } return; }
    if (!missing.has(i)) continue;

    const blob = file.slice(i * chunkSize, Math.min((i + 1) * chunkSize, file.size));
    if (!reportedFirst) {
      reportedFirst = true;
      // the offset is the interesting part: past 4 GB some phones cannot read a
      // slice at all, and that failure is otherwise completely silent
      report('first-chunk', `i=${i} offset=${i * chunkSize} blobSize=${blob.size}`,
             init.upload_id);
    }
    let attempt = 0;
    for (;;) {
      if (stale()) { if (runToken === token) { renderCurrent(null); releaseAwake(); } return; }
      // hidden or offline: park, then re-sync. Costs no retry, so a screen that
      // stays dark for an hour does not end the upload.
      if (!navigator.onLine) {
        show({ note: '네트워크 끊김 — 돌아오면 자동으로 이어집니다' });
      } else if (document.visibilityState !== 'visible') {
        show({ note: '화면이 꺼진 상태에서도 계속 올리는 중' });
      }
      if (await waitResumable()) {
        await resync().catch(() => {});
        attempt = 0;
        if (!missing.has(i)) break;     // it landed after all
      }
      try {
        let painted = 0;
        await sendChunk(init.upload_id, i, blob, (loaded) => {
          // show movement WITHIN a chunk: on a slow uplink an 8 MB chunk takes
          // minutes, and a bar frozen that long is indistinguishable from a crash
          const now = performance.now();
          if (now - painted < 300) return;
          painted = now;
          const at = sent + loaded;
          const dt2 = (now - windowStart) / 1000;
          if (dt2 >= 1) speed = (windowBytes + loaded) / dt2;
          renderCurrent({ name: file.name, size: file.size, sent: at,
                          pct: Math.min(100, at / file.size * 100), speed,
                          wake: wakeLabel(),
                          note: `청크 ${i + 1}/${total} 전송 중 ${fmtBytes(loaded)}/${fmtBytes(blob.size)}` });
        });
        break;
      } catch (err) {
        if (!navigator.onLine) {
          // genuinely offline: wait for the radio rather than burning retries,
          // but never longer than 30 s (the 'online' event can go missing)
          await withTimeout(onceOnline(), 30000);
          continue;
        }
        // The server is the authority on what it still needs. A chunk can fail
        // because the upload is ALREADY FINISHED - another run's request landed
        // first, or a second tab completed it - and that is success, not an
        // error to show the user. Ask before counting a retry.
        const st = await api(`/api/uploads/${init.upload_id}/status`).catch(() => null);
        if (st && st.status !== 'UPLOADING') {
          renderCurrent(null);
          rememberUpload(null);
          releaseAwake();
          if (uploading === me) uploading = null;
          refresh();
          return;                       // already uploaded / verified elsewhere
        }
        if (st) {
          missing = new Set(st.missing_chunks);
          sent = st.received_bytes;
          if (!missing.has(i)) break;   // it arrived after all
        }
        attempt += 1;
        // A 4xx is the server refusing this request on its merits and retrying
        // cannot help. Everything else - dropped connection, stall, 5xx - is
        // worth another go, and we keep going rather than dumping the user back
        // to "pick the file again": on a phone the link comes and goes, and a
        // 30 GB upload has to ride that out by itself.
        console.warn(`chunk ${i} 실패(${attempt + 1}회): ${err.message}`);

        // Nothing left the device: suspect the file, not the link.
        if (err.sentBytes === 0 && !(await fileReadable(file))) {
          report('file-lost',
                 `i=${i} try=${attempt + 1} lastMod=${file.lastModified || 0} `
                 + `size=${file.size}`,
                 init.upload_id);
          rememberUpload({ uploadId: init.upload_id, name: file.name,
                           size: file.size, chunkSize: init.chunk_size });
          show({ error: '휴대폰이 이 영상 파일을 더 이상 읽지 못합니다 '
                        + '(파일이 갱신되었거나 접근 권한이 만료됨). '
                        + '아래 "파일 다시 선택"을 누르면 올라간 부분 다음부터 이어집니다.',
                 note: `${Math.round(sent / file.size * 100)}% 까지 저장됨` });
          showRepick(true);
          if (uploading === me) uploading = null;
          releaseAwake();
          refresh();
          return;
        }

        if (attempt < 3 || attempt % 5 === 0) {
          report('chunk-fail',
                 `i=${i} try=${attempt + 1} name=${err.name || '?'} `
                 + `status=${err.status || 0} sent=${err.sentBytes === undefined ? '?' : err.sentBytes}`
                 + `/${blob.size} msg=${err.message}`,
                 init.upload_id);
        }
        const fatal = err.status >= 400 && err.status < 500
                      && err.status !== 408 && err.status !== 429;
        if (fatal) {
          show({ error: `청크 ${i} 전송 실패: ${err.message} — 같은 파일을 다시 선택하면 이어집니다` });
          if (uploading === me) uploading = null;
          releaseAwake();                 // never leave the screen pinned on
          return;
        }
        const wait = Math.min(1000 * 2 ** Math.min(attempt, 5), 30000);
        show(attempt >= MAX_RETRY
          ? { note: `연결이 불안정합니다 — 계속 재시도 중 (${attempt}회, 청크 ${i + 1}/${total})`,
              error: err.message }
          : { note: `재시도 ${attempt}/${MAX_RETRY} (청크 ${i + 1}/${total}) — ${Math.round(wait / 1000)}초 후` });
        await sleep(wait);
      }
    }

    if (!reportedOk) {
      reportedOk = true;
      report('chunk-ok', `i=${i}`, init.upload_id);
    }
    missing.delete(i);
    sent += blob.size;
    windowBytes += blob.size;
    const dt = (performance.now() - windowStart) / 1000;
    if (dt >= 1) { speed = windowBytes / dt; windowBytes = 0; windowStart = performance.now(); }
    const left = speed > 0 ? (file.size - sent) / speed : 0;
    show({ note: `청크 ${i + 1}/${total}${left ? ` · 남은 시간 약 ${fmtEta(left)}` : ''}` });
  }

  show({ sent: file.size, pct: 100, speed: 0, note: '영상 검증 중...' });
  try {
    // ffprobe on a 30 GB file is not a 60-second job
    await api(`/api/uploads/${init.upload_id}/complete`,
              { method: 'POST', timeout: 900000 });
    rememberUpload(null);
    // Clear `uploading` BEFORE dropping the lock: the visibilitychange handler
    // re-acquires whenever an upload is in flight, and doing it the other way
    // round let a screen-on event take the lock straight back.
    if (uploading === me) uploading = null;
    releaseAwake();

    const type = presetChoice();
    if (type) {
      // the whole point of the preset: no questions, straight into the queue
      renderCurrent({ name: file.name, size: file.size, sent: file.size, pct: 100,
                      speed: 0, note: '대기열에 등록하는 중...' });
      try {
        const job = await api('/api/jobs', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ upload_id: init.upload_id, analysis_type: type,
                                 start: true, create_video: presetVideo() }),
        });
        const pos = job.queue_position;
        renderCurrent({ name: file.name, size: file.size, sent: file.size, pct: 100,
                        speed: 0,
                        note: `${type === 'person' ? '사람' : '차량'} 분석 `
                              + (pos && pos > 0 ? `대기 ${pos}번째로 등록됨` : '시작됨') });
        setTimeout(() => renderCurrent(null), 4000);
      } catch (e) {
        show({ sent: file.size, pct: 100, speed: 0,
               error: `분석 등록 실패: ${e.message}` });
      }
    } else {
      renderCurrent(null);
      openAnalysisModal(init.upload_id, file.name);
    }
  } catch (err) {
    show({ sent: file.size, pct: 100, speed: 0, error: `검증 실패: ${err.message}` });
    if (uploading === me) uploading = null;
    releaseAwake();
  }
  if (uploading === me) uploading = null;
  refresh();
}

function fmtEta(sec) {
  if (!Number.isFinite(sec) || sec <= 0) return '';
  const h = Math.floor(sec / 3600);
  const m = Math.round((sec % 3600) / 60);
  return h ? `${h}시간 ${m}분` : `${Math.max(1, m)}분`;
}

function renderCurrent(s) {
  const box = $('current');
  if (!s) { box.innerHTML = ''; return; }
  box.innerHTML = `
    <div class="item" style="margin-top:14px">
      <div class="row"><span class="name">${escapeHtml(s.name)}</span>
        <span class="badge UPLOADING">UPLOADING</span></div>
      <div class="bar"><i style="width:${Math.min(100, s.pct).toFixed(1)}%"></i></div>
      <div class="meta" style="margin-top:8px">
        ${s.pct.toFixed(1)}% &middot; ${fmtBytes(s.sent)} / ${fmtBytes(s.size)}
        ${s.speed ? ` &middot; ${fmtBytes(s.speed)}/s` : ''} ${s.note ? ` &middot; ${escapeHtml(s.note)}` : ''}
      </div>
      ${s.wake ? `<div class="meta wake">${escapeHtml(s.wake)}</div>` : ''}
      ${s.error ? `<div class="err">${escapeHtml(s.error)}</div>` : ''}
    </div>`;
}

// the browser releases a screen wake lock whenever the page is hidden, so it
// has to be taken again every time the user comes back
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && uploading && !uploading.cancel) keepAwake();
});

// an unhandled rejection used to kill the upload silently
window.addEventListener('unhandledrejection', (ev) => {
  if (!uploading) return;
  console.error('upload error', ev.reason);
});

$('file').addEventListener('change', (ev) => {
  const file = ev.target.files && ev.target.files[0];
  ev.target.value = '';
  if (!file) return;
  // Taken here, synchronously, while the user gesture is still live: the
  // fallback is video playback and a browser refuses to start playback outside
  // a gesture. Doing it after the first await silently failed.
  keepAwake();
  startUpload(file).catch((e) => renderCurrent({
    name: file.name, size: file.size, sent: 0, pct: 0, speed: 0, error: e.message }));
});

/** After a reload the File object is gone - browsers never hand it back - but
 *  the server still has every byte it received, so re-picking the same file
 *  continues instead of restarting. */
async function offerResume() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(RESUME_KEY) || 'null'); } catch (_) { return; }
  if (!saved) return;
  try {
    const st = await api(`/api/uploads/${saved.uploadId}/status`);
    if (st.status !== 'UPLOADING') { rememberUpload(null); return; }
    $('current').innerHTML = `
      <div class="item resume" style="margin-top:14px">
        <div class="row"><span class="name">${escapeHtml(saved.name)}</span>
          <span class="badge UPLOADING">이어서 올리기</span></div>
        <div class="bar"><i style="width:${st.upload_progress}%"></i></div>
        <div class="meta" style="margin-top:8px">
          ${st.upload_progress.toFixed(1)}% &middot; ${fmtBytes(st.received_bytes)} / ${fmtBytes(saved.size)}
          까지 서버에 저장돼 있습니다.
        </div>
        <div class="actions">
          <button class="primary" id="resume-pick">같은 파일 선택해서 이어하기</button>
          <button class="danger" id="resume-drop">취소</button>
        </div>
      </div>`;
    $('resume-pick').addEventListener('click', () => $('file').click());
    $('resume-drop').addEventListener('click', async () => {
      await api(`/api/uploads/${saved.uploadId}`, { method: 'DELETE' }).catch(() => {});
      rememberUpload(null);
      renderCurrent(null);
      refresh();
    });
  } catch (_) {
    rememberUpload(null);
  }
}

/* -------------------------------------------------------------------- list */

function escapeHtml(s) {
  return String(s === null || s === undefined ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function cardFor(u) {
  const job = u.job;
  const status = job ? job.status : u.status;
  const bits = [`<span class="badge ${escapeHtml(u.status)}">${escapeHtml(u.status)}</span>`];
  if (job) {
    bits.push(`<span class="badge ${escapeHtml(job.analysis_type)}">${job.analysis_type.toUpperCase()} ANALYSIS</span>`);
    bits.push(`<span class="badge ${escapeHtml(job.status)}">${escapeHtml(job.status)}</span>`);
    if (job.status === 'QUEUED') {
      bits.push(`<span class="badge">대기 ${job.queue_position || '-'}번째</span>`);
    }
    if (job.create_video) bits.push('<span class="badge">+ 결과 영상</span>');
  }

  let bars = '';
  if (u.status === 'UPLOADING') {
    bars += `<div class="bar"><i style="width:${u.upload_progress}%"></i></div>
             <div class="meta" style="margin-top:6px">upload ${u.upload_progress.toFixed(1)}%
             &middot; ${fmtBytes(u.received_bytes)} / ${fmtBytes(u.file_size)}</div>`;
  }
  if (job && (job.status === 'PROCESSING' || job.status === 'RESULT_VERIFYING')) {
    bars += `<div class="bar analysis"><i style="width:${job.analysis_progress}%"></i></div>
             <div class="meta" style="margin-top:6px">analysis ${job.analysis_progress.toFixed(1)}%
             &middot; frame ${job.current_frame} / ${job.total_frames || '?'}</div>`;
  }

  const info = u.video_info || {};
  const meta = [fmtBytes(u.file_size)];
  if (info.width) meta.push(`${info.width}x${info.height}`);
  if (info.fps) meta.push(`${info.fps} fps`);
  if (info.duration_seconds) meta.push(fmtTime(info.duration_seconds));

  const actions = [];
  if (u.status === 'UPLOADED' && !job) {
    actions.push(`<button class="primary" data-act="configure" data-id="${u.upload_id}" data-name="${escapeHtml(u.filename)}">Choose analysis</button>`);
  }
  if (job && job.status === 'WAITING_CONFIG') {
    actions.push(`<button class="primary" data-act="configure" data-id="${u.upload_id}" data-name="${escapeHtml(u.filename)}" data-job="${job.job_id}" data-type="${job.analysis_type}">Start analysis</button>`);
  }
  if (job && job.status === 'COMPLETED') {
    actions.push(`<button class="primary" data-act="result" data-job="${job.job_id}">결과 대시보드</button>`);
    if (job.create_video) {
      actions.push(`<a class="btnlink" href="/api/jobs/${job.job_id}/video" target="_blank" rel="noopener">결과 영상</a>`);
    }
  }
  if (job && job.status === 'FAILED') {
    actions.push(`<button data-act="retry" data-job="${job.job_id}">Retry</button>`);
  }
  if (u.status !== 'UPLOADING' && !(job && (job.status === 'QUEUED' || job.status === 'PROCESSING'))) {
    actions.push(`<button class="danger" data-act="delete" data-id="${u.upload_id}">Delete</button>`);
  }

  const err = (job && job.error_message) || u.error_message;
  return `<div class="item">
      <div class="row"><span class="name">${escapeHtml(u.filename)}</span>${bits.join('')}</div>
      <div class="meta" style="margin-top:4px">${meta.join(' &middot; ')}${u.status === 'SOURCE_DELETED' ? ' &middot; source deleted' : ''}</div>
      ${bars}
      ${err ? `<div class="err">${escapeHtml(err)}</div>` : ''}
      ${actions.length ? `<div class="actions">${actions.join('')}</div>` : ''}
    </div>`;
}

async function refresh() {
  try {
    const [{ uploads }, health] = await Promise.all([api('/api/uploads'), api('/api/health')]);
    $('health').textContent = `GPU: ${health.running_job || 'idle'} · queued ${health.queued}`
      + (health.allow_result_video ? '' : ' · 결과 영상 비활성');
    $('list').innerHTML = uploads.length ? uploads.map(cardFor).join('') : '<p class="empty">Nothing uploaded yet.</p>';
  } catch (err) {
    $('health').textContent = `server unreachable: ${err.message}`;
  }
}

$('list').addEventListener('click', async (ev) => {
  const btn = ev.target.closest('button[data-act]');
  if (!btn) return;
  const { act, id, job, name, type } = btn.dataset;
  try {
    if (act === 'configure') openAnalysisModal(id, name, job, type);
    else if (act === 'result') location.href = `/result?job=${encodeURIComponent(job)}`;
    else if (act === 'retry') { await api(`/api/jobs/${job}/start`, { method: 'POST' }); refresh(); }
    else if (act === 'delete') {
      if (confirm('Delete this video and its record?')) {
        await api(`/api/uploads/${id}`, { method: 'DELETE' });
        refresh();
      }
    }
  } catch (err) { alert(err.message); }
});

setInterval(refresh, 1000);
refresh();
offerResume();

/* --------------------------------------------------------- analysis config */

const modal = $('modal');
let ctx = null;     // { uploadId, filename, jobId, type, frame, entry: [], exit: [], active }

function openModal() { modal.classList.remove('hidden'); }
function closeModal() { modal.classList.add('hidden'); ctx = null; }
$('modal-close').addEventListener('click', closeModal);

function showStep(which) {
  for (const id of ['step-type']) {
    $(id).classList.toggle('hidden', id !== which);
  }
}

function openAnalysisModal(uploadId, filename, jobId, type) {
  ctx = { uploadId, filename, jobId: jobId || null, type: type || null };
  $('modal-title').textContent = filename;
  const box = document.querySelector('#step-type .opt-video');
  if (box) box.checked = false;          // never remembered between uploads
  $('type-msg').textContent = '';
  document.querySelectorAll('#step-type button[data-type]')
    .forEach((b) => { b.disabled = false; });
  openModal();
  showStep('step-type');
}

/** Picking the type IS the whole configuration: the job is queued straight
 *  away and the single GPU worker takes it as soon as the one before it
 *  finishes. Person ROIs are drawn afterwards, on the result page. */
document.querySelectorAll('#step-type button[data-type]').forEach((btn) => {
  btn.addEventListener('click', async () => {
    const type = btn.dataset.type;
    const buttons = document.querySelectorAll('#step-type button[data-type]');
    buttons.forEach((b) => { b.disabled = true; });
    $('type-msg').textContent = '대기열에 등록하는 중...';
    try {
      const createVideo = wantsVideo('step-type');
      let job;
      if (ctx.jobId) {
        // the job row already existed (a card that was left unconfigured)
        job = await api(`/api/jobs/${ctx.jobId}/start`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ create_video: createVideo }),
        });
      } else {
        job = await api('/api/jobs', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ upload_id: ctx.uploadId, analysis_type: type,
                                 start: true, create_video: createVideo }),
        });
      }
      const pos = job.queue_position;
      $('type-msg').textContent = pos && pos > 0
        ? `대기열 ${pos}번째로 등록했습니다.`
        : '대기열에 등록했습니다.';
      await refresh();
      setTimeout(closeModal, 700);
    } catch (err) {
      buttons.forEach((b) => { b.disabled = false; });
      $('type-msg').textContent = `실패: ${err.message}`;
    }
  });
});

/** The "결과 영상 생성" box of the step that is currently on screen.
 *  Unchecked by default - rendering costs an extra decode of the whole video. */
function wantsVideo(stepId) {
  const box = document.querySelector(`#${stepId} .opt-video`);
  return !!(box && box.checked);
}



/* http:// on a LAN address is not a secure context, so navigator.wakeLock does
 * not exist there and the screen will blank mid-upload. Point the user at the
 * https port instead. */
(function insecureNotice() {
  const box = $('insecure');
  if (!box) return;
  if (window.isSecureContext) { box.classList.add('hidden'); return; }
  box.classList.remove('hidden');
  fetch('/api/health').then((r) => r.json()).then((h) => {
    const port = h.https_port;
    const link = $('https-link');
    if (!port) { link.remove(); return; }
    link.href = `https://${location.hostname}:${port}${location.pathname}`;
    link.textContent = `https://${location.hostname}:${port} 로 열기`;
  }).catch(() => {});
})();


/* Stamp the running build into the header. Without this there is no way to tell
 * a phone on the current code from one that never reloaded the page. */
(function stampVersion() {
  const header = document.querySelector('header');
  if (!header) return;
  const el = document.createElement('div');
  el.textContent = `v${CLIENT_VERSION}`;
  el.style.cssText = 'font-size:11px;opacity:.55;margin-left:auto;padding-left:10px';
  header.appendChild(el);
})();


/* Anything that escapes entirely - a thrown error, a rejected promise nobody
 * handled - would otherwise take the upload down without a trace. */
window.addEventListener('error', (ev) => {
  report('window-error', `${ev.message} @${ev.filename}:${ev.lineno}`);
});
window.addEventListener('unhandledrejection', (ev) => {
  const r = ev.reason;
  report('unhandled-rejection',
         r && r.message ? `${r.name || 'Error'}: ${r.message}` : String(r));
});
report('page-load', `${navigator.userAgent.slice(0, 120)} secure=${window.isSecureContext}`);


/* ---- connection self-test ----
 *
 * When small requests succeed and a big one always fails, the useful question
 * is "how big is too big?". This sends increasing bodies to a sink endpoint and
 * reports the first size that does not make it, both on screen and to the
 * server log. It writes nothing and touches no upload.
 */
async function runNetTest() {
  const btn = $('nettest');
  const out = $('nettest-out');
  const sizes = [64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024];
  if (btn) btn.disabled = true;
  const lines = [];
  report('nettest-start', `sizes=${sizes.map((x) => x / 1024 + 'K').join(',')}`);

  for (const size of sizes) {
    const label = size >= 1048576 ? `${size / 1048576} MB` : `${size / 1024} KB`;
    if (out) out.textContent = `${lines.join(' · ')}${lines.length ? ' · ' : ''}${label} 시험 중...`;
    const blob = new Blob([new Uint8Array(size)]);
    const t0 = performance.now();
    try {
      const res = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/net-test', true);
        xhr.timeout = 180000;
        xhr.onload = () => (xhr.status === 200
          ? resolve(JSON.parse(xhr.responseText))
          : reject(new Error(`HTTP ${xhr.status}`)));
        xhr.onerror = () => reject(new Error('연결 끊김'));
        xhr.ontimeout = () => reject(new Error('시간 초과'));
        xhr.send(blob);
      });
      const dt = (performance.now() - t0) / 1000;
      const ok = res.received === size;
      lines.push(`${label} ${ok ? 'OK' : '일부만(' + res.received + ')'} ${dt.toFixed(1)}s`);
      report('nettest', `${size} -> received=${res.received} ${dt.toFixed(1)}s`);
      if (!ok) break;
    } catch (e) {
      const dt = (performance.now() - t0) / 1000;
      lines.push(`${label} 실패(${e.message}) ${dt.toFixed(1)}s`);
      report('nettest', `${size} -> FAIL ${e.message} ${dt.toFixed(1)}s`);
      break;
    }
  }

  if (out) out.textContent = lines.join(' · ');
  if (btn) btn.disabled = false;
}

if ($('nettest')) $('nettest').addEventListener('click', runNetTest);


/* One tap back into the file picker.
 *
 * When Android drops the cached copy of a picked file the only way forward is
 * to pick it again, and burying that behind the same button the user already
 * pressed made it look like a dead end. The upload itself carries on from
 * whatever the server already has, so this costs nothing but the tap.
 */
function showRepick(on) {
  const box = $('repick');
  if (box) box.classList.toggle('hidden', !on);
}

if ($('repick-btn')) {
  $('repick-btn').addEventListener('click', () => {
    showRepick(false);
    $('file').click();
  });
}
