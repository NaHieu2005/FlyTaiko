import * as THREE from './vendor/three.module.js';

const $ = id => document.getElementById(id), viewer = $('viewer');
const backend = path => window.FLYTAIKO_BACKEND_URL ?
    new URL(path.replace(/^\/+/, ''), window.FLYTAIKO_BACKEND_URL.replace(/\/+$/, '') + '/').href : path;
async function revealPhaseAReplay() {
    if (new URLSearchParams(location.search).get('dataset') === 'playing-god-v19') return;
    if (!$('phase-a-link').hidden) return;
    try {
        const response = await fetch(backend('demos/malecns-taiko-v18-phase-a/manifest.json'),
            {method: 'HEAD', cache: 'no-cache'});
        if (response.ok) $('phase-a-link').hidden = false;
    } catch (_) { /* The background export has not published its manifest yet. */ }
}
revealPhaseAReplay();
setInterval(revealPhaseAReplay, 60000);
let manifest, game, trace, values, shown = -1, version = 0, aborter;
let observationBytes, observationPaint = -1;
let audioPromise, audioStatus = 'Đang tải nhạc…', audioPendingPlayback = false;
let brainDirty = true, lastHUD = 0, loadComplete = false;
let edgeRows = [], edgeLines, lightNodes, lightPulses, neuronContrast, neuronIntensity;
let lastBrainPaint = 0;
const scene = new THREE.Scene(), cloud = new THREE.Group(); scene.add(cloud);
const camera = new THREE.PerspectiveCamera(45, 1, 1, 2000); camera.position.z = 340;
let renderer, surface, highlights, background, normalized;
try { renderer = new THREE.WebGLRenderer({antialias: true}); renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5)); surface = renderer.domElement; }
catch (_) { surface = document.createElement('canvas'); $('brain').dataset.mode = 'canvas-projection'; }
$('brain').append(surface);
function resize() {
    const width = $('brain').clientWidth;
    if (renderer) renderer.setSize(width, 360); else { surface.width = width; surface.height = 360; }
    camera.aspect = width / 360; camera.updateProjectionMatrix(); brainDirty = true;
    if (viewer.contentDocument?.body.classList.contains('recorded-mode')) {
        viewer.style.height = Math.ceil(viewer.contentDocument.body.scrollHeight + 2) + 'px';
    }
}
resize(); window.addEventListener('resize', resize);
let drag;
surface.onpointerdown = e => { drag = [e.clientX, e.clientY]; surface.setPointerCapture(e.pointerId); };
surface.onpointerup = surface.onpointercancel = () => drag = null;
surface.onpointermove = e => {
    if (!drag) return;
    cloud.rotation.y += (e.clientX - drag[0]) * .01; cloud.rotation.x += (e.clientY - drag[1]) * .01;
    drag = [e.clientX, e.clientY]; brainDirty = true;
};
const ready = new Promise((resolve, reject) => {
    if (viewer.contentWindow?.loadRecordedReplay) return resolve(viewer.contentWindow);
    const timeout = setTimeout(() => reject(Error('Player không tải được. Hãy tải lại trang.')), 30000);
    viewer.addEventListener('load', () => {
        clearTimeout(timeout);
        if (!viewer.contentWindow.loadRecordedReplay) reject(Error('Player thiếu recorded_taiko_player.js'));
        else resolve(viewer.contentWindow);
    }, {once: true});
});
function safePath(path) {
    if (!/^demos\/(?:malecns-taiko(?:-v1[6789](?:-phase-a)?)?|playing-god-v(?:19|23)|ideoless-v23|user-v(?:23|24|25)-[a-f0-9]{12})\/[a-zA-Z0-9_.-]+$/.test(path)) throw Error('Đường dẫn demo không hợp lệ');
    return path;
}
async function bytes(path, signal, progress) {
    const url = safePath(path);
    let response;
    for (let attempt = 0; attempt < 4; attempt++) {
        try { response = await fetch(backend(url), {signal, cache: 'no-cache'}); }
        catch (error) {
            if (signal?.aborted || attempt === 3) throw error;
            await new Promise(resolve => setTimeout(resolve, 250 * (attempt + 1)));
            continue;
        }
        if (response.ok) break;
        // VS Code's forwarded port has occasionally prefixed a static GET
        // with stray bytes: the server then responds 400 or 501. Retry it.
        if (![400, 501, 502, 503, 504].includes(response.status) || attempt === 3)
            throw Error(`${path}: HTTP ${response.status}`);
        await new Promise(resolve => setTimeout(resolve, 250 * (attempt + 1)));
    }
    const total = Number(response.headers.get('content-length')), reader = response.body.getReader();
    const chunks = []; let length = 0;
    while (true) {
        const result = await reader.read(); if (result.done) break;
        chunks.push(result.value); length += result.value.length;
        if (progress) progress(length, total);
    }
    const result = new Uint8Array(length); let offset = 0;
    for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.length; }
    return result.buffer;
}
async function json(path, signal) { return JSON.parse(new TextDecoder().decode(await bytes(path, signal))); }
async function refreshReplayLibrary(selected) {
    const response = await fetch(backend('/api/replays'), {cache: 'no-store'});
    if (!response.ok) throw Error(`Replay library: HTTP ${response.status}`);
    const entries = (await response.json()).replays;
    const picker = $('library');
    picker.replaceChildren(...entries.map(item => new Option(item.label, item.dataset)));
    if (entries.some(item => item.dataset === selected)) picker.value = selected;
    picker.disabled = !entries.length;
    picker.onchange = () => { location.href = `malecns-taiko.html?dataset=${encodeURIComponent(picker.value)}`; };
}
async function waitForUserReplay(dataset) {
    const jobId = dataset.replace(/^user-v(?:23|24|25)-/, '');
    document.querySelector('h1').textContent = 'Đang tạo replay…';
    $('checkpoint-label').textContent = `${dataset.slice(5, 8).toUpperCase()} · ĐANG SUY LUẬN`;
    while (true) {
        const response = await fetch(backend(`/api/replays/${jobId}`), {cache: 'no-store'});
        if (!response.ok) throw Error(`Không đọc được trạng thái job ${jobId}: HTTP ${response.status}`);
        const job = await response.json();
        if (job.status === 'failed') throw Error(`Job ${jobId} thất bại: ${job.error || 'xem worker.log'}`);
        if (job.status === 'complete') {
            refreshReplayLibrary(dataset).catch(() => {});
            return json(`demos/${dataset}/manifest.json`);
        }
        const progress = job.progress_frames && job.total_frames_approx ?
            ` · ${job.progress_frames.toLocaleString()} / ~${job.total_frames_approx.toLocaleString()} frame` : '';
        $('status').textContent = `${job.chart?.title || 'Replay'} [${job.mods || 'NM'}]: ${job.status}${progress}. Trang sẽ tự mở replay khi hoàn tất.`;
        await new Promise(resolve => setTimeout(resolve, 5000));
    }
}
async function verify(buffer, expected) {
    if (!crypto.subtle) throw Error('Cần mở demo qua localhost để kiểm tra SHA-256');
    const digest = await crypto.subtle.digest('SHA-256', buffer);
    const actual = Array.from(new Uint8Array(digest), v => v.toString(16).padStart(2, '0')).join('');
    if (actual !== expected) throw Error('SHA-256 của dữ liệu demo không khớp');
}
function audio() {
    if (!manifest.audio_url) return Promise.resolve(null);
    if (!audioPromise) audioPromise = (async () => {
        const buffer = await bytes(manifest.audio_url, undefined, (loaded, total) => {
            audioStatus = `Đang tải nhạc ${(loaded / 1048576).toFixed(1)}${total ? ` / ${(total / 1048576).toFixed(1)}` : ''} MB…`;
            if (!loadComplete || audioPendingPlayback) $('status').textContent = audioStatus;
        });
        await verify(buffer, manifest.audio_sha256);
        audioStatus = 'Đang giải mã nhạc…';
        if (!loadComplete || audioPendingPlayback) $('status').textContent = audioStatus;
        return (await ready).taikoState.audioContext.decodeAudioData(buffer);
    })().catch(error => { audioPromise = null; throw error; });
    return audioPromise;
}
async function anatomy() {
    const buffer = await bytes(manifest.anatomy.url); await verify(buffer, manifest.anatomy.sha256);
    if (buffer.byteLength !== manifest.anatomy.count * 3 * 4) throw Error('Sai kích thước giải phẫu');
    const raw = new Int32Array(buffer), lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < raw.length; i++) { const a = i % 3; lo[a] = Math.min(lo[a], raw[i]); hi[a] = Math.max(hi[a], raw[i]); }
    const scale = 180 / Math.max(...hi.map((v, a) => v - lo[a]));
    normalized = p => Array.from(p, (v, a) => (v - (lo[a] + hi[a]) / 2) * scale * (a === 1 ? -1 : 1));
    const positions = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i += 3) positions.set(normalized(raw.subarray(i, i + 3)), i);
    const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    background = new THREE.Points(geometry, new THREE.PointsMaterial({color: 0x35506b, size: .65, transparent: true, opacity: .45})); cloud.add(background);
    highlights = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({size: 3, vertexColors: true})); cloud.add(highlights);
    rebuildEdges();
    shown = -1; brainDirty = true;
}
function rebuildEdges() {
    if (!normalized || !trace || !edgeRows.length) return;
    if (edgeLines) { cloud.remove(edgeLines); edgeLines.geometry.dispose(); }
    if (lightNodes) { cloud.remove(lightNodes); lightNodes.geometry.dispose(); }
    if (lightPulses) { cloud.remove(lightPulses); lightPulses.geometry.dispose(); }
    const positions = [];
    for (const [source, target] of edgeRows) {
        const from = trace.sample_soma[source];
        if (!from) continue;
        positions.push(...normalized(from), ...normalized(target));
    }
    const lines = new THREE.BufferGeometry(); lines.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    edgeLines = new THREE.LineSegments(lines, new THREE.LineBasicMaterial({color: 0x336c83, transparent: true, opacity: .32}));
    cloud.add(edgeLines);
    lightNodes = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({
        size: 13, vertexColors: true, transparent: true, opacity: .8,
        blending: THREE.AdditiveBlending, depthWrite: false,
    })); cloud.add(lightNodes);
    lightPulses = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({
        size: 6, vertexColors: true, transparent: true, opacity: .9,
        blending: THREE.AdditiveBlending, depthWrite: false,
    })); cloud.add(lightPulses); shown = -1; brainDirty = true;
}
function timeLabel(ms) { const seconds = Math.max(0, Math.floor(ms / 1000)); return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`; }
function metricsText(m) {
    return `Kết quả đã ghi: Great ${m.great} · Good ${m.good} · Miss ${m.miss} · Acc ${(100 * m.accuracy).toFixed(2)}% · Spinner ${m.swell.completed}/${m.swell.objects} · Drumroll ${m.drumroll.completed}/${m.drumroll.objects}`;
}
async function loadObservations(newTrace, mine, signal) {
    const spec = newTrace.observations;
    if (!spec) return;
    $('input-status').textContent = 'Đang tải ảnh RGB đầu vào…';
    try {
        const observed = await bytes(spec.url, signal, (loaded, total) => {
            if (mine === version) $('input-status').textContent =
                `Đang tải ảnh RGB ${(loaded / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB…`;
        });
        await verify(observed, spec.sha256);
        const bounds = spec.retina_bounds, warp = spec.retina_x_warp;
        if (![[512, 96], [1000, 300]].some(([w, h]) => spec.width === w && spec.height === h) ||
            spec.frames.length !== newTrace.time_ms.length ||
            !Array.isArray(bounds) || bounds.length !== 4 ||
            (warp != null && (!Array.isArray(warp) || warp.length !== 2 ||
                !(0 < warp[0] && warp[0] < 1 && bounds[0] < warp[1] && warp[1] < bounds[2]))) ||
            !(0 <= bounds[0] && bounds[0] < bounds[2] && bounds[2] <= 1 &&
              0 <= bounds[1] && bounds[1] < bounds[3] && bounds[3] <= 1) ||
            spec.frames.some(([offset, length, stamp], i) => !Number.isInteger(offset) ||
                !Number.isInteger(length) || length < 32 || offset < 0 || offset + length > observed.byteLength ||
                !Number.isInteger(stamp) || stamp > newTrace.time_ms[i] || newTrace.time_ms[i] - stamp > 50))
            throw Error('Sai chỉ mục ảnh RGB đã ghi');
        if (mine !== version) return;
        observationBytes = observed;
        window.taikoDemo.observations = observed;
        observationPaint = -1;
        $('input-status').textContent = 'Ảnh RGB đã sẵn sàng.';
        if (shown >= 0) updateObservation(shown);
    } catch (error) {
        if (mine === version && error.name !== 'AbortError') $('input-status').textContent =
            'Không tải được ảnh RGB: ' + error.message;
    }
}
async function load(index) {
    const mine = ++version; aborter?.abort(); aborter = new AbortController(); const signal = aborter.signal;
    loadComplete = false;
    const w = await ready; w.pauseGame();
    for (const id of ['play', 'pause', 'restart', 'seek']) $(id).disabled = true;
    $('status').textContent = 'Đang tải replay và trace thực…';
    const item = manifest.replays[index];
    try {
        const [newGame, newTrace] = await Promise.all([json(item.gameplay_url, signal), json(item.trace_url, signal)]);
        if (mine !== version) return;
        $('status').textContent = 'Đang tải hoạt động neuron…';
        const binary = await bytes(newTrace.data_url, signal, (loaded, total) => {
            if (mine === version) $('status').textContent = `Đang tải trace ${(loaded / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB…`;
        });
        await verify(binary, newTrace.sha256);
        const expected = newTrace.shape.reduce((a, b) => a * b, 1) * 4;
        if (binary.byteLength !== expected || newTrace.shape[0] !== newTrace.time_ms.length ||
            newTrace.shape[1] !== 2 || newTrace.shape[2] !== newTrace.body_ids.length ||
            newTrace.sample_soma.length !== newTrace.body_ids.length) throw Error('Sai kích thước trace thực');
        if (mine !== version) return;
        game = newGame; trace = newTrace; values = new Float32Array(binary); shown = -1;
        observationBytes = null; observationPaint = -1;
        $('model-input').getContext('2d').clearRect(0, 0, 512, 96);
        $('retina-input').getContext('2d').clearRect(0, 0, 137, 53);
        $('input-status').textContent = newTrace.observations ? 'Đang tải ảnh RGB đầu vào…' : 'Replay này chưa có ảnh RGB đầu vào đã ghi.';
        const n = trace.body_ids.length;
        // Display contrast is per recorded soma, not a biological firing rate.
        // A global 0.5-unit floor hid most measured, located neurons entirely.
        neuronContrast = new Float32Array(n).fill(.02); neuronIntensity = new Float32Array(n);
        for (let frame = 0; frame < trace.time_ms.length; frame++) {
            for (let i = 0; i < n; i++) {
                neuronContrast[i] = Math.max(neuronContrast[i],
                    Math.abs(values[frame * 2 * n + i] - values[i]));
            }
        }
        rebuildEdges();
        w.loadRecordedReplay(game, null); w.setVolume(Number($('volume').value));
        $('seek').max = game.duration_ms; $('seek').value = 0;
        $('metrics').textContent = metricsText(game.metrics);
        $('clock').textContent = `0:00 / ${timeLabel(game.duration_ms)}`;
        $('status').textContent = manifest.audio_url ? 'Sẵn sàng. Nhấn Phát replay để nghe nhạc và xem taiko.' : 'Sẵn sàng. Nhấn Phát replay để xem taiko (chưa có file nhạc trên server).';
        for (const id of ['play', 'pause', 'restart', 'seek']) $(id).disabled = false;
        loadComplete = true; resize();
        window.taikoDemo = {game, trace, values, observations: null, scope: item.kind, ready: true};
        loadObservations(newTrace, mine, signal);
        if (manifest.audio_url) audio().catch(error => {
            if (mine === version) $('status').textContent = 'Không tải được nhạc: ' + error.message;
        });
    } catch (error) {
        if (mine === version && error.name !== 'AbortError') { $('status').textContent = 'Lỗi: ' + error.message; $('neural').textContent = 'Không tải được replay. Có thể chọn lại bài để thử lại.'; }
    }
}
$('examples').onchange = () => load(Number($('examples').value)).catch(e => $('status').textContent = 'Lỗi: ' + e.message);
$('skin').onchange = async () => {
    const selected = $('skin').value;
    try {
        const player = await ready;
        await player.setRecordedTaikoSkin(selected);
        const saved = localStorage.getItem('flytaiko-hit-volume-' + selected);
        $('hit-volume').value = saved === null ? '.55' : saved;
        player.setRecordedHitVolume(Number($('hit-volume').value));
    }
    catch (error) {
        $('skin').value = 'default';
        await viewer.contentWindow.setRecordedTaikoSkin('default');
        $('status').textContent = 'Không tải được skin: ' + error.message;
    }
};
$('play').onclick = async () => {
    try {
        const mine = version;
        const w = viewer.contentWindow;
        if (!w?.taikoState?.audioContext) throw Error('Audio player is not ready');
        // Start resume synchronously under the click's user activation, but
        // don't wait for it before reporting download progress. Some browsers
        // leave its promise pending until their audio device is ready.
        const resume = w.taikoState.audioContext.resume();
        if (manifest.audio_url) {
            audioPendingPlayback = true;
            $('status').textContent = audioPromise ? audioStatus : 'Đang tải nhạc 0 MB…';
            const buffer = await audio();
            if (mine !== version) return;
            w.setRecordedAudio(buffer);
        }
        $('status').textContent = 'Đang khởi động âm thanh…';
        await resume;
        await w.startGame();
        $('status').textContent = 'Đang phát replay · phím và judgment giữ nguyên.';
    }
    catch (e) { $('status').textContent = 'Không phát được nhạc: ' + e.message; }
    finally { audioPendingPlayback = false; }
};
$('pause').onclick = () => { viewer.contentWindow.pauseGame(); $('status').textContent = 'Đã tạm dừng.'; };
$('restart').onclick = () => { viewer.contentWindow.pauseGame(); viewer.contentWindow.seekGame(0); $('status').textContent = 'Đã về đầu. Nhấn Phát replay.'; };
$('seek').oninput = () => viewer.contentWindow.seekGame(Number($('seek').value));
$('volume').oninput = () => viewer.contentWindow.setVolume(Number($('volume').value));
$('hit-volume').oninput = () => {
    const value = Number($('hit-volume').value);
    viewer.contentWindow.setRecordedHitVolume(value);
    localStorage.setItem('flytaiko-hit-volume-' + $('skin').value, String(value));
};
window.addEventListener('message', e => {
    if (e.source !== viewer.contentWindow || e.origin !== location.origin) return;
    if (e.data?.type === 'TAIKO_RECORDED_ENDED') $('status').textContent = 'Đã phát hết replay. Có thể chọn toàn bài hoặc phát lại.';
});
function sample(time) {
    let lo = 0, hi = trace.time_ms.length - 1;
    if (time < trace.time_ms[0]) return -1;
    while (lo < hi) { const m = Math.ceil((lo + hi) / 2); if (trace.time_ms[m] <= time) lo = m; else hi = m - 1; }
    return lo;
}
async function updateObservation(index) {
    if (!observationBytes || !trace?.observations || index < 0 || index === observationPaint) return;
    observationPaint = index;
    const mine = version, localBytes = observationBytes;
    const [offset, length, observationTime] = trace.observations.frames[index];
    const blob = new Blob([localBytes.slice(offset, offset + length)], {type: 'image/png'});
    try {
        const bitmap = await createImageBitmap(blob);
        if (mine === version && localBytes === observationBytes && index === shown) {
            const modelInput = $('model-input');
            if (modelInput.width !== bitmap.width || modelInput.height !== bitmap.height) {
                modelInput.width = bitmap.width; modelInput.height = bitmap.height;
                modelInput.style.aspectRatio = `${bitmap.width}/${bitmap.height}`;
            }
            modelInput.getContext('2d').drawImage(bitmap, 0, 0);
            const [l, t, r, b] = trace.observations.retina_bounds;
            const frameWidth = bitmap.width, frameHeight = bitmap.height;
            const x = Math.round(l * (frameWidth - 1)), y = Math.round(t * (frameHeight - 1));
            const width = Math.round(r * (frameWidth - 1)) - x + 1;
            const height = Math.round(b * (frameHeight - 1)) - y + 1;
            const crop = $('retina-input');
            if (crop.width !== width || crop.height !== height) { crop.width = width; crop.height = height; }
            crop.getContext('2d').drawImage(bitmap, x, y, width, height, 0, 0, width, height);
            $('input-status').textContent = `Đầu vào thật lúc ${(observationTime / 1000).toFixed(3)} s · frame ghi ${(trace.time_ms[index] / 1000).toFixed(3)} s · ${frameWidth}×${frameHeight} RGB`;
        }
        bitmap.close();
    } catch (error) {
        if (mine === version) $('input-status').textContent = 'Không giải mã được frame RGB: ' + error.message;
    }
}
function updateBrain(index) {
    shown = index;
    updateObservation(index);
        if (index < 0) { $('neural').textContent = 'Chưa có frame neuron được ghi tại thời điểm này.'; return; }
    let activeCount = 0, visibleSomaCount = 0;
    const n = trace.body_ids.length;
    for (let i = 0; i < n; i++) {
        const delta = Math.abs(values[index * 2 * n + i] - values[i]);
        neuronIntensity[i] = delta < .003 ? 0 : Math.sqrt(Math.min(1, delta / neuronContrast[i]));
        if (delta >= .5) activeCount++;
        if (trace.sample_soma[i] && neuronIntensity[i] > .16) visibleSomaCount++;
    }
    $('neural').textContent = `Trace ${(trace.time_ms[index] / 1000).toFixed(3)} s · ${activeCount}/${n} neuron mẫu đổi ≥0,5 proxy · ${visibleSomaCount} soma sáng (chuẩn hoá độ tương phản) · không nội suy`;
    $('groups').textContent = trace.group_names.map((name, i) => {
        const row = trace.groups[index][i];
        return `${name || '(chưa phân nhóm)'}: ${row[0].toFixed(2)} activity proxy · ${row[1].toFixed(2)} voltage proxy`;
    }).join('\n') + '\n\nGraded-rate · no discrete spikes · không phải Hz/mV sinh học';
    if (!highlights || !normalized) return;
    const positions = [], colors = [], glowPositions = [], glowColors = [], color = new THREE.Color();
    trace.sample_soma.forEach((p, i) => {
        if (!p) return;
        positions.push(...normalized(p));
        const activity = neuronIntensity[i];
        color.setHSL(.58 - activity * .48, .85, .27 + activity * .5); colors.push(color.r, color.g, color.b);
        if (activity > .16) { glowPositions.push(...normalized(p)); glowColors.push(color.r, color.g, color.b); }
    });
    highlights.geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    highlights.geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3)); highlights.geometry.computeBoundingSphere(); brainDirty = true;
    if (lightNodes) {
        lightNodes.geometry.setAttribute('position', new THREE.Float32BufferAttribute(glowPositions, 3));
        lightNodes.geometry.setAttribute('color', new THREE.Float32BufferAttribute(glowColors, 3));
        lightNodes.geometry.computeBoundingSphere();
    }
}
function updatePulses(time) {
    if (!trace || !lightPulses || shown < 0 || !edgeRows.length) return;
    const phase = ((time % 480) + 480) % 480 / 480;
    const positions = [], colors = [];
    for (const [source, target] of edgeRows) {
        const activity = neuronIntensity[source], from = trace.sample_soma[source];
        if (activity <= .16 || !from) continue;
        const a = normalized(from), b = normalized(target);
        positions.push(...a.map((v, axis) => v + (b[axis] - v) * phase));
        colors.push(1, .35 + .65 * activity, .12);
    }
    lightPulses.geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    lightPulses.geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    lightPulses.geometry.computeBoundingSphere(); brainDirty = true;
}
function drawBrain() {
    brainDirty = false;
    if (renderer) { renderer.render(scene, camera); return; }
    const ctx = surface.getContext('2d'); ctx.clearRect(0, 0, surface.width, 360);
    if (!background) return;
    const cy = Math.cos(cloud.rotation.y), sy = Math.sin(cloud.rotation.y), cx = Math.cos(cloud.rotation.x), sx = Math.sin(cloud.rotation.x);
    const scale = Math.min(surface.width / 205, 360 / 205);
    function dot(p, i, size) {
        const x = p[i] * cy + p[i + 2] * sy, z = -p[i] * sy + p[i + 2] * cy, y = p[i + 1] * cx - z * sx;
        ctx.fillRect(surface.width / 2 + x * scale, 180 + y * scale, size, size);
    }
    ctx.fillStyle = '#263d55'; const p = background.geometry.attributes.position.array;
    for (let i = 0; i < p.length; i += 48) dot(p, i, 1);
    const hp = highlights?.geometry.attributes.position, hc = highlights?.geometry.attributes.color;
    if (hp && hc) for (let i = 0; i < hp.array.length; i += 3) {
        const c = hc.array; ctx.fillStyle = `rgb(${c[i] * 255},${c[i + 1] * 255},${c[i + 2] * 255})`; dot(hp.array, i, 3);
    }
    const edgep = edgeLines?.geometry.attributes.position?.array;
    if (edgep) {
        ctx.strokeStyle = 'rgba(75,150,180,.20)'; ctx.lineWidth = 1;
        function projected(p, i) {
            const x = p[i] * cy + p[i + 2] * sy, z = -p[i] * sy + p[i + 2] * cy, y = p[i + 1] * cx - z * sx;
            return [surface.width / 2 + x * scale, 180 + y * scale];
        }
        for (let i = 0; i < edgep.length; i += 6) {
            const a = projected(edgep, i), b = projected(edgep, i + 3);
            ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
        }
        const gp = lightNodes?.geometry.attributes.position?.array;
        if (gp) for (let i = 0; i < gp.length; i += 3) {
            const [x, y] = projected(gp, i), halo = ctx.createRadialGradient(x, y, 0, x, y, 15);
            halo.addColorStop(0, 'rgba(255,245,125,.9)'); halo.addColorStop(1, 'rgba(255,180,35,0)');
            ctx.fillStyle = halo; ctx.beginPath(); ctx.arc(x, y, 15, 0, Math.PI * 2); ctx.fill();
        }
        const pp = lightPulses?.geometry.attributes.position?.array;
        if (pp) { ctx.fillStyle = '#ffe281'; for (let i = 0; i < pp.length; i += 3) {
            const [x, y] = projected(pp, i); ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill();
        } }
    }
}
function animate(now) {
    requestAnimationFrame(animate);
    if (loadComplete && game && trace) {
        const state = viewer.contentWindow.taikoState, time = state.currentTimeMs;
        const index = sample(time); if (index !== shown) updateBrain(index);
        if (state.isPlaying && now - lastBrainPaint >= 40) {
            cloud.rotation.y += .004; updatePulses(time); lastBrainPaint = now; brainDirty = true;
        }
        if (now - lastHUD > 200) {
            $('seek').value = time; $('clock').textContent = `${timeLabel(time)} / ${timeLabel(game.duration_ms)}`;
            const fps = state.displayFps || 0;
            const sceneFps = window.flyTaikoScene?.displayFps || 0;
            $('fps').textContent = `Taiko ${fps ? fps.toFixed(0) : '—'} FPS · 3D ${sceneFps ? sceneFps.toFixed(0) : '—'} FPS · target 120 (limited by display/browser).`;
            $('play').textContent = state.isPlaying ? 'Đang phát' : (time >= game.duration_ms ? 'Phát lại' : 'Phát replay'); lastHUD = now;
        }
    }
    if (brainDirty) drawBrain();
}
requestAnimationFrame(animate);
async function initialize() { try {
    let requested = new URLSearchParams(location.search).get('dataset');
    if (!requested) {
        const response = await fetch(backend('/api/replays'), {cache: 'no-store'});
        if (!response.ok) throw Error(`Replay library: HTTP ${response.status}`);
        requested = (await response.json()).replays.find(r => r.status === 'complete')?.dataset;
        if (!requested) { location.replace('create-replay.html'); return; }
    }
    const dataset = requested === 'phase-a' ? 'malecns-taiko-v18-phase-a' :
        requested === 'playing-god-v19' ? 'playing-god-v19' :
        requested === 'playing-god-v23' ? 'playing-god-v23' :
        requested === 'ideoless-v23' ? 'ideoless-v23' :
        /^user-v(?:23|24|25)-[a-f0-9]{12}$/.test(requested || '') ? requested :
        requested === 'v19' ? 'malecns-taiko-v19' :
        requested === 'v18' ? 'malecns-taiko-v18' :
        requested === 'v17' ? 'malecns-taiko-v17' :
        requested === 'v16' ? 'malecns-taiko-v16' : 'malecns-taiko';
    refreshReplayLibrary(dataset).catch(error => { $('library').replaceChildren(new Option(error.message, '')); });
    manifest = /^user-v(?:23|24|25)-/.test(dataset) ? await waitForUserReplay(dataset) :
        await json(`demos/${dataset}/manifest.json`);
    if (dataset !== 'playing-god-v19' && dataset !== 'playing-god-v23' && dataset !== 'ideoless-v23' && !/^user-v(?:23|24|25)-/.test(dataset)) {
        document.querySelector('h1').textContent = 'IDEALESS IDEOLOGY · CATASTROPHE';
        $('ideology-sv-link').hidden = false;
        $('source-links').innerHTML = '<a href="malecns.html?manifest=demos/malecns-phase-a-replays.json">Xem ảnh RGB gốc</a> · <a href="demos/malecns-taiko/manifest.json">Manifest/provenance</a> · <a href="demos/malecns-phase-a-ideology-full.json" download>Replay gốc đầy đủ (155 MB)</a> · <a href="https://male-cns.janelia.org/download/">Nguồn MaleCNS, CC-BY-4.0</a>';
    }
    if (dataset === 'playing-god-v19') {
        document.querySelector('h1').textContent = 'Playing God · Superstition';
        $('checkpoint-label').textContent = 'V19 · BONUS TRAIN MAP';
        $('scope-description').textContent = 'Replay offline từ checkpoint v19 trên map bonus đã dùng khi train. Không phải kết quả trên bài chưa từng học. Máy chủ chỉ có chart .osu, chưa có file nhạc.';
        $('source-links').innerHTML = '<a href="demos/playing-god-v19/manifest.json">Manifest/provenance</a> · <a href="demos/playing-god-v19-replay.json" download>Replay gốc</a>';
    }
    if (dataset === 'playing-god-v23') {
        document.querySelector('h1').textContent = 'Playing God · Superstition';
        $('checkpoint-label').textContent = 'V23 · PHASE C BEST · TRAIN MAP';
        $('scope-description').textContent = 'Replay offline nguyên bài từ checkpoint v23 Phase C. Map này nằm trong tập bonus train; không phải bài chưa từng học. Đầu vào thật là RGB 1000×300; máy chủ chưa có file nhạc.';
        $('source-links').innerHTML = '<a href="demos/playing-god-v23/manifest.json">Manifest/provenance</a> · <a href="demos/playing-god-v23-replay.json" download>Replay gốc</a>';
    }
    if (dataset === 'ideoless-v23') {
        document.querySelector('h1').textContent = 'IDEALESS IDEOLOGY · CATASTROPHE';
        $('checkpoint-label').textContent = 'V23 · PHASE C BEST · FULL MAP';
        $('scope-description').textContent = 'Replay offline nguyên bài từ checkpoint v23 Phase C, ảnh đầu vào RGB 1000×300. Phím và judgment là kết quả mô phỏng, không căn theo note; chưa phải điều khiển osu! thật.';
        $('source-links').innerHTML = '<a href="demos/ideoless-v23/manifest.json">Manifest/provenance</a> · <a href="demos/ideoless-v23-replay.json" download>Replay gốc</a>';
    }
    if (/^user-v(?:23|24|25)-/.test(dataset)) {
        const meta = manifest.replays[0]?.label || 'Map Taiko đã tải lên';
        document.querySelector('h1').textContent = meta;
        const modelVersion = dataset.slice(5, 8);
        $('checkpoint-label').textContent = `${modelVersion.toUpperCase()} · MAP NGƯỜI DÙNG`;
        $('scope-description').textContent = `Replay offline từ file .osz đã tải lên. Đây là suy luận từ checkpoint ${modelVersion}; không train thêm, không điều khiển osu! thật.`;
        $('source-links').innerHTML = `<a href="demos/${dataset}/manifest.json">Manifest/provenance</a>`;
    }
    if (dataset === 'malecns-taiko-v16') {
        $('checkpoint-label').textContent = 'V16 · CHECKPOINT ĐƯỢC CHỌN';
        $('scope-description').textContent = 'Replay offline của chiến dịch v16 trên chart native đã xác minh. Phím, timing và judgment giữ nguyên, không căn hit về note. SV lấy từ chart native; âm thanh lấy từ OSZ cùng bài. Kết quả kiểm tra độc lập ở tệp quality.json.';
        $('source-links').innerHTML = '<a href="demos/malecns-taiko-v16/manifest.json">Manifest/provenance</a> · <a href="demos/malecns-taiko-v16/quality.json">Kiểm tra chất lượng</a> · <a href="demos/malecns-v16-final-replay.json" download>Replay gốc v16</a> · <a href="https://male-cns.janelia.org/download/">Nguồn MaleCNS, CC-BY-4.0</a>';
    }
    if (dataset === 'malecns-taiko-v18') {
        $('checkpoint-label').textContent = 'V18 · OSU TAIKO SV 16:9';
        $('scope-description').textContent = 'Replay offline của chiến dịch v18 với SV quy đổi theo vùng cuộn osu!taiko 16:9. Giao diện và khung RGB model nhận được hiển thị riêng; đây chưa phải điều khiển osu! thật.';
        $('source-links').innerHTML = '<a href="demos/malecns-taiko-v18/manifest.json">Manifest/provenance</a> · <a href="demos/malecns-v18-final-replay.json" download>Replay gốc v18</a> · <a href="https://male-cns.janelia.org/download/">Nguồn MaleCNS, CC-BY-4.0</a>';
    }
    if (dataset === 'malecns-taiko-v18-phase-a') {
        $('checkpoint-label').textContent = 'V18 · PHASE A BEST · EPOCH 9';
        $('scope-description').textContent = 'Replay toàn bài từ checkpoint Phase A được chọn ở epoch 9, chưa cần chờ đánh giá độc lập cuối chiến dịch. Phím và judgment là kết quả mô phỏng đã ghi; không căn theo note.';
        $('source-links').innerHTML = '<a href="demos/malecns-taiko-v18-phase-a/manifest.json">Manifest/provenance</a> · <a href="demos/malecns-v18-warmval-phase-a-replay.json" download>Replay Phase A gốc</a> · <a href="https://male-cns.janelia.org/download/">Nguồn MaleCNS, CC-BY-4.0</a>';
    }
    if (dataset === 'malecns-taiko-v19') {
        $('checkpoint-label').textContent = 'V19 · DEFAULT-STYLE INPUT · FOVEATED VIEW';
        $('scope-description').textContent = 'Replay từ checkpoint v19 được chọn sau khi train lại với ảnh default-style 512×96 và photoreceptor lấy mẫu nhiều tỷ lệ trên vùng x=40…511. Phím và judgment là kết quả mô phỏng đã ghi, không căn theo note.';
        $('source-links').innerHTML = '<a href="demos/malecns-taiko-v19/manifest.json">Manifest/provenance</a> · <a href="demos/malecns-v19-skin-foveated-replay.json" download>Replay v19 gốc</a> · <a href="https://male-cns.janelia.org/download/">Nguồn MaleCNS, CC-BY-4.0</a>';
    }
    $('source-links').querySelectorAll('a[href^="demos/"]').forEach(link => {
        link.href = backend(link.getAttribute('href'));
    });
    manifest.replays.forEach((r, i) => $('examples').add(new Option(r.label, i))); $('examples').disabled = false;
    anatomy().catch(e => { $('brain').textContent = 'Không tải được giải phẫu: ' + e.message; });
    bytes(manifest.sample_edges.url).then(async response => {
        await verify(response, manifest.sample_edges.sha256);
        const rows = JSON.parse(new TextDecoder().decode(response));
        if (rows.length !== manifest.sample_edges.count || rows.some(r => !Number.isInteger(r[0]) || r[0] < 0))
            throw Error('Sai dữ liệu cạnh thần kinh');
        edgeRows = rows; rebuildEdges();
    }).catch(e => { $('neural').title = 'Không tải được các cạnh giải phẫu: ' + e.message; });
    await load(0);
} catch (e) { $('status').textContent = 'Lỗi tải demo: ' + e.message; } }
initialize();
