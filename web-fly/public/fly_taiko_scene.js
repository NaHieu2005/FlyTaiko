/* Presentation-only Flybody scene driven by the existing recorded Taiko replay.
 * It reads rendered pixels and recorded key actions; it does not run inference.
 */
import * as THREE from './vendor/three.module.js';

const host = document.getElementById('fly-stage');
const status = document.getElementById('fly-status');
const viewer = document.getElementById('viewer');
const MODEL_ROOT = 'data/flybody/';
const fullscreenButton = document.getElementById('fly-fullscreen');
fullscreenButton.onclick = async () => {
    try {
        if (document.fullscreenElement === host) await document.exitFullscreen();
        else await host.requestFullscreen();
    } catch (error) { status.textContent = 'Could not enter fullscreen: ' + error.message; }
};
document.addEventListener('fullscreenchange', () => {
    fullscreenButton.textContent = document.fullscreenElement === host ? '⤢ Exit fullscreen' : '⛶ Fullscreen';
});

let lastOverlay = 0, brainLayout = null, brainTrace = null;
const FRAME_MS = 1000 / 120;
function frameDue(now, clock) {
    if (clock.next === null || now - clock.next > 1000) {
        clock.next = now + FRAME_MS;
        return true;
    }
    if (now + .25 < clock.next) return false;
    clock.next += (Math.floor(Math.max(0, now - clock.next) / FRAME_MS) + 1) * FRAME_MS;
    return true;
}
function recordSceneFrame(now, counter) {
    if (counter.start === null) counter.start = now;
    counter.frames++;
    if (now - counter.start >= 1000) {
        if (window.flyTaikoScene) window.flyTaikoScene.displayFps = counter.frames * 1000 / (now - counter.start);
        counter.start = now;
        counter.frames = 0;
    }
}
function sampleIndex(times, time) {
    let left = 0, right = times.length - 1;
    if (!times.length || time < times[0]) return -1;
    while (left < right) {
        const middle = Math.ceil((left + right) / 2);
        if (times[middle] <= time) left = middle;
        else right = middle - 1;
    }
    return left;
}
function brainOverlay(time) {
    const canvas = document.getElementById('fly-brain-canvas'), ctx = canvas.getContext('2d');
    ctx.fillStyle = '#091421'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    const demo = window.taikoDemo, trace = demo?.trace, values = demo?.values;
    if (!trace?.sample_soma || !values) return;
    if (brainTrace !== trace) {
        const soma = trace.sample_soma, located = soma.filter(p => p !== null);
        const xs = located.map(p => p[0]), zs = located.map(p => p[2]);
        if (!located.length) return;
        const xMin = Math.min(...xs), xSpan = Math.max(1, Math.max(...xs) - xMin);
        const zMin = Math.min(...zs), zSpan = Math.max(1, Math.max(...zs) - zMin);
        brainLayout = soma.map(p => p === null ? null :
            [26 + (p[0] - xMin) / xSpan * (canvas.width - 52),
             22 + (p[2] - zMin) / zSpan * (canvas.height - 44)]);
        brainTrace = trace;
    }
    const index = sampleIndex(trace.time_ms, time), count = trace.shape[2];
    const base = index < 0 ? 0 : index * 2 * count;
    ctx.fillStyle = 'rgba(63,129,190,.10)';
    ctx.beginPath(); ctx.ellipse(canvas.width / 2, canvas.height / 2,
        canvas.width * .43, canvas.height * .42, 0, 0, Math.PI * 2); ctx.fill();
    for (let i = 0; i < count; i++) {
        if (!brainLayout[i]) continue; // No measured soma location: never invent one.
        const [x, y] = brainLayout[i];
        const change = Math.abs(values[base + i] - values[i]);
        const intensity = Math.max(0, Math.min(1, (Math.log10(change + 1e-4) + 4) / 4));
        ctx.beginPath(); ctx.arc(x, y, 1.7 + 3.2 * intensity, 0, Math.PI * 2);
        ctx.fillStyle = intensity > .5 ? `rgba(255,214,106,${.35 + .65 * intensity})` :
            `rgba(103,188,248,${.32 + .5 * intensity})`;
        ctx.fill();
    }
}
function urOverlay(time, avgError) {
    const canvas = document.getElementById('fly-ur-canvas'), ctx = canvas.getContext('2d');
    const width = canvas.width, height = canvas.height;
    const colors = {great: '#21d6ef', good: '#49e900', miss: '#e5b43f'};
    ctx.clearRect(0, 0, width, height);
    const rawOd = Number(window.taikoDemo?.game?.metadata?.overall_difficulty);
    const od = Number.isFinite(rawOd) ? Math.max(0, Math.min(10, rawOd)) : 5;
    const great = Math.floor(50 - 3 * od) - .5;
    const good = Math.floor(od <= 5 ? 120 - 8 * od : 110 - 6 * od) - .5;
    // Keep the real OD-dependent Great/Good boundaries, while making the
    // outer (miss) zone fill the remainder like the reference UR meter.
    const range = Math.max(great / .27, good / .64);
    const xFor = error => Math.max(2, Math.min(width - 2, width * (error + range) / (2 * range)));
    const bandTop = 32, bandHeight = 22;
    for (const [left, right, color] of [
        [-range, range, colors.miss], [-good, good, colors.good], [-great, great, colors.great],
    ]) {
        ctx.fillStyle = color;
        ctx.fillRect(xFor(left), bandTop, xFor(right) - xFor(left), bandHeight);
    }
    ctx.fillStyle = 'rgba(255,255,255,.12)';
    ctx.fillRect(2, bandTop, width - 4, 2);
    const events = window.taikoDemo?.game?.events;
    let sum = 0, hitCount = 0, tickCount = 0;
    if (events?.length) {
        const end = lowerBound(events, time + .001);
        for (let i = end - 1; i >= 0 && tickCount < 32; i--) {
            const event = events[i], error = event.timing_error_ms;
            const age = time - event.time_ms;
            if (age > 1800) break;
            if (!Number.isFinite(error) || !['great', 'good', 'miss'].includes(event.judgment)) continue;
            ctx.fillStyle = colors[event.judgment];
            ctx.globalAlpha = Math.max(0, 1 - age / 1800);
            ctx.fillRect(xFor(error) - 1, bandTop - 10, 2, bandHeight + 20);
            tickCount++;
            if (event.judgment !== 'miss') { sum += error; hitCount++; }
        }
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#fff';
    ctx.fillRect(Math.round(width / 2) - 2, bandTop - 12, 4, bandHeight + 24);
    if (hitCount) {
        const x = xFor(sum / hitCount);
        ctx.fillStyle = '#fff'; ctx.beginPath();
        ctx.moveTo(x - 21, 16); ctx.lineTo(x + 21, 16); ctx.lineTo(x, 39); ctx.fill();
    }
    const roundedAverage = Number.isFinite(avgError) ? Math.round(avgError * 10) / 10 : null;
    document.getElementById('fly-avg-error').textContent = roundedAverage !== null ?
        `${roundedAverage > 0 ? '+' : ''}${roundedAverage.toFixed(1)} ms` : '—';
}
function updateOverlay(now) {
    if (now - lastOverlay < 100) return;
    lastOverlay = now;
    const page = viewer.contentWindow, snapshot = page?.recordedReplaySnapshot?.();
    const time = snapshot?.time_ms ?? 0;
    const count = (snapshot?.great || 0) + (snapshot?.good || 0) + (snapshot?.miss || 0);
    document.getElementById('fly-accuracy').textContent =
        (count ? 100 * ((snapshot.great || 0) + .5 * (snapshot.good || 0)) / count : 100).toFixed(2) + '%';
    document.getElementById('fly-great').textContent = snapshot?.great ?? 0;
    document.getElementById('fly-good').textContent = snapshot?.good ?? 0;
    document.getElementById('fly-miss').textContent = snapshot?.miss ?? 0;
    document.getElementById('fly-extra').textContent = snapshot?.false_hits ?? 0;
    document.getElementById('fly-time').textContent = `${Math.floor(time / 60000)}:${String(Math.floor(time / 1000) % 60).padStart(2, '0')}`;
    urOverlay(time, snapshot?.mean_signed_error_ms); brainOverlay(time);
}

function actionKeys(id) {
    return [[], ['F'], ['J'], ['D'], ['K'], ['F', 'J'], ['D', 'K']][id] || [];
}

function lowerBound(actions, time) {
    let left = 0, right = actions.length;
    while (left < right) {
        const middle = (left + right) >> 1;
        if (actions[middle].time_ms < time) left = middle + 1;
        else right = middle;
    }
    return left;
}

function recordedForces() {
    const force = {D: 0, F: 0, J: 0, K: 0};
    const state = viewer.contentWindow?.taikoState;
    if (!state?.isPlaying || !Array.isArray(state.replayActions)) return force;
    const time = state.currentTimeMs, actions = state.replayActions;
    for (let i = lowerBound(actions, time - 100); i < actions.length; i++) {
        const delta = time - actions[i].time_ms;
        if (delta < 0) break;
        if (delta > 100) continue;
        for (const key of actionKeys(actions[i].action_id))
            force[key] = Math.max(force[key], 1 - delta / 100);
    }
    return force;
}

// The model has articulated forelegs; the middle legs are part of the body mesh.
// Locate all four tarsi from actual vertices, not guessed desktop coordinates.
function footMean(vertices, predicate) {
    const selected = vertices.filter(predicate).sort((a, b) => a.y - b.y);
    if (!selected.length) throw Error('Flybody foot endpoints not found');
    const lowest = selected.slice(0, Math.max(4, Math.ceil(selected.length * .1)));
    return lowest.reduce((mean, p) => mean.add(p), new THREE.Vector3()).divideScalar(lowest.length);
}

function footTargets(parts, pivots) {
    const body = parts.body, left = parts.front_left, right = parts.front_right;
    const middle = side => footMean(body, p => p.x * side > .15 && p.x * side < .25 &&
        p.y < -.12 && p.z > .01 && p.z < .04);
    return {
        D: middle(1), F: footMean(left, () => true).add(new THREE.Vector3(...pivots.front_left)),
        J: footMean(right, () => true).add(new THREE.Vector3(...pivots.front_right)),
        K: middle(-1),
    };
}

function labelTexture(text) {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 128;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff'; ctx.font = 'bold 82px system-ui';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(text, 64, 65);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
}

async function start() {
    const renderer = new THREE.WebGLRenderer({antialias: true, powerPreference: 'low-power'});
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    host.append(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111827);
    const camera = new THREE.PerspectiveCamera(43, 1, .01, 40);
    const look = new THREE.Vector3(0, .12, -.08);
    let yaw = .55, pitch = .30, distance = 2.25;
    function placeCamera() {
        camera.position.set(look.x + distance * Math.sin(yaw) * Math.cos(pitch),
            look.y + distance * Math.sin(pitch),
            look.z + distance * Math.cos(yaw) * Math.cos(pitch));
        camera.lookAt(look);
    }
    placeCamera();
    scene.add(new THREE.HemisphereLight(0xd8e5ff, 0x3c2847, 2.1));
    const light = new THREE.DirectionalLight(0xffd8b7, 3.2);
    light.position.set(1.5, 2, 2); scene.add(light);
    const rim = new THREE.DirectionalLight(0x6ba7ff, 1.3);
    rim.position.set(-1.5, .6, -1.5); scene.add(rim);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(20, 20),
        new THREE.MeshStandardMaterial({color: 0x171d29, roughness: 1}));
    floor.rotation.x = -Math.PI / 2; floor.position.y = -.28; scene.add(floor);
    const grid = new THREE.GridHelper(20, 40, 0x4d607b, 0x26364e);
    grid.position.y = -.278; scene.add(grid);
    const desk = new THREE.Mesh(new THREE.BoxGeometry(1.4, .06, 1.0),
        new THREE.MeshStandardMaterial({color: 0x343d50, roughness: .8}));
    desk.position.set(0, -.19, -.05); scene.add(desk);

    const monitor = new THREE.Mesh(new THREE.BoxGeometry(1.25, .72, .05),
        new THREE.MeshStandardMaterial({color: 0x090b12, metalness: .2, roughness: .5}));
    monitor.position.set(0, .35, -.55); scene.add(monitor);
    const stand = new THREE.Mesh(new THREE.BoxGeometry(.10, .17, .07), monitor.material);
    stand.position.set(0, -.09, -.55); scene.add(stand);
    const base = new THREE.Mesh(new THREE.BoxGeometry(.35, .02, .23), monitor.material);
    base.position.set(0, -.16, -.55); scene.add(base);
    const blank = document.createElement('canvas'); blank.width = 1000; blank.height = 300;
    const blankPaint = blank.getContext('2d');
    blankPaint.fillStyle = '#151924'; blankPaint.fillRect(0, 0, 1000, 300);
    blankPaint.fillStyle = '#93a6c2'; blankPaint.font = 'bold 36px system-ui';
    blankPaint.textAlign = 'center'; blankPaint.fillText('Taiko replay', 500, 165);
    let screenTexture = new THREE.CanvasTexture(blank);
    let textureWidth=blank.width,textureHeight=blank.height;
    screenTexture.colorSpace = THREE.SRGBColorSpace;
    screenTexture.minFilter = THREE.LinearFilter;
    const screen = new THREE.Mesh(new THREE.PlaneGeometry(1.18, 1.18*9/16),
        new THREE.MeshBasicMaterial({map: screenTexture, toneMapped: false}));
    screen.position.set(0, .35, -.521); scene.add(screen);
    // Explicit judgment target on the monitor itself (x=150 of 1000 lane px).
    const hitX = -1.18 / 2 + 1.18 * 187.5 / 1000;
    const targetRing = new THREE.Mesh(new THREE.RingGeometry(.043, .052, 48),
        new THREE.MeshBasicMaterial({color: 0xf0f7ff, transparent: true, opacity: .95,
            side: THREE.DoubleSide, toneMapped: false}));
    targetRing.position.set(hitX, .35+1.18*9/16*(.5-231.4453125/562.5), -.511); scene.add(targetRing);
    const targetGlow = new THREE.Mesh(new THREE.RingGeometry(.054, .058, 48),
        new THREE.MeshBasicMaterial({color: 0x6fd9ff, transparent: true, opacity: .45,
            side: THREE.DoubleSide, toneMapped: false}));
    targetGlow.position.set(hitX, .35+1.18*9/16*(.5-231.4453125/562.5), -.510); scene.add(targetGlow);

    const pads = {};
    for (const [name, color] of [
        ['D', 0x3b9ef6], ['F', 0xf75d60],
        ['J', 0xf75d60], ['K', 0x3b9ef6],
    ]) {
        const material = new THREE.MeshStandardMaterial({color, emissive: color,
            emissiveIntensity: .15, roughness: .5});
        const pad = new THREE.Mesh(new THREE.BoxGeometry(.075, .02, .075), material);
        scene.add(pad);
        const glyph = new THREE.Mesh(new THREE.PlaneGeometry(.055, .055),
            new THREE.MeshBasicMaterial({map: labelTexture(name), transparent: true,
                depthWrite: false, side: THREE.DoubleSide}));
        glyph.rotation.x = -Math.PI / 2;
        scene.add(glyph);
        pads[name] = {pad, glyph, material};
    }

    const fly = new THREE.Group(); scene.add(fly);
    const legs = {left: new THREE.Group(), right: new THREE.Group()};
    fly.add(legs.left, legs.right);
    const colors = {body: 0x9e6834, black: 0x15110e, red: 0xad331f,
        ocelli: 0xe6b351, 'bristle-brown': 0x281c10, lower: 0xbb8949,
        brown: 0x52351f};
    const materials = Object.fromEntries(Object.entries(colors).map(([key, color]) =>
        [key, new THREE.MeshStandardMaterial({color, roughness: .65})]));
    materials.membrane = new THREE.MeshStandardMaterial({color: 0xaabbcc,
        transparent: true, opacity: .4, side: THREE.DoubleSide, depthWrite: false});
    const [metaResponse, binaryResponse] = await Promise.all([
        fetch(MODEL_ROOT + 'model.json'), fetch(MODEL_ROOT + 'model.bin'),
    ]);
    if (!metaResponse.ok || !binaryResponse.ok) throw Error('Flybody mesh download failed');
    const meta = await metaResponse.json(), binary = await binaryResponse.arrayBuffer();
    if (meta.version !== 1 || !Array.isArray(meta.parts) || binary.byteLength !== 1711800)
        throw Error('Invalid Flybody mesh layout');
    const vertices = {body: [], front_left: [], front_right: []};
    for (const part of meta.parts) {
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(binary.slice(part.positionByteOffset,
            part.positionByteOffset + part.positionCount * 12));
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        for (let i = 0; i < positions.length; i += 3)
            vertices[part.group].push(new THREE.Vector3(positions[i], positions[i + 1], positions[i + 2]));
        geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(
            binary.slice(part.indexByteOffset, part.indexByteOffset + part.indexCount * 4)), 1));
        geometry.computeVertexNormals();
        const mesh = new THREE.Mesh(geometry, materials[part.material] || materials.body);
        if (part.group === 'front_left' || part.group === 'front_right') {
            const group = part.group === 'front_left' ? legs.left : legs.right;
            if (!group.children.length) group.position.fromArray(meta.pivots[part.group]);
            group.add(mesh);
        } else {
            mesh.position.fromArray(meta.pivots[part.group]); fly.add(mesh);
        }
    }
    const bounds = new THREE.Box3().setFromObject(fly);
    fly.position.sub(bounds.getCenter(new THREE.Vector3()));
    fly.rotation.y = Math.PI;
    fly.position.y -= .02;
    fly.position.z += .35;
    const localFeet = footTargets(vertices, meta.pivots);
    fly.updateMatrixWorld(true);
    const footWorld = Object.fromEntries(Object.entries(localFeet).map(([key, point]) =>
        [key, fly.localToWorld(point.clone())]));
    for (const [key, {pad, glyph}] of Object.entries(pads)) {
        const foot = footWorld[key];
        pad.position.set(foot.x, foot.y - .01, foot.z);
        glyph.position.set(foot.x, foot.y + .001, foot.z);
    }
    const restLegY = {left: legs.left.position.y, right: legs.right.position.y};
    const restBodyY = fly.position.y;
    // The mesh's middle legs are fused to the thorax. A subtle whole-body roll
    // makes their tarsi reach the Kat pads without inventing a rigid leg joint.
    const katRollSign = Math.sign(footWorld.D.x - footWorld.K.x) || -1;
    let bodyRoll = 0, lastPoseTime = null;
    const radius = bounds.getBoundingSphere(new THREE.Sphere()).radius;
    if (radius < .04 || radius > 2) throw Error('Unexpected Flybody mesh dimensions');
    status.textContent = 'Flybody 3D · drag to rotate';

    const resize = () => {
        const width = Math.max(1, host.clientWidth), height = Math.max(1, host.clientHeight);
        renderer.setSize(width, height, false);
        camera.aspect = width / height; camera.updateProjectionMatrix();
    };
    new ResizeObserver(resize).observe(host); resize();
    let drag = null;
    renderer.domElement.onpointerdown = event => {
        drag = [event.clientX, event.clientY]; renderer.domElement.setPointerCapture(event.pointerId);
    };
    renderer.domElement.onpointerup = renderer.domElement.onpointercancel = () => { drag = null; };
    renderer.domElement.onpointermove = event => {
        if (!drag) return;
        yaw += (event.clientX - drag[0]) * .006;
        pitch = Math.max(-.15, Math.min(1.1, pitch - (event.clientY - drag[1]) * .006));
        drag = [event.clientX, event.clientY]; placeCamera();
    };
    renderer.domElement.onwheel = event => {
        event.preventDefault();
        distance = Math.max(1.3, Math.min(4, distance + Math.sign(event.deltaY) * .1));
        placeCamera();
    };
    let visible = true, lastPlayerFrame = -1;
    const frameClock = {next: null}, fpsCounter = {start: null, frames: 0};
    new IntersectionObserver(entries => {
        visible = entries[0]?.isIntersecting ?? true;
        if (!visible) { frameClock.next = null; fpsCounter.start = null; fpsCounter.frames = 0;
            if (window.flyTaikoScene) window.flyTaikoScene.displayFps = 0; }
    }).observe(host);
    function draw(now) {
        requestAnimationFrame(draw);
        if (!visible || !frameDue(now, frameClock)) return;
        const page = viewer.contentWindow;
        const state = page?.taikoState;
        const canvas = page?.document?.getElementById('game-canvas');
        if (canvas && page.document.body.classList.contains('recorded-mode')) {
            if (screenTexture.image !== canvas || textureWidth !== canvas.width || textureHeight !== canvas.height) {
                const old = screenTexture;
                screenTexture = new THREE.CanvasTexture(canvas);
                screenTexture.colorSpace = THREE.SRGBColorSpace;
                screenTexture.minFilter = THREE.LinearFilter;
                screen.material.map = screenTexture; screen.material.needsUpdate = true;
                old.dispose(); lastPlayerFrame = -1;
                textureWidth=canvas.width;textureHeight=canvas.height;
            }
            const currentFrame = page.recordedReplaySnapshot?.()?.rendered_frames ?? state?.frameCount;
            if (lastPlayerFrame !== currentFrame) {
                screenTexture.needsUpdate = true;
                lastPlayerFrame = currentFrame ?? -1;
            }
        }
        const force = recordedForces();
        const dt = lastPoseTime === null ? 1 / 60 : Math.min(.05, (now - lastPoseTime) / 1000);
        lastPoseTime = now;
        // The fly is turned 180° around Y above, reversing the world-space
        // effect of its local Z rotation. D must lean toward D, K toward K.
        const targetRoll = katRollSign * (force.D - force.K) * .045;
        bodyRoll += (targetRoll - bodyRoll) * (1 - Math.exp(-dt * 15));
        fly.rotation.z = bodyRoll;
        fly.position.y = restBodyY - Math.abs(bodyRoll) * .045;
        for (const [key, {pad, glyph, material}] of Object.entries(pads)) {
            const active = force[key];
            const displacement = active * (key === 'F' || key === 'J' ? .011 : .006);
            pad.position.y = footWorld[key].y - .01 - displacement;
            glyph.position.y = footWorld[key].y + .001 - displacement;
            material.emissiveIntensity = .15 + active * 1.8;
        }
        legs.left.position.y = restLegY.left - force.F * .011;
        legs.right.position.y = restLegY.right - force.J * .011;
        updateOverlay(now);
        renderer.render(scene, camera);
        recordSceneFrame(now, fpsCounter);
    }
    requestAnimationFrame(draw);
    window.flyTaikoScene = {ready: true, targetFps: 120, displayFps: 0, modelParts: meta.parts.length,
        footTargets: Object.fromEntries(Object.entries(footWorld).map(([key, foot]) =>
            [key, [foot.x, foot.y, foot.z]])),
        getKeyForce: () => Object.fromEntries(Object.entries(pads).map(([key, item]) =>
            [key, item.material.emissiveIntensity]))};
}

async function fallback(originalError) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) throw originalError;
    host.append(canvas);
    const [metaResponse, binaryResponse] = await Promise.all([
        fetch(MODEL_ROOT + 'model.json'), fetch(MODEL_ROOT + 'model.bin'),
    ]);
    if (!metaResponse.ok || !binaryResponse.ok) throw originalError;
    const meta = await metaResponse.json(), binary = await binaryResponse.arrayBuffer();
    if (binary.byteLength !== 1711800 || !Array.isArray(meta.parts)) throw originalError;
    const palette = {body: '#956950', black: '#241b23', red: '#a64b37',
        ocelli: '#edba69', 'bristle-brown': '#443027', lower: '#ae825e',
        brown: '#674e40', membrane: 'rgba(176,207,241,.34)'};
    const points = [], vertices = {body: [], front_left: [], front_right: []};
    for (const part of meta.parts) {
        const positions = new Float32Array(binary.slice(part.positionByteOffset,
            part.positionByteOffset + part.positionCount * 12));
        const stride = Math.max(1, Math.ceil(part.positionCount / 1400));
        for (let i = 0; i < part.positionCount; i++) {
            const p = new THREE.Vector3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
            vertices[part.group].push(p);
            if (i % stride === 0) points.push({x: p.x, y: p.y, z: p.z,
                group: part.group, color: palette[part.material] || palette.body});
        }
    }
    const feet = footTargets(vertices, meta.pivots);
    status.textContent = 'Flybody mesh · 2D fallback (WebGL unavailable)';
    let visible = true, projectedPoints = [], projectedWidth = 0, projectedHeight = 0;
    const frameClock = {next: null}, fpsCounter = {start: null, frames: 0};
    new IntersectionObserver(entries => {
        visible = entries[0]?.isIntersecting ?? true;
        if (!visible) { frameClock.next = null; fpsCounter.start = null; fpsCounter.frames = 0;
            if (window.flyTaikoScene) window.flyTaikoScene.displayFps = 0; }
    }).observe(host);
    function draw(now) {
        requestAnimationFrame(draw);
        if (!visible || !frameDue(now, frameClock)) return;
        const w = Math.max(1, host.clientWidth), h = Math.max(1, host.clientHeight);
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
        ctx.fillStyle = '#111827'; ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = '#28364b'; ctx.lineWidth = 1;
        for (let x = -w; x < w * 2; x += 44) {
            ctx.beginPath(); ctx.moveTo(x, h); ctx.lineTo(w / 2 + (x - w / 2) * .28, h * .48); ctx.stroke();
        }
        for (let y = h * .56; y < h; y += 35) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
        }
        const sw = Math.min(w * .68, h * 1.2), sh = sw * 9/16;
        const sx = (w - sw) / 2, sy = h * .08;
        ctx.fillStyle = '#060910'; ctx.fillRect(sx - 8, sy - 8, sw + 16, sh + 16);
        const game = viewer.contentWindow?.document?.getElementById('game-canvas');
        if (game && viewer.contentWindow.document.body.classList.contains('recorded-mode'))
            ctx.drawImage(game, sx, sy, sw, sh);
        else { ctx.fillStyle = '#172130'; ctx.fillRect(sx, sy, sw, sh); }
        const hitX = sx + sw * .1875, hitY = sy + sh * (231.4453125/562.5);
        ctx.strokeStyle = '#f0f7ff'; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(hitX, hitY, Math.max(9, sw * .04), 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = '#0b111c'; ctx.fillRect(w / 2 - 6, sy + sh + 8, 12, 26);
        const force = recordedForces();
        const px = w / 2, py = h * .73, scale = Math.min(w * .75, h * 1.35);
        function project(p) {
            const x = -p.x, x2 = x * .84 - p.z * .54, depth = x * .54 + p.z * .84;
            return [px + x2 * scale, py - (p.y + depth * .22) * scale];
        }
        if (projectedWidth !== w || projectedHeight !== h) {
            projectedWidth = w; projectedHeight = h;
            projectedPoints = points.map(point => {
                const pivot = meta.pivots[point.group] || [0, 0, 0];
                const [x, y] = project({x: point.x + pivot[0], y: point.y + pivot[1],
                    z: point.z + pivot[2]});
                return {x, y, group: point.group, color: point.color};
            });
        }
        for (const [key, p] of Object.entries(feet)) {
            const [x, y] = project(p);
            const size = Math.max(28, Math.min(42, scale * .07));
            ctx.fillStyle = key === 'D' || key === 'K' ? '#247cbd' : '#bf414b';
            ctx.fillRect(x - size / 2, y, size, 23);
            ctx.strokeStyle = force[key] ? '#fff6be' : '#607488';
            ctx.lineWidth = force[key] ? 3 : 1; ctx.strokeRect(x - size / 2, y, size, 23);
            ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.font = 'bold 14px system-ui';
            ctx.fillText(key, x, y + 16);
        }
        for (const point of projectedPoints) {
            const shift = point.group === 'front_left' ? force.F :
                point.group === 'front_right' ? force.J : 0;
            ctx.fillStyle = point.color;
            ctx.fillRect(point.x, point.y + shift * .011 * scale,
                point.group === 'body' ? 2.5 : 2, point.group === 'body' ? 2.5 : 2);
        }
        updateOverlay(now);
        recordSceneFrame(now, fpsCounter);
    }
    requestAnimationFrame(draw);
    window.flyTaikoScene = {ready: true, mode: 'canvas2d', targetFps: 120, displayFps: 0,
        modelParts: meta.parts.length,
        footTargets: Object.fromEntries(Object.entries(feet).map(([key, p]) =>
            [key, [p.x, p.y, p.z]])),
        getKeyForce: recordedForces};
}

start().catch(error => fallback(error).catch(fallbackError => {
    status.textContent = 'Could not display Flybody: ' + fallbackError.message;
}));
