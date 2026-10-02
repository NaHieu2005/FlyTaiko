/* Presentation only. No neural inference, new judgments, or timestamp snapping.
 * Legacy OSZ/replay player stays unchanged until loadRecordedReplay is called. */
(() => {
    'use strict';
    const canvas = document.getElementById('game-canvas');
    const ctx = canvas.getContext('2d');
    const $ = id => document.getElementById(id);
    const HIT = 150, Y = 150, SCALE = (1000 - HIT) / (512 - 64);
    const KEY_NAMES = [[], ['F'], ['J'], ['D'], ['K'], ['F', 'J'], ['D', 'K']];
    const COLORS = {great: '#21d6ef', good: '#49e900', miss: '#e5b43f'};
    const legacyStart = window.startGame;
    let replay = null, source = null, position = 0, clockBase = 0, audioBase = 0;
    let playing = false, starting = false, generation = 0;
    let eventIndex = 0, actionIndex = 0, counts, combo = 0, maxCombo = 0;
    let sumError = 0, errorCount = 0, offsets = [], lastEvent = null;
    let falseHits = 0, longHits = new Map(), longResults = new Map(), lights = {};
    let lastKeyTime = {D: -Infinity, F: -Infinity, J: -Infinity, K: -Infinity};
    let raf = 0, nextFrame = null, fpsStart = null, fpsFrames = 0, displayFps = 0, frameCount = 0;
    let statsDirty = false, needsDraw = true;
    let skinName = 'default', skinSprites = null, hitVolume = .55;
    let skinSounds = null;
    const KOISHI_ROOT = 'skins/koishi/';
    const KOISHI_IMAGES = {
        barRight: 'taiko-bar-right.png',
        hit: 'taikohitcircle@2x.png', hitOverlay: 'taikohitcircleoverlay@2x.png',
        big: 'taikobigcircle@2x.png', bigOverlay: 'taikobigcircleoverlay@2x.png',
        rollMiddle: 'taiko-roll-middle@2x.png', rollEnd: 'taiko-roll-end@2x.png',
        spinner: 'spinner-circle@2x.png',
    };

    function loadImage(path) {
        return new Promise((resolve, reject) => {
            const image = new Image();
            image.onload = () => resolve(image);
            image.onerror = () => reject(Error(`Skin asset failed: ${path}`));
            image.src = path;
        });
    }
    // The supplied @2x bases have opaque white backgrounds and grey artwork.
    // Remove white before tinting, retaining antialiased edge coverage.
    function tintSprite(image, color) {
        const surface = document.createElement('canvas');
        surface.width = image.naturalWidth; surface.height = image.naturalHeight;
        const paint = surface.getContext('2d');
        paint.drawImage(image, 0, 0);
        const pixels = paint.getImageData(0, 0, surface.width, surface.height);
        const rgb = color === 'don' ? [240, 75, 88] : color === 'kat' ? [68, 174, 244] : [230, 180, 25];
        for (let i = 0; i < pixels.data.length; i += 4) {
            const darkness = 255 - Math.min(pixels.data[i], pixels.data[i + 1], pixels.data[i + 2]);
            pixels.data[i + 3] = Math.round(pixels.data[i + 3] * Math.min(1, darkness / 70));
            pixels.data[i] = rgb[0]; pixels.data[i + 1] = rgb[1]; pixels.data[i + 2] = rgb[2];
        }
        paint.putImageData(pixels, 0, 0);
        return surface;
    }
    window.setRecordedTaikoSkin = async function (name) {
        if (name !== 'default' && name !== 'koishi') throw Error('Unknown Taiko skin');
        if (name === 'koishi' && !skinSprites) {
            const entries = await Promise.all(Object.entries(KOISHI_IMAGES).map(async ([key, filename]) =>
                [key, await loadImage(KOISHI_ROOT + filename)]));
            const loaded = Object.fromEntries(entries);
            for (const size of ['hit', 'big']) for (const color of ['don', 'kat', 'roll'])
                loaded[`${size}_${color}`] = tintSprite(loaded[size], color);
            loaded.rollMiddleTint = tintSprite(loaded.rollMiddle, 'roll');
            loaded.rollEndTint = tintSprite(loaded.rollEnd, 'roll');
            skinSprites = loaded;
        }
        skinName = name;
        if (name === 'koishi' && !skinSounds) {
            const loadSound = async file => {
                const response = await fetch(KOISHI_ROOT + file);
                if (!response.ok) throw Error(`Skin hitsound HTTP ${response.status}`);
                return audioContext.decodeAudioData(await response.arrayBuffer());
            };
            skinSounds = {
                don: await loadSound('taiko-drum-hitnormal.ogg'),
                kat: await loadSound('taiko-drum-hitclap.ogg'),
            };
        }
        document.body.classList.toggle('koishi-skin', name === 'koishi');
        schedule();
        if (replay && !playing) { draw(position); needsDraw = false; }
    };
    window.setRecordedHitVolume = value => { hitVolume = Math.max(0, Math.min(1, Number(value) || 0)); };
    function defaultHitSound(kind) {
        const length = Math.round(audioContext.sampleRate * .09);
        const buffer = audioContext.createBuffer(1, length, audioContext.sampleRate);
        const samples = buffer.getChannelData(0), frequency = kind === 'kat' ? 510 : 180;
        for (let i = 0; i < length; i++) {
            const t = i / audioContext.sampleRate;
            const noise = Math.sin(i * 12.9898) * Math.sin(i * 78.233);
            samples[i] = (Math.sin(2 * Math.PI * frequency * t * (1 - t * 1.5)) * .65 + noise * .18)
                * Math.exp(-t * (kind === 'kat' ? 48 : 36));
        }
        return buffer;
    }
    const defaultSounds = {};
    function playHit(action) {
        if (!playing || hitVolume <= 0) return;
        const kat = action.action_id === 3 || action.action_id === 4 || action.action_id === 6;
        const kind = kat ? 'kat' : 'don';
        const buffer = skinName === 'koishi' ? skinSounds?.[kind] :
            (defaultSounds[kind] ||= defaultHitSound(kind));
        if (!buffer) return;
        const voice = audioContext.createBufferSource(), level = audioContext.createGain();
        voice.buffer = buffer;
        level.gain.value = hitVolume * (action.action_id >= 5 ? .8 : .65);
        voice.connect(level); level.connect(audioContext.destination);
        voice.start(Math.max(audioContext.currentTime, audioBase +
            (action.time_ms - clockBase) / (1000 * (replay.clock_rate || 1))));
        voice.onended = () => { voice.disconnect(); level.disconnect(); };
    }

    // Keep the same approach time as the native 512px observation viewport.
    // A note retains its own parsed native speed; later SV points do not retime it.
    function noteX(note, time, endpoint = false) {
        const speed = note.scroll_px_per_ms > 0 ? note.scroll_px_per_ms : 448 / 1200;
        return HIT + ((endpoint ? note.end_t : note.t) - time) * speed * SCALE;
    }
    // Phase accumulator avoids accidentally halving FPS at a nominal 120Hz.
    function frameDeadline(now, next, fps = 120) {
        if (next === null || now - next > 1000) return {draw: true, next: now + 1000 / fps};
        if (now + .25 < next) return {draw: false, next};
        return {draw: true, next: next + (Math.floor(Math.max(0, now - next) / (1000 / fps)) + 1) * 1000 / fps};
    }
    window.recordedTaikoGeometry = {noteX, frameDeadline, hitX: HIT, viewportScale: SCALE};

    function clock() {
        if (!playing) return position;
        return Math.min(replay.duration_ms, clockBase +
            (audioContext.currentTime - audioBase) * 1000 * (replay.clock_rate || 1));
    }
    function stopSource() {
        if (source) { source.onended = null; try { source.stop(); } catch (_) {} source.disconnect(); source = null; }
    }
    function resetStats() {
        eventIndex = actionIndex = combo = maxCombo = sumError = errorCount = falseHits = 0;
        counts = {great: 0, good: 0, miss: 0}; offsets = []; lastEvent = null;
        $('ur-ticks').replaceChildren();
        longHits = new Map(); lights = {};
        lastKeyTime = {D: -Infinity, F: -Infinity, J: -Infinity, K: -Infinity};
        statsDirty = true;
    }
    function consume(time) {
        while (eventIndex < replay.events.length && replay.events[eventIndex].time_ms <= time) {
            const event = replay.events[eventIndex++];
            counts[event.judgment]++;
            combo = event.judgment === 'miss' ? 0 : combo + 1;
            maxCombo = Math.max(maxCombo, combo);
            if (event.judgment !== 'miss' && Number.isFinite(event.timing_error_ms)) {
                sumError += event.timing_error_ms; errorCount++;
                const tick = document.createElement('div'); tick.className = 'ur-tick';
                tick.style.left = Math.max(0, Math.min(100, (event.timing_error_ms + 150) / 3)) + '%';
                tick.style.backgroundColor = COLORS[event.judgment];
                $('ur-ticks').append(tick);
                offsets.push({error: event.timing_error_ms, time: event.time_ms, tick});
                if (offsets.length > 32) offsets.shift().tick.remove();
            }
            lastEvent = event; statsDirty = true;
        }
        while (actionIndex < replay.actions.length && replay.actions[actionIndex].time_ms <= time) {
            const action = replay.actions[actionIndex++];
            playHit(action);
            for (const key of KEY_NAMES[action.action_id]) {
                lights[key] = action.time_ms + 70;
                lastKeyTime[key] = action.time_ms;
            }
            if (action.judgment === 'swell_tick' || action.judgment === 'drumroll_tick') {
                longHits.set(action.object_index, (longHits.get(action.object_index) || 0) + 1);
                statsDirty = true;
            } else if (action.judgment === 'ignore' && action.object_index === undefined) {
                falseHits++; statsDirty = true;
            }
        }
    }
    function longStats(kind) {
        const rows = replay.notes.filter(n => n.type === kind);
        let hits = 0, required = 0, completed = 0;
        for (const note of rows) {
            const count = longHits.get(note.object_index) || 0;
            const need = longResults.get(note.object_index)?.required || note.required_hits || 0;
            hits += count; required += need; completed += Number(need > 0 && count >= need);
        }
        return {objects: rows.length, completed, hits, required};
    }
    function updateHUD(time) {
        if (statsDirty) {
            for (const name of ['great', 'good', 'miss']) $('stat-' + name + '-val').textContent = counts[name];
            const total = counts.great + counts.good + counts.miss;
            $('stat-acc-val').textContent = (total ? 100 * (counts.great + .5 * counts.good) / total : 100).toFixed(2) + '%';
            $('combo-text').textContent = combo + 'x';
            const avg = errorCount ? sumError / errorCount : 0;
            $('stat-error-val').textContent = (avg > 0 ? '+' : '') + avg.toFixed(1) + 'ms';
            const arrow = $('ur-avg-arrow'); arrow.style.display = offsets.length ? 'block' : 'none';
            if (offsets.length) arrow.style.left = Math.max(0, Math.min(100, (offsets.reduce((s, o) => s + o.error, 0) / offsets.length + 150) / 3)) + '%';
            const roll = longStats('drumroll'), swell = longStats('swell');
            $('long-status').textContent = `Drumroll ${roll.completed}/${roll.objects} · Spinner ${swell.completed}/${swell.objects} · Extra ${falseHits}`;
            statsDirty = false;
        }
        for (const item of offsets) item.tick.style.opacity = Math.max(0, 1 - (time - item.time) / 1800);
        while (offsets.length && time - offsets[0].time >= 1800) offsets.shift().tick.remove();
        if (!offsets.length) $('ur-avg-arrow').style.display = 'none';
        const judgment = $('judgment-text');
        judgment.style.opacity = lastEvent && time - lastEvent.time_ms < 250 ? 1 : 0;
        if (lastEvent) { judgment.textContent = lastEvent.judgment.toUpperCase(); judgment.style.color = COLORS[lastEvent.judgment]; }
        for (const key of ['D', 'F', 'J', 'K']) {
            const el = $('key-' + key.toLowerCase()), active = lights[key] > time;
            el.classList.toggle('active-kat', active && (key === 'D' || key === 'K'));
            el.classList.toggle('active-don', active && (key === 'F' || key === 'J'));
        }
    }
    function circle(x, radius, color, big = false) {
        if (skinName === 'koishi' && skinSprites) {
            // Keep one on-screen note geometry for every selectable skin.
            const size = radius * 2;
            const kind = big ? 'big' : 'hit';
            const ink = color === '#5ab4f0' ? 'kat' : color === '#e6b419' ? 'roll' : 'don';
            ctx.drawImage(skinSprites[`${kind}_${ink}`], x - size / 2, Y - size / 2, size, size);
            ctx.drawImage(skinSprites[`${kind}Overlay`], x - size / 2, Y - size / 2, size, size);
            return;
        }
        ctx.beginPath(); ctx.arc(x, Y, radius, 0, Math.PI * 2);
        ctx.fillStyle = color; ctx.fill(); ctx.lineWidth = 3; ctx.strokeStyle = '#fff'; ctx.stroke();
    }
    function drawKeys(time) {
        // Independent D/F/J/K indicators live above the lane, away from
        // scrolling notes, the judgment circle and the bottom-left combo.
        ctx.fillStyle = 'rgba(9,12,20,.9)'; ctx.fillRect(6, 4, 184, 43);
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.font = 'bold 17px Segoe UI';
        for (const [index, key] of ['D', 'F', 'J', 'K'].entries()) {
            const x = 12 + index * 44, age = time - lastKeyTime[key];
            const pulse = Math.max(0, 1 - age / 180);
            ctx.fillStyle = '#242b37'; ctx.fillRect(x, 9, 40, 32);
            ctx.globalAlpha = .22 + .78 * pulse;
            ctx.fillStyle = key === 'D' || key === 'K' ? '#389ff5' : '#f35259';
            ctx.fillRect(x + 2, 11, 36, 28);
            ctx.globalAlpha = 1;
            ctx.fillStyle = '#fff'; ctx.fillText(key, x + 20, 26);
        }
        ctx.textBaseline = 'alphabetic';
    }
    function draw(time) {
        ctx.clearRect(0, 0, 1000, 300);
        if (skinName === 'koishi' && skinSprites) {
            ctx.fillStyle = '#130d16'; ctx.fillRect(0, 0, 1000, 300);
            // Keep the uploaded lane texture. Its 180px left drum artwork is
            // deliberately not overlaid on the note path: four keys are shown
            // separately below the lane for this replay player.
            ctx.drawImage(skinSprites.barRight, 0, 50, 1000, 200);
        } else {
            ctx.fillStyle = '#222530'; ctx.fillRect(0, 110, 1000, 80);
            ctx.strokeStyle = '#4b5163'; ctx.lineWidth = 1;
            for (const y of [110, 190]) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(1000, y); ctx.stroke(); }
        }
        ctx.font = 'bold 16px Segoe UI'; ctx.textAlign = 'right'; ctx.fillStyle = '#aab6c8';
        ctx.fillText('NATIVE SV · RECORDED KEYS', 978, 28);
        // Back-to-front matches the original player; evaluate nothing here.
        for (let i = replay.notes.length - 1; i >= 0; i--) {
            const note = replay.notes[i], x = noteX(note, time);
            if (note.circle_index !== undefined) {
                if (time >= note.judged_at_ms || x < -90 || x > 1090) continue;
                const big = note.type.includes('big');
                circle(x, big ? 45 : 30, note.type.startsWith('kat') ? '#5ab4f0' : '#f05a5a', big);
            } else {
                const end = noteX(note, time, true);
                if (note.type === 'swell') {
                    if (time > note.end_t || x > 1100) continue;
                    const active = time >= note.t, centre = active ? HIT : x;
                    if (centre < -85) continue;
                    const hits = longHits.get(note.object_index) || 0;
                    const required = longResults.get(note.object_index)?.required || note.required_hits || 1;
                    const progress = Math.min(1, hits / required);
                    ctx.save();
                    if (skinName === 'koishi' && skinSprites) {
                        ctx.drawImage(skinSprites.spinner, centre - 76, Y - 76, 152, 152);
                    } else {
                        ctx.fillStyle = 'rgba(145,65,215,.42)'; ctx.strokeStyle = '#eee9ff'; ctx.lineWidth = 5;
                        ctx.beginPath(); ctx.arc(centre, Y, 66, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
                    }
                    ctx.strokeStyle = progress >= 1 ? '#75eb96' : '#d1a2ff'; ctx.lineWidth = 10;
                    ctx.beginPath(); ctx.arc(centre, Y, 76, -Math.PI / 2,
                        -Math.PI / 2 + 2 * Math.PI * Math.max(progress, .003)); ctx.stroke();
                    ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.font = 'bold 22px Segoe UI';
                    ctx.fillText('SPIN', centre, Y - 3);
                    ctx.font = 'bold 16px Segoe UI'; ctx.fillText(active ? `${hits}/${required}` : 'D ↔ F', centre, Y + 24);
                    ctx.restore();
                    continue;
                }
                if (time > note.end_t + (note.tick_spacing_ms || 0) / 2 || end < HIT - 50 || x > 1060) continue;
                const left = Math.max(HIT, x), right = Math.min(1060, end);
                const color = '#e6b419';
                if (skinName === 'koishi' && skinSprites) {
                    ctx.fillStyle = color; ctx.fillRect(left, Y - 59, Math.max(0, right - left), 118);
                    ctx.drawImage(skinSprites.rollMiddleTint, left, Y - 59, Math.max(0, right - left), 118);
                    if (end >= HIT && end <= 1060) ctx.drawImage(skinSprites.rollEndTint, end - 59, Y - 59, 59, 118);
                } else {
                    ctx.fillStyle = color; ctx.fillRect(left, Y - 25, Math.max(0, right - left), 50);
                    ctx.strokeStyle = '#eee'; ctx.strokeRect(left, Y - 25, Math.max(0, right - left), 50);
                }
                if (note.tick_spacing_ms > 0) {
                    const speed = (note.scroll_px_per_ms || 448 / 1200) * SCALE;
                    const first = Math.max(0, Math.ceil((time - note.t) / note.tick_spacing_ms));
                    const last = Math.min(Math.round((note.end_t - note.t) / note.tick_spacing_ms),
                        Math.floor((time + (1000 - HIT) / speed - note.t) / note.tick_spacing_ms));
                    ctx.fillStyle = '#fff3b7';
                    for (let k = first; k <= last; k++) {
                        const tx = HIT + (note.t + k * note.tick_spacing_ms - time) * speed;
                        if (tx >= HIT) ctx.fillRect(tx - 2, Y - 15, 4, 30);
                    }
                }
                if (x >= HIT - 50) circle(x, note.strong ? 45 : 30, color, note.strong);
                if (time >= note.t && time <= note.end_t) {
                    const hits = longHits.get(note.object_index) || 0;
                    const required = longResults.get(note.object_index)?.required || note.required_hits || 0;
                    ctx.fillStyle = hits >= required ? '#75eb96' : '#eee'; ctx.textAlign = 'left';
                    ctx.fillText(`DRUMROLL  ${hits}/${required}`, 220, 245);
                    ctx.fillRect(220, 268, Math.min(1, hits / Math.max(1, required)) * 350, 6);
                }
            }
        }
        drawKeys(time);
        updateHUD(time); frameCount++;
    }
    function loop(now) {
        raf = 0;
        const deadline = frameDeadline(now, nextFrame);
        if (deadline.draw && (playing || needsDraw)) {
            nextFrame = deadline.next; position = clock(); consume(position); draw(position); needsDraw = false;
            if (playing) {
                if (fpsStart === null) { fpsStart = now; fpsFrames = 0; }
                fpsFrames++;
                if (now - fpsStart >= 1000) { displayFps = fpsFrames * 1000 / (now - fpsStart); fpsStart = now; fpsFrames = 0; }
            }
            Object.assign(window.taikoState, {currentTimeMs: position, isPlaying: playing, displayFps, frameCount});
            if (playing && position >= replay.duration_ms) {
                playing = false; stopSource(); window.taikoState.isPlaying = false;
                window.parent.postMessage({type: 'TAIKO_RECORDED_ENDED'}, location.origin);
            }
        }
        if (playing || needsDraw) raf = requestAnimationFrame(loop);
    }
    function schedule() { needsDraw = true; nextFrame = null; if (!raf) raf = requestAnimationFrame(loop); }
    function validate(data) {
        if (data.schema_version !== 1 || !Number.isFinite(data.duration_ms) || data.duration_ms <= 0) throw Error('Invalid recorded replay schema');
        if (!Number.isFinite(data.clock_rate ?? 1) || (data.clock_rate ?? 1) < .5 || (data.clock_rate ?? 1) > 2)
            throw Error('Invalid replay clock rate');
        for (const rows of [data.events, data.actions]) {
            if (!Array.isArray(rows) || rows.some((r, i) => !Number.isFinite(r.time_ms) || (i > 0 && r.time_ms < rows[i - 1].time_ms))) throw Error('Unordered recorded timeline');
        }
        if (data.events.some((e, i) => e.note_index !== i || !COLORS[e.judgment])) throw Error('Invalid canonical note events');
        if (data.actions.some(a => !Number.isInteger(a.action_id) || a.action_id < 1 || a.action_id > 6)) throw Error('Invalid recorded key');
        const circles = data.notes.filter(n => n.circle_index !== undefined);
        if (circles.length !== data.events.length || circles.some((n, i) => n.circle_index !== i || n.judged_at_ms !== data.events[i].time_ms)) throw Error('Invalid canonical note indices');
        for (const name of ['great', 'good', 'miss']) if (data.events.filter(e => e.judgment === name).length !== data.metrics[name]) throw Error('Recorded metric mismatch');
    }
    window.loadRecordedReplay = function (data, buffer) {
        validate(data); window.pauseGame(); generation++; replay = data; audioBuffer = buffer;
        position = 0; resetStats(); longResults = new Map(data.long_results.map(r => [r.object_index, r]));
        document.body.classList.add('recorded-mode'); document.querySelector('h1').textContent = 'FlyTaiko · taiko replay';
        if (!$('long-status')) { const el = document.createElement('div'); el.id = 'long-status'; el.className = 'stat-acc'; $('stats-container').append(el); }
        Object.assign(window.taikoState, {kind: 'recorded-male-cns', currentTimeMs: 0,
            replayActions: data.actions, targetFps: 120, displayFps: 0, isPlaying: false});
        displayFps = 0; frameCount = 0; fpsStart = null; schedule();
        window.parent.postMessage({type: 'TAIKO_RECORDED_READY'}, location.origin);
    };
    window.setRecordedAudio = function (buffer) { audioBuffer = buffer; };
    window.startGame = async function () {
        if (!replay) return legacyStart();
        if (playing || starting) return;
        const mine = generation; starting = true;
        try {
            await audioContext.resume();
            if (mine !== generation) return;
            if (position >= replay.duration_ms) { position = 0; resetStats(); }
            clockBase = position; audioBase = audioContext.currentTime;
            if (audioBuffer && position < audioBuffer.duration * 1000) {
                source = audioContext.createBufferSource(); source.buffer = audioBuffer;
                source.playbackRate.value = replay.clock_rate || 1;
                source.connect(gainNode);
                source.start(audioBase, Math.max(0, position / 1000));
            }
            playing = true; window.taikoState.isPlaying = true; fpsStart = null; schedule();
        } finally { starting = false; }
    };
    window.pauseGame = function () {
        generation++;
        if (replay && playing) { position = clock(); consume(position); }
        playing = false; stopSource(); window.taikoState.isPlaying = false;
        if (replay) { window.taikoState.currentTimeMs = position; schedule(); }
    };
    window.seekGame = function (time) {
        if (!replay || !Number.isFinite(time)) return;
        const resume = playing; window.pauseGame(); position = Math.max(0, Math.min(replay.duration_ms, time));
        resetStats(); consume(position); window.taikoState.currentTimeMs = position; schedule();
        if (!resume) { draw(position); needsDraw = false; }
        if (resume && position < replay.duration_ms) window.startGame();
    };
    window.recordedReplaySnapshot = function () {
        if (!replay) return null;
        return {time_ms: position, ...counts, combo, max_combo: maxCombo, false_hits: falseHits,
            processed_actions: actionIndex, processed_events: eventIndex,
            mean_signed_error_ms: errorCount ? sumError / errorCount : null,
            swell: longStats('swell'), drumroll: longStats('drumroll'),
            target_fps: 120, display_fps: displayFps, rendered_frames: frameCount};
    };
})();
