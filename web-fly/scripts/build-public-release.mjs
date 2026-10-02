// Build only the Taiko web client. Uploaded maps, music, replays, and model
// weights stay on the separately hosted GPU service, never in the Git deploy.
import {cp, mkdir, readFile, rm, writeFile} from 'node:fs/promises';
import {join} from 'node:path';

const source = 'public';
const output = 'dist-public';
const rawBackend = process.env.FLYTAIKO_BACKEND_URL;
if (!rawBackend) throw Error('Set FLYTAIKO_BACKEND_URL to the public HTTPS GPU API origin');
const url = new URL(rawBackend);
if ((url.protocol !== 'https:' && !(url.protocol === 'http:' &&
    ['localhost', '127.0.0.1'].includes(url.hostname))) ||
    url.username || url.password || url.pathname !== '/' || url.search || url.hash)
    throw Error('FLYTAIKO_BACKEND_URL must be an HTTPS origin (localhost HTTP for testing)');
const backend = url.origin;

await rm(output, {recursive: true, force: true});
await mkdir(output, {recursive: true});
for (const name of [
    'index.html', 'malecns-taiko.html', 'create-replay.html',
    'taiko_viewer.html', 'malecns_taiko_demo.js',
    'fly_taiko_scene.js', 'recorded_taiko_player.js', 'replay_upload.js',
]) await cp(join(source, name), join(output, name));
for (const directory of ['vendor', 'data/flybody'])
    await cp(join(source, directory), join(output, directory), {recursive: true});
await writeFile(join(output, 'runtime-config.js'),
                `window.FLYTAIKO_BACKEND_URL = ${JSON.stringify(backend)};\n`);

// The personal osu! skin is not redistributed in the public package.
const playerPath = join(output, 'malecns-taiko.html');
const player = await readFile(playerPath, 'utf8');
await writeFile(playerPath, player
    .replace('<option value="koishi">Koishi Komeiji</option>', '')
    .replace(/<section class="input-panel panel"><h2>Ảnh đầu vào nét cao v22 · pilot<\/h2>[\s\S]*?<\/section>/, '')
    .replace('href="map-browser.html"', `href="${backend}/map-browser.html"`)
    .replace('href="sv-ideology.html"', `href="${backend}/sv-ideology.html"`));
const files = ['index.html', 'malecns-taiko.html', 'create-replay.html', 'taiko_viewer.html'];
for (const file of files) {
    const body = await readFile(join(output, file), 'utf8');
    if (body.includes('skins/koishi')) throw Error(`Private skin leaked into ${file}`);
}
console.log(`Built ${output} for ${backend}; no uploaded data, caches, audio, or weights included.`);
