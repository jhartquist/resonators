// --- config ---
const TEX_COLS = 1024;
const BINS_PER_SEMITONE = 5;
const LO_MIDI = 21;            // A0
const HI_MIDI = 108;           // C8
const STATS_HZ = 5;

const BIN_COUNT = (HI_MIDI - LO_MIDI + 1) * BINS_PER_SEMITONE;

function midiToHz(midi) { return 440 * Math.pow(2, (midi - 69) / 12); }
function logFreqs() {
  const out = new Float32Array(BIN_COUNT);
  for (let i = 0; i < BIN_COUNT; i++) out[i] = midiToHz(LO_MIDI + i / BINS_PER_SEMITONE);
  return out;
}

const COLORMAPS = {
  viridis:   [68,1,84, 64,67,135, 41,120,142, 32,144,140, 68,190,112, 121,209,81, 189,222,38, 253,231,37],
  magma:     [0,0,4, 24,15,61, 68,15,118, 121,28,109, 187,55,84, 233,90,52, 252,158,73, 252,253,191],
  inferno:   [0,0,4, 27,12,65, 87,15,109, 144,33,92, 199,58,47, 235,107,17, 251,180,38, 252,255,164],
  plasma:    [13,8,135, 84,2,163, 139,10,165, 185,50,137, 219,92,104, 244,136,73, 254,188,43, 240,249,33],
  grayscale: [0,0,0, 36,36,36, 73,73,73, 109,109,109, 146,146,146, 182,182,182, 219,219,219, 255,255,255],
};

// --- WebGL ---
const canvas = document.getElementById('canvas');
const gl = canvas.getContext('webgl2', {
  antialias: false,
  premultipliedAlpha: false,
  preserveDrawingBuffer: true,
});
if (!gl) throw new Error('WebGL2 required');

const VERT = `#version 300 es
  in vec2 aPosition;
  out vec2 vUV;
  void main() {
    vUV = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
  }`;
const FRAG = `#version 300 es
  precision highp float;
  uniform sampler2D uMags;
  uniform sampler2D uCmap;
  uniform int uWriteHead;
  uniform vec2 uTexSize;
  uniform float uDbMin;
  uniform float uDbMax;
  uniform vec2 uViewX;
  uniform vec2 uViewY;
  in vec2 vUV;
  out vec4 outColor;
  void main() {
    float tFrac = mix(uViewX.x, uViewX.y, vUV.x);
    float colIdx = mod(float(uWriteHead - 1) - (1.0 - tFrac) * uTexSize.x, uTexSize.x);
    float bin = mix(uViewY.x, uViewY.y, vUV.y) * uTexSize.y;
    vec2 mUV = vec2((colIdx + 0.5) / uTexSize.x, (bin + 0.5) / uTexSize.y);
    float mag = texture(uMags, mUV).r;
    float db = 20.0 * log(max(mag, 1e-12)) / log(10.0);
    float t = clamp((db - uDbMin) / (uDbMax - uDbMin), 0.0, 1.0);
    outColor = texture(uCmap, vec2(t, 0.5));
  }`;

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
const program = gl.createProgram();
gl.attachShader(program, compile(gl.VERTEX_SHADER, VERT));
gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAG));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
gl.useProgram(program);

const vbo = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
const aPosition = gl.getAttribLocation(program, 'aPosition');
gl.enableVertexAttribArray(aPosition);
gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0);

const u = {
  mags: gl.getUniformLocation(program, 'uMags'),
  cmap: gl.getUniformLocation(program, 'uCmap'),
  writeHead: gl.getUniformLocation(program, 'uWriteHead'),
  texSize: gl.getUniformLocation(program, 'uTexSize'),
  dbMin: gl.getUniformLocation(program, 'uDbMin'),
  dbMax: gl.getUniformLocation(program, 'uDbMax'),
  viewX: gl.getUniformLocation(program, 'uViewX'),
  viewY: gl.getUniformLocation(program, 'uViewY'),
};
gl.uniform1i(u.mags, 0);
gl.uniform1i(u.cmap, 1);

const linearFloat = !!gl.getExtension('OES_texture_float_linear');
const magFilter = linearFloat ? gl.LINEAR : gl.NEAREST;

const magTex = gl.createTexture();
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, magTex);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, TEX_COLS, BIN_COUNT, 0, gl.RED, gl.FLOAT, null);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, magFilter);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, magFilter);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

const cmapTex = gl.createTexture();
gl.activeTexture(gl.TEXTURE1);
gl.bindTexture(gl.TEXTURE_2D, cmapTex);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

function setColormap(name) {
  const data = new Uint8Array(COLORMAPS[name]);
  const stops = data.length / 3;
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, cmapTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, stops, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, data);
}
setColormap('magma');

// --- ring buffer state ---
let writeHead = 0;
let floorDb = -80;
let rangeDb = 40;
let paused = false;
const pending = [];
let dspUs = 0;
let peakDb = -Infinity;

// --- mic + worklet ---
let audioCtx = null, workletNode = null, micStream = null, micActive = false;

async function startMic() {
  if (micActive) return;
  document.activeElement?.blur();
  document.getElementById('landingNote').innerHTML = 'requesting microphone access…';
  try {
    audioCtx = new AudioContext();
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    const wasmModule = await WebAssembly.compileStreaming(fetch('./pkg/resonators_bg.wasm'));
    await audioCtx.audioWorklet.addModule('./worklet.js');
    workletNode = new AudioWorkletNode(audioCtx, 'resonators-processor', {
      processorOptions: {
        wasmModule,
        sampleRate: audioCtx.sampleRate,
        freqs: logFreqs(),
        tauScale: 2.0,
      },
    });
    workletNode.port.onmessage = (e) => {
      if (e.data.ready) return;
      if (paused) return;
      const { mags, dspUs: us, peak: p } = e.data;
      pending.push(mags);
      if (pending.length > TEX_COLS) pending.splice(0, pending.length - TEX_COLS);
      dspUs = us;
      peakDb = p > 0 ? 20 * Math.log10(p) : -Infinity;
    };
    audioCtx.createMediaStreamSource(micStream).connect(workletNode);
    micActive = true;
    document.body.classList.add('mic-on');
    document.getElementById('landing').classList.add('hidden');
    document.getElementById('pauseToggle').disabled = false;
    document.getElementById('pauseLabel').textContent = 'Pause';
    document.getElementById('exportBtn').disabled = false;
    layoutAxis();
  } catch (err) {
    document.getElementById('landingNote').innerHTML =
      `<span class="err">${err.name === 'NotAllowedError' ? 'microphone permission denied' : 'could not start microphone'}</span>`;
  }
}

document.getElementById('startBtn').addEventListener('click', startMic);

// --- texture writes ---
function flushPending() {
  if (!pending.length) return;
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, magTex);
  for (const mags of pending) {
    gl.texSubImage2D(gl.TEXTURE_2D, 0, writeHead, 0, 1, mags.length, gl.RED, gl.FLOAT, mags);
    writeHead = (writeHead + 1) % TEX_COLS;
  }
  pending.length = 0;
}

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.floor(canvas.clientWidth * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
    gl.viewport(0, 0, w, h);
  }
}

function render() {
  resize();
  flushPending();
  gl.clearColor(0.04, 0.04, 0.043, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(program);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, magTex);
  gl.uniform1i(u.writeHead, writeHead);
  gl.uniform2f(u.texSize, TEX_COLS, BIN_COUNT);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, cmapTex);
  gl.uniform1f(u.dbMin, floorDb);
  gl.uniform1f(u.dbMax, floorDb + rangeDb);
  gl.uniform2f(u.viewX, 0, 1);
  gl.uniform2f(u.viewY, 0, 1);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

let lastStatsT = 0;
function frame(t) {
  render();
  if (t - lastStatsT > 1000 / STATS_HZ) {
    updateStats();
    lastStatsT = t;
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// --- stats ---
const srStat = document.getElementById('srStat');
const dspStat = document.getElementById('dspStat');
const peakStat = document.getElementById('peakStat');
function updateStats() {
  if (!micActive) return;
  const sr = audioCtx.sampleRate;
  const budgetUs = (128 / sr) * 1e6;
  const pct = (dspUs / budgetUs * 100);
  const peakStr = isFinite(peakDb) ? `${peakDb.toFixed(0)} dB` : '−∞ dB';
  srStat.textContent = `${(sr/1000).toFixed(0)} kHz`;
  dspStat.textContent = `${pct.toFixed(0)}% budget`;
  peakStat.textContent = `peak ${peakStr}`;
}

// --- axis labels ---
const axisEl = document.getElementById('axisLabels');
function layoutAxis() {
  axisEl.innerHTML = '';
  for (let midi = LO_MIDI; midi <= HI_MIDI; midi++) {
    if (midi % 12 !== 0) continue;
    const octave = Math.floor(midi / 12) - 1;
    const yFrac = (midi - LO_MIDI) * BINS_PER_SEMITONE / BIN_COUNT;
    const div = document.createElement('div');
    div.textContent = 'C' + octave;
    div.style.bottom = `${yFrac * 100}%`;
    axisEl.appendChild(div);
  }
}
layoutAxis();

// --- controls ---
document.getElementById('cmap').addEventListener('change', (e) => setColormap(e.target.value));

const floorEl = document.getElementById('floor');
const floorV = document.getElementById('floorV');
floorEl.addEventListener('input', () => {
  floorDb = +floorEl.value;
  floorV.textContent = `${floorDb} dB`.replace('-', '−');
});
const rangeEl = document.getElementById('range');
const rangeV = document.getElementById('rangeV');
rangeEl.addEventListener('input', () => {
  rangeDb = +rangeEl.value;
  rangeV.textContent = `${rangeDb} dB`;
});

// --- pause (spacebar + tap canvas + sidebar button) ---
const liveStat = document.getElementById('liveStat');
const pauseToggle = document.getElementById('pauseToggle');
const pauseLabel = document.getElementById('pauseLabel');
function togglePause() {
  if (!micActive) return;
  paused = !paused;
  document.body.classList.toggle('paused', paused);
  liveStat.textContent = paused ? 'paused' : 'live';
  pauseLabel.textContent = paused ? 'Play' : 'Pause';
}
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !e.target.matches('input, select, button, textarea')) {
    e.preventDefault();
    togglePause();
  }
});
canvas.addEventListener('click', togglePause);
pauseToggle.addEventListener('click', togglePause);

// --- sidebar toggle ---
const sidebar = document.getElementById('sidebar');
function setSidebar(open) {
  document.body.classList.toggle('sidebar-open', open);
  sidebar.setAttribute('aria-hidden', String(!open));
}
document.getElementById('settingsBtn').addEventListener('click', () => {
  setSidebar(!document.body.classList.contains('sidebar-open'));
});
document.getElementById('closeBtn').addEventListener('click', () => setSidebar(false));

// --- export ---
document.getElementById('exportBtn').addEventListener('click', () => {
  canvas.toBlob((blob) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `resonators-${Date.now()}.png`;
    a.click();
    URL.revokeObjectURL(a.href);
  }, 'image/png');
});
