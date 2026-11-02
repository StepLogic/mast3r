"""Self-contained three.js viewer for an articulated model.

The clouds are sent to the browser once; moving the joint only updates the part's 4x4 matrix,
so the slider and Play run at display frame rate. jointMatrix() in the page mirrors
artsfm.articulate.joint_transform (checked by tests/test_webviewer.py).

    python3 -m artsfm.webviewer <model_dir> [-o viewer.html]

URL options: viewer.html#value=20&bypart&play
"""
import argparse
import base64
import hashlib
import json
import os

import numpy as np

from artsfm.model_io import load_model
from artsfm.viewer import BASE_RGB, PART_RGB, slider_spec

THREE = 'https://cdn.jsdelivr.net/npm/three@0.160.0'


def _b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode('ascii')


def _link(P, C, hull, max_points, rng):
    if len(P) > max_points:
        keep = np.sort(rng.choice(len(P), max_points, replace=False))
        P, C = P[keep], C[keep]
    out = dict(pos=_b64(P, np.float32), rgb=_b64(np.clip(np.round(C * 255), 0, 255), np.uint8), hull=None)
    if hull is not None:
        out['hull'] = dict(pos=_b64(hull.vertices, np.float32), idx=_b64(hull.faces, np.uint32))
    return out


def viewer_payload(model, max_points=150_000, seed=0):
    """JSON-able description of the model for the page (clouds base64-encoded, OpenCV model frame)."""
    rng = np.random.default_rng(seed)
    j = model['joint']
    lo, hi, unit = slider_spec(j)
    ext_lo, ext_hi, _ = slider_spec(j, extend=True)
    base_P, part_P = model['base']['pts'], model['part']['pts']
    allP = np.concatenate([base_P, part_P]) if len(part_P) else base_P
    d = np.asarray(j['axis_dir'], float)
    d = d / np.linalg.norm(d)
    pc = part_P.mean(0) if len(part_P) else allP.mean(0)
    if j['type'] == 'prismatic':
        axis_o = pc
    else:
        o0 = np.asarray(j['axis_origin'], float)
        axis_o = o0 + d * ((pc - o0) @ d)
    joint = {k: (np.asarray(v, float).tolist() if isinstance(v, (list, np.ndarray)) else v) for k, v in j.items()}
    return dict(
        object=model.get('object', ''), joint=joint,
        up=np.asarray(model.get('up', [0, -1, 0]), float).tolist(),
        center=allP.mean(0).tolist(), extent=float(np.ptp(allP, 0).max()),
        axis=dict(o=axis_o.tolist(), d=d.tolist()),
        slider=dict(lo=lo, hi=hi, ext_lo=ext_lo, ext_hi=ext_hi, unit=unit),
        colours=dict(base=BASE_RGB.tolist(), part=PART_RGB.tolist()),
        base=_link(base_P, model['base']['rgb'], model.get('base_hull'), max_points, rng),
        part=_link(part_P, model['part']['rgb'], model.get('part_hull'), max_points, rng),
        warnings=list(model.get('warnings', [])),
    )


def build_html(model, max_points=150_000):
    payload = json.dumps(viewer_payload(model, max_points)).replace('</', '<\\/')
    return _TEMPLATE.replace('__THREE__', THREE).replace('__PAYLOAD__', payload)


def write_html(model_dir, out_path=None, max_points=150_000):
    model = load_model(model_dir)
    out_path = out_path or os.path.join(model_dir, 'viewer.html')
    with open(out_path, 'w') as f:
        f.write(build_html(model, max_points))
    return out_path


_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Articulated model</title>
<style>
  :root { --bg:#111317; --panel:rgba(24,27,33,.92); --fg:#e8e8e8; --muted:#9aa0a6; --accent:#f2b630; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--fg); font:13px/1.4 system-ui,sans-serif; overflow:hidden; }
  #c { position:fixed; inset:0; }
  #ui { position:fixed; left:12px; top:12px; width:220px; background:var(--panel); border-radius:8px; padding:12px;
        display:flex; flex-direction:column; gap:8px; }
  #ui h1 { font-size:14px; margin:0; }
  #ui .muted { color:var(--muted); }
  #ui label { display:flex; align-items:center; gap:6px; }
  #ui input[type=range] { width:100%; accent-color:var(--accent); }
  #ui button { background:#2b3038; color:var(--fg); border:1px solid #3a404a; border-radius:6px; padding:6px; cursor:pointer; }
  #ui button.on { background:var(--accent); color:#111; border-color:var(--accent); }
  .row { display:flex; gap:6px; } .row > * { flex:1; }
  #warn { color:#f28b82; }
  summary { cursor:pointer; color:var(--muted); }
  .opts { display:flex; flex-direction:column; gap:8px; margin-top:8px; }
  @media (max-width:600px) { #ui { width:auto; right:12px; } }
</style></head>
<body>
<canvas id="c"></canvas>
<div id="ui">
  <h1 id="title"></h1>
  <div class="muted" id="jtype"></div>
  <div>Joint: <b id="val"></b></div>
  <input id="slider" type="range">
  <div class="row"><button id="play">Play</button><button id="reset">Reset view</button></div>
  <label><input id="bypart" type="checkbox"> Colour by part</label>
  <details><summary>Options</summary><div class="opts">
    <label>Speed <input id="speed" type="range" min="0.2" max="3" step="0.1" value="1"></label>
    <label>Point size <input id="psize" type="range" min="0.5" max="6" step="0.5" value="2"></label>
    <label><input id="hulls" type="checkbox"> Show collision hulls</label>
    <label><input id="axis" type="checkbox" checked> Show axis</label>
    <label><input id="extend" type="checkbox"> Allow beyond observed range</label>
  </div></details>
  <div id="warn"></div>
</div>
<script>
// Plain JS (no three.js) so the joint maths can be checked without WebGL or network.
const DATA = __PAYLOAD__;
function jointMatrix(j, v) {  // row-major 4x4, mirrors artsfm.articulate.joint_transform
  const n = Math.hypot(...j.axis_dir), d = j.axis_dir.map(x => x / n);
  const T = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  if (j.type === 'prismatic') { for (let i = 0; i < 3; i++) T[i][3] = v * d[i]; return T; }
  const K = [[0,-d[2],d[1]],[d[2],0,-d[0]],[-d[1],d[0],0]], s = Math.sin(v), c = 1 - Math.cos(v);
  const o = j.axis_origin;
  for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) {
    let kk = 0; for (let m = 0; m < 3; m++) kk += K[i][m] * K[m][k];
    T[i][k] = (i === k ? 1 : 0) + s * K[i][k] + c * kk;
  }
  for (let i = 0; i < 3; i++) T[i][3] = o[i] - (T[i][0]*o[0] + T[i][1]*o[1] + T[i][2]*o[2]);
  if (j.type === 'free' && j.observed) for (let i = 0; i < 3; i++) T[i][3] += (v / j.observed) * (j.disp_along || 0) * d[i];
  return T;
}
const CHECK = location.hash.match(/^#check=(.+)$/);
if (CHECK) document.body.innerHTML = '<pre id="check">' + JSON.stringify(jointMatrix(DATA.joint, parseFloat(CHECK[1]))) + '</pre>';
</script>
<script type="importmap">{"imports":{"three":"__THREE__/build/three.module.js","three/addons/":"__THREE__/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
if (!CHECK) {
const $ = id => document.getElementById(id);
const dec = (b64, T) => { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new T(u.buffer); };
const J = DATA.joint, S = DATA.slider, deg = S.unit === 'degrees';
const toJoint = x => deg ? x * Math.PI / 180 : x;

const renderer = new THREE.WebGLRenderer({ canvas: $('c'), antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;   // photo colours are already sRGB: pass them through
const srgb = hex => new THREE.Color().setRGB(((hex >> 16) & 255) / 255, ((hex >> 8) & 255) / 255, (hex & 255) / 255);
const scene = new THREE.Scene(); scene.background = srgb(0x111317);
const root = new THREE.Group(); root.scale.set(1, -1, -1); scene.add(root);   // OpenCV model frame -> GL

function link(L, rgb) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(dec(L.pos, Float32Array), 3));
  g.setAttribute('color', new THREE.BufferAttribute(dec(L.rgb, Uint8Array), 3, true));
  const mat = new THREE.PointsMaterial({ size: 2, sizeAttenuation: false, vertexColors: true });
  const grp = new THREE.Group(); grp.add(new THREE.Points(g, mat));
  let hull = null;
  if (L.hull) {
    const hg = new THREE.BufferGeometry();
    hg.setAttribute('position', new THREE.BufferAttribute(dec(L.hull.pos, Float32Array), 3));
    hg.setIndex(new THREE.BufferAttribute(dec(L.hull.idx, Uint32Array), 1));
    hull = new THREE.Mesh(hg, new THREE.MeshBasicMaterial({ color: new THREE.Color().setRGB(...rgb), transparent: true,
                                                            opacity: 0.25, side: THREE.DoubleSide, depthWrite: false }));
    hull.visible = false; grp.add(hull);
  }
  return { grp, mat, hull, rgb };
}
const base = link(DATA.base, DATA.colours.base), part = link(DATA.part, DATA.colours.part);
root.add(base.grp, part.grp);
part.grp.matrixAutoUpdate = false;

const ext = DATA.extent, d = new THREE.Vector3(...DATA.axis.d), o = new THREE.Vector3(...DATA.axis.o);
const axis = new THREE.Mesh(new THREE.CylinderGeometry(0.004 * ext, 0.004 * ext, 1.2 * ext, 12),
                            new THREE.MeshBasicMaterial({ color: srgb(0xf2b630) }));
axis.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d); axis.position.copy(o); root.add(axis);

const camera = new THREE.PerspectiveCamera(50, 1, 0.01 * ext, 100 * ext);
const target = new THREE.Vector3(...DATA.center).multiply(root.scale);
camera.up.set(...DATA.up).multiply(root.scale).normalize();
const controls = new OrbitControls(camera, renderer.domElement);
function resetView() { camera.position.set(0, 0, 0); controls.target.copy(target); controls.update(); }  // view-0 camera
resetView();

function resize() { renderer.setSize(innerWidth, innerHeight, false); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); }
addEventListener('resize', resize); resize();

$('title').textContent = DATA.object;
$('jtype').textContent = J.type + (J.approximate ? ' (approximate screw)' : '') + ' joint';
$('warn').textContent = DATA.warnings.join(' · ');
const slider = $('slider');
function setRange() {
  const [lo, hi] = $('extend').checked ? [S.ext_lo, S.ext_hi] : [S.lo, S.hi];
  slider.min = lo; slider.max = hi; slider.step = (hi - lo) / 1000;
  setValue(Math.min(Math.max(+slider.value, lo), hi));
}
function setValue(x) {
  slider.value = x;
  $('val').textContent = deg ? `${(+x).toFixed(1)}°` : `${(+x).toFixed(3)} units`;
  const T = jointMatrix(J, toJoint(+x));
  part.grp.matrix.set(...T.flat());
}
slider.value = 0; setRange();
slider.oninput = () => { playing = false; $('play').classList.remove('on'); $('play').textContent = 'Play'; setValue(+slider.value); };
$('extend').onchange = setRange;
$('reset').onclick = resetView;
$('psize').oninput = e => { base.mat.size = part.mat.size = +e.target.value; };
$('axis').onchange = e => { axis.visible = e.target.checked; };
$('hulls').onchange = e => { for (const L of [base, part]) if (L.hull) L.hull.visible = e.target.checked; };
$('bypart').onchange = e => {
  for (const L of [base, part]) { L.mat.vertexColors = !e.target.checked; L.mat.color.setRGB(...(e.target.checked ? L.rgb : [1, 1, 1])); L.mat.needsUpdate = true; }
};

let playing = false, phase = 0, last = performance.now();
$('play').onclick = () => {
  playing = !playing; $('play').classList.toggle('on', playing); $('play').textContent = playing ? 'Pause' : 'Play';
  const lo = +slider.min, hi = +slider.max;   // start the sweep from the current value
  phase = hi > lo ? Math.acos(1 - 2 * (+slider.value - lo) / (hi - lo)) / Math.PI : 0;
};
// URL options, e.g. viewer.html#value=20&bypart&play
const Q = new URLSearchParams(location.hash.slice(1));
if (Q.has('bypart')) { $('bypart').checked = true; $('bypart').onchange({ target: $('bypart') }); }
if (Q.has('value')) setValue(Math.min(Math.max(+Q.get('value'), +slider.min), +slider.max));
if (Q.has('play')) $('play').onclick();
renderer.setAnimationLoop(now => {
  const dt = (now - last) / 1000; last = now;
  if (playing) {   // eased ping-pong over the slider range, one way per 1.5 s at speed 1
    phase += dt * +$('speed').value / 1.5;
    const lo = +slider.min, hi = +slider.max;
    setValue(lo + (hi - lo) * (1 - Math.cos(Math.PI * phase)) / 2);
  }
  controls.update(); renderer.render(scene, camera);
});
}
</script>
</body></html>
"""


PAGE_VERSION = hashlib.md5(_TEMPLATE.encode()).hexdigest()[:8]   # cache key for pages built from this template


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model_dir')
    p.add_argument('-o', '--out', default=None, help='default: <model_dir>/viewer.html')
    p.add_argument('--max_points', type=int, default=150_000, help='per link')
    a = p.parse_args()
    print(write_html(a.model_dir, a.out, a.max_points))


if __name__ == '__main__':
    main()
