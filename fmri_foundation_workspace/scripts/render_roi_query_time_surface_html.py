#!/usr/bin/env python3
"""Create an interactive full-cortex HTML demo for ROI query-time maps.

The visualization uses the complete fsaverage5 pial cortical mesh, paints only
the Destrieux visual parcels used by the ROI-query model, and leaves the rest of
the cortex gray. It is meant for presentation/demo use.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from nilearn import datasets, surface

WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from extract_visual_roi_targets_from_tribe import (  # noqa: E402
    build_group_masks,
    build_roi_masks,
    load_destrieux_maps,
)


DEFAULT_RESULTS = WORKSPACE / "results" / "eeg_image_bridge"
DEFAULT_OUT_DIR = DEFAULT_RESULTS / "atm_roi_surface_time_maps"
DEFAULT_NILEARN_DIR = WORKSPACE / "cache" / "nilearn"
WINDOWS = [
    ("w000_100", "0-100 ms"),
    ("w100_200", "100-200 ms"),
    ("w200_300", "200-300 ms"),
    ("w300_400", "300-400 ms"),
    ("w400_500", "400-500 ms"),
    ("w500_600", "500-600 ms"),
    ("w600_700", "600-700 ms"),
    ("w700_800", "700-800 ms"),
    ("w800_900", "800-900 ms"),
    ("w900_1000", "900-1000 ms"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_masks(roi_kind: str, data_dir: Path) -> tuple[np.ndarray, list[str]]:
    label_map, labels = load_destrieux_maps(data_dir)
    if roi_kind == "parcel":
        masks, names, _ = build_roi_masks(label_map, labels)
    elif roi_kind == "group":
        masks, names, _ = build_group_masks(label_map, labels)
    else:
        raise ValueError(f"Unknown roi_kind: {roi_kind}")
    return masks, names


def roi_groups(names: list[str]) -> list[str]:
    out = []
    for name in names:
        stripped = name[3:] if name.startswith(("lh_", "rh_")) else name
        if stripped in {"S_calcarine", "Pole_occipital"}:
            out.append("early_calcarine")
        elif stripped in {"G_cuneus", "S_parieto_occipital", "G_oc-temp_med-Lingual", "S_oc-temp_med_and_Lingual"}:
            out.append("medial_occipital")
        elif stripped in {
            "G_and_S_occipital_inf",
            "G_occipital_middle",
            "G_occipital_sup",
            "S_oc_middle_and_Lunatus",
            "S_oc_sup_and_transversal",
            "S_occipital_ant",
        }:
            out.append("lateral_occipital")
        elif stripped in {"G_oc-temp_lat-fusifor", "S_oc-temp_lat", "S_collat_transv_post"}:
            out.append("ventral_occipitotemporal")
        elif stripped in {"G_temporal_inf", "G_temporal_middle"}:
            out.append("inferior_temporal")
        elif stripped in {"G_oc-temp_med-Parahip", "Pole_temporal"}:
            out.append("semantic_object_temporal")
        else:
            out.append("visual")
    return out


def build_assignment(masks: np.ndarray) -> list[int]:
    assignment = np.full(20484, -1, dtype=np.int16)
    for idx, mask in enumerate(masks):
        assignment[mask] = idx
    return assignment.tolist()


def collect_values(
    rows: list[dict[str, str]],
    run_name: str,
    roi_kind: str,
    condition: str,
    value_key: str,
    roi_names: list[str],
) -> list[list[float | None]]:
    by_window = []
    for window, _ in WINDOWS:
        values = {name: None for name in roi_names}
        for row in rows:
            if row["run"] != run_name or row["roi_kind"] != roi_kind:
                continue
            if row["condition"] != condition or row["window"] != window:
                continue
            try:
                value = float(row[value_key])
            except ValueError:
                continue
            if np.isfinite(value):
                values[row["roi_name"]] = value
        by_window.append([values[name] for name in roi_names])
    return by_window


def combine_meshes(fsaverage: dict[str, str], surface_name: str) -> tuple[np.ndarray, np.ndarray]:
    if surface_name == "pial":
        left = surface.load_surf_mesh(fsaverage["pial_left"])
        right = surface.load_surf_mesh(fsaverage["pial_right"])
    elif surface_name == "inflated":
        left = surface.load_surf_mesh(fsaverage["infl_left"])
        right = surface.load_surf_mesh(fsaverage["infl_right"])
    else:
        raise ValueError(f"Unknown surface: {surface_name}")
    coords = np.vstack([left.coordinates, right.coordinates]).astype("float32")
    faces = np.vstack([left.faces, right.faces + left.coordinates.shape[0]]).astype("int32")
    if surface_name == "inflated":
        coords[: left.coordinates.shape[0], 0] -= 54.0
        coords[left.coordinates.shape[0] :, 0] += 54.0
    # Keep native orientation and center the mesh.
    coords = coords - coords.mean(axis=0, keepdims=True)
    return coords, faces


def compact_float_list(values: np.ndarray, ndigits: int = 4) -> list[list[float]]:
    return np.round(values.astype("float32"), ndigits).tolist()


def html_template(payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ROI Query-Time Cortical Demo</title>
  <style>
    html, body {{ margin: 0; height: 100%; background: #07080b; color: #f1f3f5; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }}
    #app {{ position: fixed; inset: 0; display: grid; grid-template-rows: 1fr auto; }}
    #canvas {{ width: 100%; height: 100%; }}
    #hud {{ position: fixed; left: 18px; top: 14px; max-width: min(620px, calc(100vw - 36px)); padding: 14px 16px; border: 1px solid rgba(255,255,255,.12); background: rgba(8,10,14,.78); backdrop-filter: blur(10px); border-radius: 10px; }}
    #title {{ font-size: 15px; font-weight: 700; margin-bottom: 4px; }}
    #subtitle {{ font-size: 12px; color: #b6bdc8; line-height: 1.35; }}
    #controls {{ position: fixed; left: 18px; right: 18px; bottom: 16px; display: grid; grid-template-columns: auto auto 1fr auto; gap: 12px; align-items: center; padding: 12px 14px; border: 1px solid rgba(255,255,255,.12); background: rgba(8,10,14,.82); backdrop-filter: blur(10px); border-radius: 10px; }}
    button, select {{ background: #151922; color: #f1f3f5; border: 1px solid rgba(255,255,255,.15); border-radius: 7px; padding: 8px 10px; font-size: 13px; }}
    input[type=range] {{ width: 100%; }}
    #timeLabel {{ font-variant-numeric: tabular-nums; min-width: 92px; text-align: right; color: #f7d66b; }}
    #legend {{ position: fixed; right: 18px; top: 18px; width: 160px; padding: 10px; background: rgba(8,10,14,.78); border: 1px solid rgba(255,255,255,.12); border-radius: 10px; }}
    #bar {{ height: 12px; border-radius: 999px; background: linear-gradient(90deg, #2b6cb0, #f4f4f5, #d9480f); margin: 8px 0; }}
    #legendText {{ font-size: 11px; color: #b6bdc8; display: flex; justify-content: space-between; }}
  </style>
</head>
<body>
  <div id="app"><canvas id="canvas"></canvas></div>
  <div id="hud">
    <div id="title">ROI Query-Time Cortical Demo</div>
    <div id="subtitle"></div>
  </div>
  <div id="legend">
    <div style="font-size:12px;color:#d8dde6;">vertex color scale</div>
    <div id="bar"></div>
    <div id="legendText"><span id="minText"></span><span>0</span><span id="maxText"></span></div>
  </div>
  <div id="controls">
    <button id="play">Play</button>
    <select id="mode">
      <option value="keep">keep corr signal</option>
      <option value="drop">drop delta</option>
    </select>
    <input id="slider" type="range" min="0" max="9" value="0" step="1" />
    <div id="timeLabel">0-100 ms</div>
  </div>
  <script type="module">
    import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
    import {{ OrbitControls }} from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';

    const payload = {payload_json};
    const canvas = document.getElementById('canvas');
    const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07080b);
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 1000);
    camera.position.set(0, -240, 95);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, -12, 8);

    const pos = new Float32Array(payload.positions.flat());
    const idx = new Uint32Array(payload.faces.flat());
    const colors = new Float32Array(payload.positions.length * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setIndex(new THREE.BufferAttribute(idx, 1));
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({{
      vertexColors: true,
      roughness: 0.92,
      metalness: 0.02,
      side: THREE.DoubleSide
    }});
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x22252b, 1.25));
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(80, -120, 120);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x7fb3ff, 0.65);
    fill.position.set(-90, 80, 60);
    scene.add(fill);

    function colorKeep(v, vmax) {{
      if (v === null || Number.isNaN(v)) return [0.26, 0.27, 0.30];
      const t = Math.max(-1, Math.min(1, v / vmax));
      if (t >= 0) {{
        return [0.95 + 0.05*t, 0.94 - 0.54*t, 0.88 - 0.70*t];
      }}
      const a = -t;
      return [0.88 - 0.65*a, 0.92 - 0.52*a, 0.98 - 0.12*a];
    }}
    function colorDrop(v, vmax) {{
      if (v === null || Number.isNaN(v)) return [0.26, 0.27, 0.30];
      const t = Math.max(0, Math.min(1, v / vmax));
      return [0.20 + 0.80*t, 0.16 + 0.45*Math.sqrt(t), 0.22 - 0.10*t];
    }}
    function updateColors() {{
      const mode = document.getElementById('mode').value;
      const w = Number(document.getElementById('slider').value);
      const values = payload.values[mode][w];
      const vmax = payload.vmax[mode];
      const colorFn = mode === 'keep' ? colorKeep : colorDrop;
      const assignment = payload.assignment;
      for (let i = 0; i < assignment.length; i++) {{
        const roi = assignment[i];
        const value = roi >= 0 ? values[roi] : null;
        const c = colorFn(value, vmax);
        colors[i*3] = c[0]; colors[i*3+1] = c[1]; colors[i*3+2] = c[2];
      }}
      geometry.attributes.color.needsUpdate = true;
      document.getElementById('timeLabel').textContent = payload.windows[w][1];
      document.getElementById('subtitle').textContent = payload.run_name + ' | ' + mode + ' | ' + payload.windows[w][1] + ' | non-target cortex shown in gray';
      document.getElementById('minText').textContent = mode === 'keep' ? (-vmax).toFixed(2) : '0';
      document.getElementById('maxText').textContent = vmax.toFixed(2);
      document.getElementById('bar').style.background = mode === 'keep'
        ? 'linear-gradient(90deg, #2b6cb0, #f4f4f5, #d9480f)'
        : 'linear-gradient(90deg, #35283a, #b33f32, #ffd166)';
    }}
    document.getElementById('slider').addEventListener('input', updateColors);
    document.getElementById('mode').addEventListener('change', updateColors);
    let playing = false;
    let timer = null;
    document.getElementById('play').addEventListener('click', () => {{
      playing = !playing;
      document.getElementById('play').textContent = playing ? 'Pause' : 'Play';
      if (playing) {{
        timer = setInterval(() => {{
          const s = document.getElementById('slider');
          s.value = (Number(s.value) + 1) % payload.windows.length;
          updateColors();
        }}, 750);
      }} else {{
        clearInterval(timer);
      }}
    }});

    function resize() {{
      const w = window.innerWidth, h = window.innerHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }}
    window.addEventListener('resize', resize);
    resize();
    updateColors();
    renderer.setAnimationLoop(() => {{
      controls.update();
      renderer.render(scene, camera);
    }});
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-time-csv", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--roi-kind", choices=["parcel", "group"], default="parcel")
    parser.add_argument("--surface", choices=["pial", "inflated"], default="pial")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--nilearn-data-dir", type=Path, default=DEFAULT_NILEARN_DIR)
    args = parser.parse_args()

    rows = read_rows(args.query_time_csv)
    masks, names = load_masks(args.roi_kind, args.nilearn_data_dir)
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5", data_dir=str(args.nilearn_data_dir))
    coords, faces = combine_meshes(fsaverage, args.surface)
    keep_values = collect_values(rows, args.run_name, args.roi_kind, "keep", "corr_signal", names)
    drop_values = collect_values(rows, args.run_name, args.roi_kind, "drop", "drop_delta_from_full", names)
    keep_finite = [abs(v) for frame in keep_values for v in frame if v is not None and np.isfinite(v)]
    drop_finite = [v for frame in drop_values for v in frame if v is not None and np.isfinite(v)]
    payload = {
        "run_name": args.run_name,
        "roi_kind": args.roi_kind,
        "atlas": "destrieux_surface_fsaverage5",
        "positions": compact_float_list(coords),
        "faces": faces.tolist(),
        "assignment": build_assignment(masks),
        "roi_names": names,
        "roi_groups": roi_groups(names),
        "windows": WINDOWS,
        "values": {"keep": keep_values, "drop": drop_values},
        "vmax": {
            "keep": max(keep_finite) if keep_finite else 1.0,
            "drop": max(drop_finite) if drop_finite else 1.0,
        },
        "note": "Visualization only: query-time attribution painted onto fixed visual atlas parcels; gray cortex is outside the supervised ROI set.",
    }
    out_dir = args.out_dir / args.tag / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"interactive_full_cortex_query_time_{args.surface}.html"
    out_path.write_text(html_template(payload), encoding="utf-8")
    summary_path = out_dir / "interactive_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "html": str(out_path),
                "run_name": args.run_name,
                "roi_kind": args.roi_kind,
                "n_vertices": int(coords.shape[0]),
                "n_faces": int(faces.shape[0]),
                "n_roi": len(names),
                "surface": args.surface,
                "note": payload["note"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"html": str(out_path), "summary": str(summary_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
