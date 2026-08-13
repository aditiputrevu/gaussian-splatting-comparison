# Interactive web viewer for the live demo

We use [antimatter15/splat](https://github.com/antimatter15/splat) — a
lightweight, dependency-free WebGL viewer that loads a single `.splat` file
and lets you fly around the reconstruction in a browser. No build step, no
server-side rendering, works well for a live in-class demo on any laptop.

## 1. Convert a trained 3DGS model to .splat

The reference 3DGS repo trains to a `point_cloud.ply` (in
`<scene>/gaussian_splat_model/point_cloud/iteration_30000/point_cloud.ply`).
`antimatter15/splat` ships a converter for exactly this file:

```bash
bash viewer/package_for_viewer.sh scene1
```

This clones the viewer repo (if not already present) into
`third_party/splat-viewer` and runs its `convert.py` on your trained model,
producing `viewer/exports/scene1.splat`.

## 2. Run the viewer locally

```bash
cd third_party/splat-viewer
python -m http.server 8080
```

Then open `http://localhost:8080/?url=/path/to/your/scene1.splat` (or drag-and-drop
the `.splat` file onto the page — the viewer supports that directly). Use
WASD + mouse to fly around.

## 3. For the demo

Pick your best 1-2 scenes (e.g. the highest-PSNR scene and the hardest/most
interesting one) per the proposal's plan ("Package one or two of the trained
splats into the interactive viewer for a live in-class demo"). Test on the
classroom's machine/browser beforehand if possible -- WebGL2 support and a
decent GPU matter for frame rate on larger splats.
