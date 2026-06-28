# GUI Framework Evaluation: Bevy vs egui vs gpui

**Issue:** ferritin-pya  
**Date:** 2026-06-28  
**Decision:** Continue with Bevy for 3D protein visualization

---

## What we need

- 3D mesh rendering: ribbon/cartoon tubes, ball-and-stick spheres, surface meshes
- Lighting and materials: ambient/directional light, PBR or Phong shading
- Camera: orbit/zoom/pan controls for a molecule viewer
- Overlays: residue labels, selection highlighting, color legend
- macOS and Linux support (Windows nice-to-have)

---

## Options evaluated

### 1. Bevy

**Pros**
- Full 3D PBR renderer with mesh, lighting, camera out of the box
- `ferritin-bevy` already generates meshes and renders structures; the integration path is proven
- Rich plugin ecosystem: `bevy_panorbit_camera`, `bevy_mod_picking`, etc.
- Actively developed; 0.15 → 0.16 upgrade path is well-documented

**Cons**
- API churn between minor versions (breaking changes on each 0.x release)
- Heavy dependency tree (~60 crates); full rebuild is slow
- No immediate-mode 2D widgets — need an egui-bevy bridge for sidebars and property panels

**Verdict:** Best fit for 3D mesh viewing. The 0.15 → 0.19 compat issue cited in the bug was caused by API-version mismatch, not a structural problem with Bevy.

---

### 2. egui

**Pros**
- Excellent immediate-mode 2D GUI; low boilerplate for dev tools and inspection panels
- `egui_wgpu` backend enables custom 3D render passes inside a panel
- Small dependency footprint; fast iteration

**Cons**
- Not designed as a primary 3D scene renderer; custom 3D requires wgpu render passes written by hand
- Mesh, lighting, and camera systems must all be built from scratch on top of wgpu
- Cartoon ribbons, per-residue coloring, depth-correct transparency: significant work

**Verdict:** Good for inspector panels and debug overlays inside a Bevy window (via `bevy_egui`). Not a replacement for Bevy as the 3D renderer.

---

### 3. Zed gpui

**Pros**
- GPU-accelerated; designed for Zed's text editor rendering
- High performance for 2D UI elements

**Cons**
- No 3D mesh primitives; built for 2D vector/text rendering
- Small public ecosystem; sparse documentation outside Zed internals
- Would require the same wgpu-from-scratch approach as egui for 3D

**Verdict:** Wrong tool for protein 3D visualization.

---

## Recommendation

**Keep Bevy. Add egui-bevy for 2D overlays.**

- `ferritin-bevy` stays the 3D rendering crate (mesh generation + Bevy scene)
- Add `bevy_egui` as a dev-dependency in `ferritin-examples` for inspection panels
- Pin to a specific Bevy minor version in `Cargo.toml` and update only deliberately

The secondary structure rendering bug (ferritin-z46) and cartoon mesh quality are the
next priority, not a framework switch.  Switching to egui or gpui would require rebuilding
all 3D logic from scratch without meaningful gains for protein visualization.
