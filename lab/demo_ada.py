"""Rebuild the README poses using the installed extension. Run with Blender -b -P."""
import json
from pathlib import Path

import bmesh
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "lab" / "output"
OUT.mkdir(exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(ROOT / "blenderModel" / "new_face.blend"))
head = bpy.data.objects["Ada"]
assert not bpy.data.shape_keys and not bpy.data.actions and not head.vertex_groups
bpy.context.view_layer.objects.active = head
head.faceauto_parts.task = "MOUTH"
head.faceauto_parts.mouth = "OPEN"
scene = bpy.context.scene
scene.render.resolution_x = scene.render.resolution_y = 480
scene.render.resolution_percentage = 100


def render(name):
    scene.render.filepath = str(OUT / (name + ".png"))
    bpy.ops.render.render(write_still=True)


def select_vertices(indices):
    bpy.ops.mesh.select_all(action="DESELECT")
    bm = bmesh.from_edit_mesh(head.data)
    bm.verts.ensure_lookup_table()
    for i in indices:
        bm.verts[int(i)].select_set(True)
    bmesh.update_edit_mesh(head.data, destructive=False)


render("ada")
p = np.array([v.co[:] for v in head.data.vertices])
# Two example clicks in Ada's local coordinates. The plugin traces both rims.
clicks = [(0.035, -0.78557, -0.323), (0.035, -0.75302, -0.413)]
seeds = [int(np.argmin(np.linalg.norm(p-pt, axis=1))) for pt in clicks]
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_mode(type="VERT")
for side, i in zip(("upper", "lower"), seeds):
    select_vertices([i])
    assert bpy.ops.faceauto.lip_seed(side=side) == {"FINISHED"}
assert bpy.ops.faceauto.lip_trace(action="both") == {"FINISHED"}
assert bpy.ops.faceauto.lip_trace(action="apply") == {"FINISHED"}
arcs = json.loads(head.faceauto_lips.preview)[-1]
corners = [arcs[0][0], arcs[0][-1]]
select_vertices(corners)
assert bpy.ops.faceauto.speech_mark() == {"FINISHED"}
bpy.ops.object.mode_set(mode="OBJECT")
assert bpy.ops.faceauto.speech_generate() == {"FINISHED"}
assert bpy.ops.faceauto.speech_generate() == {"FINISHED"}
keys = head.data.shape_keys.key_blocks
assert len(keys) == 7

width = np.ptp(p[corners, 0])
camera = scene.camera
camera.location.x = np.mean(p[corners, 0])
camera.location.z = -.37
camera.data.ortho_scale = width*1.62
render("neutral")
def pixel(i):
    x,_,z = p[i]
    return [float(240 + (x-camera.location.x)*480/camera.data.ortho_scale),
            float(240 - (z-camera.location.z)*480/camera.data.ortho_scale)]
markers = dict(arcs=[[pixel(i) for i in r] for r in arcs],
               seeds=[pixel(i) for i in seeds], corners=[pixel(i) for i in corners])
(OUT / "markers.json").write_text(json.dumps(markers))
poses = ("pp","ff","aa","ih","oh","ou")
metrics = {}
for name in (*poses, "ih_ou_half"):
    for n in poses:
        keys["vrc.v_"+n].value = .5 if name=="ih_ou_half" and n in ("ih","ou") else float(n==name)
    bpy.context.view_layer.update()
    ev = head.evaluated_get(bpy.context.evaluated_depsgraph_get())
    q = np.array([v.co[:] for v in ev.data.vertices])
    gap = q[seeds[0],2] - q[seeds[1],2]
    metrics[name] = dict(gap_pct=float(100*gap/width), width_pct=float(100*np.ptp(q[corners,0])/width),
                         protrusion_pct=float(100*(p[seeds,1]-q[seeds,1]).mean()/width))
    render(name)
for n in poses:
    keys["vrc.v_"+n].value = 0
(OUT / "metrics.json").write_text(json.dumps(metrics,indent=2))
bpy.context.preferences.filepaths.save_version=0
scene.render.filepath="//"
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "ada_visemes.blend"))
