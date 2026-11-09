"""Build a welded oral lining from the open seam of a FaceAuto mesh (local -Y front)."""
from math import pi,sin
import bmesh
import bpy
from mathutils import Vector

from .opening import rim_names
LINING="FaceAuto_OralLining"
# depth, half-width scale, upper/lower vault height, center drop, lip motion transfer
PROFILE=((.012,.997,.009,.0,1.),(.030,.99,.018,.0,.99),
         (.060,.99,.033,.0,.97),(.12,1.03,.066,.01,.92),
         (.24,1.14,.15,.035,.80),(.44,1.32,.38,.08,.53),
         (.64,1.45,.56,.12,.24),(.91,1.42,.59,.13,.09),
         (1.14,1.12,.49,.12,.025),(1.32,.62,.26,.10,0.))


def ordered_bank(bm,bank,ends):
    start,end=ends
    path=[start]
    while path[-1]!=end:
        tip=path[-1]
        opts=[e.other_vert(tip) for e in tip.link_edges if e.is_boundary and e.other_vert(tip) in bank
              and (len(path)<2 or e.other_vert(tip)!=path[-2])]
        if len(opts)!=1 or opts[0] in path:
            raise ValueError("切边必须是从一个嘴角到另一个嘴角的连续开放边链。")
        path.append(opts[0])
    if len(path)!=len(bank):
        raise ValueError("切边组还有游离顶点。")
    return path


def build(obj,bm):
    deform=bm.verts.layers.deform.active
    gids=[obj.vertex_groups[n].index for n in rim_names(obj)]
    banks=[set(),set()]
    for v in bm.verts:
        weights=v[deform]
        for vs,g in zip(banks,gids):
            if weights.get(g,0)>0:
                vs.add(v)
    ends=sorted(banks[0]&banks[1],key=lambda v:v.co.x)
    if len(ends)!=2 or ends[-1].co.x<=ends[0].co.x:
        raise ValueError("需要两个共用嘴角及其间已经分开的上下切边。")
    upper,lower=[ordered_bank(bm,bank,ends) for bank in banks]
    ring=upper+lower[-2:0:-1]
    perimeter=[bm.edges.get((v,ring[(i+1)%len(ring)])) for i,v in enumerate(ring)]
    if not all(e and e.is_boundary for e in perimeter):
        raise ValueError("切边已有内壁或存在非流形边。")
    layers=list(bm.verts.layers.shape.values())
    # Editing the Basis lets Blender propagate its smoothing displacement to relative keys.
    # Store every authored expression's seam delta before the geometry changes.
    offsets=[[v[layer]-v.co for layer in layers] for v in ring]
    w=ends[-1].co.x-ends[0].co.x
    cx=(ends[0].co.x+ends[-1].co.x)*.5
    # Apply the same neutral curve to matching seam banks without changing corner positions.
    def soften(chain):
        saved=[v.co.copy() for v in chain]
        for _ in range(3):
            nxt=[pt.copy() for pt in saved]
            for i in range(1,len(chain)-1):
                a,v,b=saved[i-1:i+2]
                t=(v.x-a.x)/(b.x-a.x) if abs(b.x-a.x)>1e-8 else .5
                t=min(1,max(0,t))
                target=a.lerp(b,t)
                nxt[i].y=.60*v.y+.40*target.y
                nxt[i].z=.60*v.z+.40*target.z
            saved=nxt
        return saved
    neutral=soften(upper)+soften(lower)[-2:0:-1]
    for v,pt in zip(ring,neutral):
        v.co=pt
    winding=next(loop for loop in perimeter[0].link_loops if loop.edge==perimeter[0]).vert==ring[0]
    # Exterior uses 0 -> 1? Then run the new quad 1 -> 0 along the same edge.
    def face(vertices,material):
        f=bm.faces.new(list(reversed(vertices)) if winding else vertices)
        f.material_index=material
        f.smooth=True
        return f
    slot=[]
    for label,color in (("FaceAuto inner lip",(.46,.14,.15)),("FaceAuto mouth lining",(.125,.025,.032))):
        mat=bpy.data.materials.new(label)
        mat.diffuse_color=(*color,1)
        mat.use_nodes=True
        next(n for n in mat.node_tree.nodes if n.type=="BSDF_PRINCIPLED").inputs["Base Color"].default_value=(*color,1)
        obj.data.materials.append(mat)
        slot.append(len(obj.data.materials)-1)
    group=obj.vertex_groups.new(name=LINING)
    prev=ring
    lower_follow=(.68,.49,.37,.29,.22,.16,.10,.05,.02,0.)
    for j,(depth,scale,height,drop,follow) in enumerate(PROFILE):
        new=[]
        for i,pt in enumerate(neutral):
            u=min(1.,max(0.,(pt.x-ends[0].co.x)/w))
            arch=max(0.,sin(pi*u))**.53
            sign=1 if i<len(upper) else -1
            pos=Vector((cx+(pt.x-cx)*scale,pt.y+depth*w,pt.z+(sign*height*arch-drop)*w))
            v=bm.verts.new(pos)
            for layer,offset in zip(layers,offsets[i]):
                transfer=lower_follow[j] if sign<0 and offset.z<0 else follow
                v[layer]=pos+offset*transfer
            v[deform][group.index]=1.
            new.append(v)
        for i in range(len(ring)):
            n=(i+1)%len(ring)
            face((prev[i],prev[n],new[n],new[i]),slot[int(j>=3)])
        prev=new
    center=sum((v.co for v in prev),Vector())/len(prev)
    center.y+=.08*w
    back=bm.verts.new(center)
    for layer in layers:
        back[layer]=center
    back[deform][group.index]=1.
    for i in range(len(prev)):
        face((prev[i],prev[(i+1)%len(prev)],back),slot[1])
    bm.normal_update()
    return len(ring)


class FACEAUTO_OT_lining(bpy.types.Operator):
    bl_idname="faceauto.build_lining"
    bl_label="平滑唇缘并生成相连唇壁、口腔袋"
    bl_options={"REGISTER","UNDO"}

    @classmethod
    def poll(cls,context):
        obj=context.active_object
        return (context.mode=="OBJECT" and obj and obj.type=="MESH" and obj.data.users==1
                and all(obj.vertex_groups.get(n) is not None for n in rim_names(obj))
                and obj.vertex_groups.get(LINING) is None and len(context.selected_objects)==1
                and (not obj.data.shape_keys or obj.data.shape_keys.use_relative))

    def execute(self,context):
        obj=context.active_object
        old_index=obj.active_shape_key_index
        obj.active_shape_key_index=0
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            bm=bmesh.from_edit_mesh(obj.data)
            count=build(obj,bm)
            bmesh.update_edit_mesh(obj.data,loop_triangles=True,destructive=True)
        except ValueError as err:
            self.report({"ERROR"},str(err))
            return {"CANCELLED"}
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")
            obj.active_shape_key_index=old_index
        for child in obj.children:
            if child.get("faceauto_oral") and child.name.startswith("FaceAuto Oral cavity"):
                child.hide_viewport=True
                child.hide_render=True
        self.report({"INFO"},f"已沿 {count} 条切边接入唇壁和封底口腔袋。")
        return {"FINISHED"}




