"""Create or reuse eyes independently per side; lid landmarks are stored on the head."""
from math import pi, sin, cos
from collections import defaultdict
from .fields import extend_field
import json
import bmesh
import bpy
import numpy as np
from .calibration import coords

MASK="FaceAuto_PaintedEyeFaces"
SIDES=(("NegX","画面左眼","局部 X 负侧"),("PosX","画面右眼","局部 X 正侧"))


def names(side):
    return ["FaceAuto_Lid"+label+"_"+side for label in ("Upper","Lower")]


def split_loop(edges):
    adj=defaultdict(list)
    for e in edges:
        a,b=e.verts
        adj[a].append(b)
        adj[b].append(a)
    if len(adj)<6 or any(len(ns)!=2 for ns in adj.values()):
        raise ValueError("眼缘需要一圈连续的边。贴图眼请选择单块眼白与虹膜表面。")
    start=min(adj,key=lambda v:v.co.x)
    end=max(adj,key=lambda v:v.co.x)
    ring=[start]
    prev,cur=start,adj[start][0]
    while cur!=start:
        ring.append(cur)
        n=next(v for v in adj[cur] if v!=prev)
        prev,cur=cur,n
    if len(ring)!=len(adj):
        raise ValueError("选区含多个轮廓，请仅选一只眼。")
    i=ring.index(end)
    arcs=[ring[:i+1], [start]+ring[:i-1:-1]]
    if min(map(len,arcs))<3:
        raise ValueError("眼缘点数不足。")
    return sorted(arcs,key=lambda arc:-np.mean([v.co.z for v in arc]))


def mask_modifier(obj):
    mod=next((m for m in obj.modifiers if m.type=="NODES" and m.node_group and m.node_group.get(MASK)),None)
    if mod:
        return
    ng=bpy.data.node_groups.new("FaceAuto · 贴图眼选区", "GeometryNodeTree")
    ng[MASK]=True
    for direction in ("INPUT","OUTPUT"):
        ng.interface.new_socket(name="Geometry",in_out=direction,socket_type="NodeSocketGeometry")
    source=ng.nodes.new("NodeGroupInput")
    dest=ng.nodes.new("NodeGroupOutput")
    attr=ng.nodes.new("GeometryNodeInputNamedAttribute")
    attr.data_type="INT"
    attr.inputs["Name"].default_value=MASK
    delete=ng.nodes.new("GeometryNodeDeleteGeometry")
    delete.domain="FACE"
    delete.mode="ONLY_FACE"
    ng.links.new(source.outputs["Geometry"],delete.inputs["Geometry"])
    ng.links.new(attr.outputs["Attribute"],delete.inputs["Selection"])
    ng.links.new(delete.outputs["Geometry"],dest.inputs["Geometry"])
    obj.modifiers.new("FaceAuto · 隐藏原贴图眼面","NODES").node_group=ng


class FACEAUTO_OT_eye_region(bpy.types.Operator):
    bl_idname="faceauto.eye_region"
    bl_label="记录眼缘"
    bl_options={"REGISTER","UNDO"}
    painted:bpy.props.BoolProperty(default=False)

    def execute(self,context):
        obj=context.edit_object
        if not obj or obj.type!="MESH" or len(context.objects_in_mode)!=1 or obj.data.users!=1:
            self.report({"ERROR"},"请单独编辑一个网格。")
            return {"CANCELLED"}
        side=obj.faceauto_parts.side
        bm=bmesh.from_edit_mesh(obj.data)
        faces={f for f in bm.faces if f.select and not f.hide} if self.painted else set()
        if self.painted:
            edges={e for f in faces for e in f.edges if sum(n in faces for n in e.link_faces)==1}
        else:
            edges={e for e in bm.edges if e.select and not e.hide}
        try:
            upper,lower=split_loop(edges)
        except ValueError as err:
            self.report({"ERROR"},str(err))
            return {"CANCELLED"}
        existing=names(side)
        if any(obj.vertex_groups.get(n) is not None for n in existing):
            self.report({"ERROR"},"该侧已有眼缘标记，请检查当前左右侧选择。")
            return {"CANCELLED"}
        bm.faces.index_update()
        bm.verts.index_update()
        arcs=[[v.index for v in arc] for arc in (upper,lower)]
        if self.painted:
            face_ids=[f.index for f in faces]
            layer=bm.faces.layers.int.get(MASK) or bm.faces.layers.int.new(MASK)
            bm.faces.ensure_lookup_table()
            faces=[bm.faces[i] for i in face_ids]
            if any(f[layer] for f in faces):
                self.report({"ERROR"},"该眼面已被标记。")
                return {"CANCELLED"}
            for f in faces:
                f[layer]=1
        dv=bm.verts.layers.deform.verify()
        bm.verts.ensure_lookup_table()
        for name,arc in zip(existing,arcs):
            gid=obj.vertex_groups.new(name=name).index
            for v in arc:
                bm.verts[v][dv][gid]=1
        obj["faceauto_eye_arc_"+side]=json.dumps(arcs)
        bmesh.update_edit_mesh(obj.data,destructive=False)
        if self.painted:
            mask_modifier(obj)
        return {"FINISHED"}


def rims(obj,side,p):
    groups=[obj.vertex_groups.get(n) for n in names(side)]
    if any(g is None for g in groups):
        raise ValueError("请先记录这一侧的眼缘。")
    arcs=[np.array(sorted((v.index for v in obj.data.vertices if any(w.group==g.index and w.weight>0 for w in v.groups)),key=lambda i:p[i,0])) for g in groups]
    if min(map(len,arcs))<3 or np.ptp(p[arcs[0],0])<=0:
        raise ValueError("眼缘标记不足。")
    return arcs


def eye_ref(obj,side):
    return getattr(obj.faceauto_parts,"eye_"+side)


def mesh_eye(obj,side,p,arcs):
    pts=p[np.unique(np.concatenate(arcs))]
    width=np.ptp(pts[:,0])
    radius=max(.61*width,.58*np.ptp(pts[:,2]))
    center=(pts.min(axis=0)+pts.max(axis=0))*.5
    center[1]=pts[:,1].max()+radius+.01*width
    vertices=[(0.,-radius,0.)]
    faces=[]
    slots=[]
    count=64
    angles=sorted(set([i*pi/36 for i in range(1,36)]+[.19,.44,.49]))
    for j,a in enumerate(angles):
        for i in range(count):
            phi=2*pi*i/count
            vertices.append((radius*sin(a)*cos(phi),-radius*cos(a),radius*sin(a)*sin(phi)))
        for i in range(count):
            u=1+j*count+i
            v=1+j*count+(i+1)%count
            faces.append((0,u,v) if j==0 else (u-count,u,v,v-count))
            slots.append(2 if a<=.19 else 1 if a<=.44 else 3 if a<=.49 else 0)
    end=len(vertices)
    vertices.append((0.,radius,0.))
    for i in range(count):
        faces.append((end,1+(len(angles)-1)*count+(i+1)%count,1+(len(angles)-1)*count+i))
        slots.append(0)
    mesh=bpy.data.meshes.new("FaceAuto eye "+side)
    mesh.from_pydata(vertices,[],faces)
    mesh.update()
    eye=bpy.data.objects.new(mesh.name,mesh)
    obj.users_collection[0].objects.link(eye)
    eye.parent=obj
    eye.location=center
    for label,color in (("Sclera",(.86,.81,.71)),("Iris",(.33,.14,.038)),("Pupil",(.009,.005,.003)),("Limbus",(.066,.036,.023))):
        mat=bpy.data.materials.new("FaceAuto "+label)
        mat.diffuse_color=(*color,1)
        mat.use_nodes=True
        next(n for n in mat.node_tree.nodes if n.type=="BSDF_PRINCIPLED").inputs["Base Color"].default_value=(*color,1)
        mesh.materials.append(mat)
    for f,idx in zip(mesh.polygons,slots):
        f.material_index=idx
        f.use_smooth=True
    setattr(obj.faceauto_parts,"eye_"+side,eye)
    return eye


def add_field(obj,name,base,ids,field,value):
    if not obj.data.shape_keys:
        obj.shape_key_add(name="Basis",from_mix=False)
    k=obj.shape_key_add(name=name,from_mix=False)
    for i,offset in zip(ids,field):
        k.data[int(i)].co=base[i]+offset
    k.value=value
    return k


def soften_painted(obj,p,side):
    keyname="FaceAuto_EyeContour_"+side
    keys=obj.data.shape_keys
    existing=keys.key_blocks.get(keyname) if keys else None
    if existing:
        return p+(coords(existing.data)-p)*existing.value
    arcs=json.loads(obj["faceauto_eye_arc_"+side])
    seeds={}
    width=np.ptp(p[arcs[0],0])
    for arc in arcs:
        smooth=p[arc].copy()
        for _ in range(7):
            nxt=smooth.copy()
            nxt[1:-1]=.5*smooth[1:-1]+.25*(smooth[:-2]+smooth[2:])
            smooth=nxt
        seeds.update({i:pt-p[i] for i,pt in zip(arc,smooth)})
    ids,field=extend_field(obj,p,seeds,width)
    add_field(obj,keyname,p,ids,field,1)
    result=p.copy()
    result[ids]+=field
    return result


def blink(obj,base,p,arcs,side):
    name="FaceAuto_Blink_"+side
    if obj.data.shape_keys and name in obj.data.shape_keys.key_blocks:
        raise ValueError("此侧已有眨眼键。")
    upper,lower=arcs
    ends=p[upper[[0,-1]]]
    width=ends[1,0]-ends[0,0]
    seeds={}
    for i in np.unique(np.r_[upper,lower]):
        t=np.clip((p[i,0]-ends[0,0])/width,0,1)
        seeds[i]=(1-t)*ends[0]+t*ends[1]-p[i]
    ids,field=extend_field(obj,p,seeds,width)
    add_field(obj,name,base,ids,field,0)

class FACEAUTO_OT_eye_action(bpy.types.Operator):
    bl_idname="faceauto.eye_action"
    bl_label="生成眼部部件"
    bl_options={"REGISTER","UNDO"}
    action:bpy.props.EnumProperty(items=[("BALL","眼球",""),("BLINK","眨眼","")])

    def execute(self,context):
        obj=context.active_object
        if not obj or obj.type!="MESH" or context.mode!="OBJECT":
            return {"CANCELLED"}
        side=obj.faceauto_parts.side
        p=coords(obj.data.shape_keys.reference_key.data if obj.data.shape_keys else obj.data.vertices)
        try:
            arcs=rims(obj,side,p)
            if self.action=="BALL":
                if getattr(obj.faceauto_parts,"route_"+side)=="EXISTING":
                    raise ValueError("请指定现有眼球，或先选择缺眼球分支。")
                if eye_ref(obj,side):
                    raise ValueError("该侧已指定眼球。")
                if getattr(obj.faceauto_parts,"route_"+side)=="PAINTED":
                    if obj.data.users!=1 or (obj.data.shape_keys and not obj.data.shape_keys.use_relative):
                        raise ValueError("平滑眼缘需要单用户、相对形态键网格。")
                    p=soften_painted(obj,p,side)
                mesh_eye(obj,side,p,arcs)
            else:
                if obj.data.users!=1 or (obj.data.shape_keys and not obj.data.shape_keys.use_relative):
                    raise ValueError("眨眼键需要单用户、相对形态键网格。")
                neutral=soften_painted(obj,p,side) if getattr(obj.faceauto_parts,"route_"+side)=="PAINTED" else p
                blink(obj,p,neutral,rims(obj,side,neutral),side)
        except ValueError as err:
            self.report({"ERROR"},str(err))
            return {"CANCELLED"}
        return {"FINISHED"}





