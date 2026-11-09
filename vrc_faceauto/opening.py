"""Build a relative opening key from the two banks of an already cut seam."""
import heapq
from collections import defaultdict

import bpy
import numpy as np

from .calibration import coords, layout


UPPER_KEY="FaceAuto_UpperLipUp"


def rim_names(obj):
    seam=("FaceAuto_SeamUpper","FaceAuto_SeamLower")
    if all(obj.vertex_groups.get(n) is not None for n in seam):
        return seam
    if obj.faceauto_parts.mouth!="SEALED":
        return ("Scan_upper_lip","Scan_lower_lip")
    return seam


def opening_offsets(obj, lift_upper=False):
    mesh=obj.data
    base=coords(mesh.shape_keys.reference_key.data if mesh.shape_keys else mesh.vertices)
    group_ids=[obj.vertex_groups[name].index for name in rim_names(obj)]
    upper,lower=[{v.index for v in mesh.vertices if any(g.group==gid and g.weight>0 for g in v.groups)}
                 for gid in group_ids]
    fixed,moving=(lower,upper) if lift_upper else (upper,lower)
    bank=moving-fixed
    if not bank or not fixed-moving:
        raise ValueError("上下切边没有独立顶点，无法生成张嘴位移。")
    rim=base[list(lower|upper)]
    left=rim[:,0].min()
    width=np.ptp(rim[:,0])
    if width<=0:
        raise ValueError("切边嘴宽为零。")
    radius=(.24 if lift_upper else .4)*width
    nearby=((base>=rim.min(axis=0)-radius)&(base<=rim.max(axis=0)+radius)).all(axis=1)
    pairs=np.empty(len(mesh.edges)*2,dtype=np.int32)
    mesh.edges.foreach_get("vertices",pairs)
    pairs=pairs.reshape(-1,2)
    pairs=pairs[nearby[pairs].all(axis=1)]
    neighbors=defaultdict(list)
    for (a,b),length in zip(pairs,np.linalg.norm(base[pairs[:,0]]-base[pairs[:,1]],axis=1)):
        neighbors[a].append((b,length))
        neighbors[b].append((a,length))
    distance=dict.fromkeys(bank,0.)
    queue=[(0.,i) for i in bank]
    heapq.heapify(queue)
    while queue:
        d,v=heapq.heappop(queue)
        if d!=distance[v]:
            continue
        for n,length in neighbors[v]:
            nd=d+length
            if n not in fixed and nd<radius and nd<distance.get(n,np.inf):
                distance[n]=nd
                heapq.heappush(queue,(nd,n))
    ids=np.array(list(distance))
    u=np.clip((base[ids,0]-left)/width,0,1)
    dz=(.12 if lift_upper else .24)*width*np.sin(np.pi*u)**2*(1-np.array(list(distance.values()))/radius)**2
    if not np.any(dz>width*.001):
        raise ValueError("当前切边没有足够的活动范围。")
    return ids,dz


class FACEAUTO_OT_opening(bpy.types.Operator):
    bl_idname="faceauto.create_opening"
    bl_label="生成张嘴形态键并预览"
    bl_options={"REGISTER","UNDO"}

    @classmethod
    def poll(cls,context):
        obj=context.active_object
        return (context.mode in {"EDIT_MESH","OBJECT"} and obj and obj.type=="MESH"
                and all(obj.vertex_groups.get(n) is not None for n in rim_names(obj))
                and not layout(obj) and obj.data.users==1 and
                (not obj.data.shape_keys or obj.data.shape_keys.use_relative)
                and (context.mode=="OBJECT" or len(context.objects_in_mode)==1))

    def execute(self,context):
        return create_key(self,context,False)


class FACEAUTO_OT_upper(bpy.types.Operator):
    bl_idname="faceauto.create_upper"
    bl_label="生成上唇上提形态键"
    bl_options={"REGISTER","UNDO"}

    @classmethod
    def poll(cls,context):
        obj=context.active_object
        return (context.mode in {"OBJECT","EDIT_MESH"} and obj and obj.type=="MESH" and obj.data.users==1
                and (context.mode=="OBJECT" or len(context.objects_in_mode)==1)
                and all(obj.vertex_groups.get(n) is not None for n in rim_names(obj))
                and (not obj.data.shape_keys or (obj.data.shape_keys.use_relative and UPPER_KEY not in obj.data.shape_keys.key_blocks)))

    def execute(self,context):
        return create_key(self,context,True)


def create_key(operator,context,lift_upper):
    obj=context.active_object
    editing=context.mode=="EDIT_MESH"
    if editing:
        bpy.ops.object.mode_set(mode="OBJECT")
    try:
        ids,dz=opening_offsets(obj,lift_upper)
    except ValueError as exc:
        if editing:
            bpy.ops.object.mode_set(mode="EDIT")
        operator.report({"ERROR"},str(exc))
        return {"CANCELLED"}
    if not obj.data.shape_keys:
        obj.shape_key_add(name="Basis",from_mix=False)
    if lift_upper:
        # Older oral kits attached the upper teeth and cavity to an upper lip vertex.
        group=obj.vertex_groups[rim_names(obj)[0]].index
        for child in obj.children:
            if (child.get("faceauto_oral") and child.parent_type=="VERTEX" and
                any(g.group==group for g in obj.data.vertices[child.parent_vertices[0]].groups)):
                world=child.matrix_world.copy()
                child.parent_type="OBJECT"
                child.matrix_world=world
    key=obj.shape_key_add(name=UPPER_KEY if lift_upper else "Open_All",from_mix=False)
    for idx,offset in zip(ids,dz):
        key.data[int(idx)].co.z+=offset if lift_upper else -offset
    key.value=1
    obj.active_shape_key_index=obj.data.shape_keys.key_blocks.find(key.name)
    obj.show_only_shape_key=False
    context.view_layer.update()
    operator.report({"INFO"},f"已生成 {key.name}，当前预览 1.0。")
    return {"FINISHED"}



