"""Blender edit-mode workflow for seeded inner-lip vertex groups."""
import hashlib
import json

import bmesh
import bpy
import numpy as np

from .lip_trace import trace


class FaceAutoLipState(bpy.types.PropertyGroup):
    method: bpy.props.EnumProperty(name="定位方式",items=[
        ("AUTO","中央两点","上唇和下唇分别选一个中央点，沿四边形边流追踪到横向转折"),
        ("GUIDED","端点及引导点","每条唇缘选左右端点，可添加中间点来约束路径")])
    upper: bpy.props.StringProperty(options={"HIDDEN"})
    lower: bpy.props.StringProperty(options={"HIDDEN"})
    preview: bpy.props.StringProperty(options={"HIDDEN"})


def edit_mesh(context):
    obj=context.edit_object
    bm=bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    return obj,bm


def snapshot(bm):
    points=np.array([tuple(v.co) for v in bm.verts])
    faces=[tuple(v.index for v in f.verts) for f in bm.faces if not f.hide and not any(v.hide for v in f.verts)]
    digest=hashlib.sha256(points.tobytes())
    digest.update(np.asarray([i for f in faces for i in (*f,-1)],dtype="i4").tobytes())
    return points,faces,digest.hexdigest()


def resolved(obj,bm):
    state=obj.faceauto_lips
    points,faces,stamp=snapshot(bm)
    seeds=[]
    for raw in (state.upper,state.lower):
        if not raw:
            raise ValueError("请先分别记录上唇和下唇种子。")
        saved=json.loads(raw)
        if saved["mesh"]!=stamp:
            raise ValueError("网格几何或可见范围已变化，请重新记录上下唇种子。")
        seeds.append(saved["ids"])
    paths=trace(points,faces,*seeds,state.method=="GUIDED")
    return paths, json.dumps([stamp,state.method,*seeds,paths])


class EditLipOperator:
    @classmethod
    def poll(cls,context):
        return (context.mode=="EDIT_MESH" and len(context.objects_in_mode)==1
                and context.tool_settings.mesh_select_mode[0])


class FACEAUTO_OT_lip_seed(EditLipOperator,bpy.types.Operator):
    bl_idname="faceauto.lip_seed"
    bl_label="记录唇缘种子"
    bl_options={"UNDO"}
    side:bpy.props.EnumProperty(items=[("upper","上唇",""),("lower","下唇","")])

    def execute(self,context):
        obj,bm=edit_mesh(context)
        state=obj.faceauto_lips
        ids=[v.index for v in bm.verts if v.select and not v.hide]
        if (state.method=="AUTO" and len(ids)!=1) or (state.method=="GUIDED" and len(ids)<2):
            self.report({"ERROR"},"中央两点模式每次选一个点，多点模式每条唇缘至少选两个端点。")
            return {"CANCELLED"}
        setattr(state,self.side,json.dumps(dict(mesh=snapshot(bm)[2],ids=ids)))
        state.preview=""
        return {"FINISHED"}


class FACEAUTO_OT_lip_trace(EditLipOperator,bpy.types.Operator):
    bl_idname="faceauto.lip_trace"
    bl_label="预览唇缘或写入顶点组"
    bl_options={"UNDO"}
    action:bpy.props.EnumProperty(items=[("upper","预览上唇",""),("lower","预览下唇",""),
                                          ("both","预览两条唇缘",""),("apply","更新两个顶点组","")])

    def execute(self,context):
        obj,bm=edit_mesh(context)
        state=obj.faceauto_lips
        try:
            paths,token=resolved(obj,bm)
            if self.action=="apply":
                if state.preview!=token:
                    raise ValueError("请先预览当前种子生成的路径，再写入顶点组。")
                if obj.data.users>1:
                    raise ValueError("网格数据由多个物体共享，请先将其转为单用户。")
                for name in ("Scan_upper_lip","Scan_lower_lip"):
                    current=obj.vertex_groups.get(name)
                    if current and current.lock_weight:
                        raise ValueError(f"{name} 的权重已锁定。")
                deform=bm.verts.layers.deform.verify()
                for name,ids in zip(("Scan_upper_lip","Scan_lower_lip"),paths):
                    group=obj.vertex_groups.get(name)
                    if group is None:
                        group=obj.vertex_groups.new(name=name)
                    selected=set(ids)
                    for v in bm.verts:
                        weights=v[deform]
                        if v.index in selected:
                            weights[group.index]=1
                        elif group.index in weights:
                            del weights[group.index]
            else:
                selected=set(paths[0] if self.action=="upper" else paths[1] if self.action=="lower" else paths[0]+paths[1])
                for f in bm.faces:
                    f.select_set(False)
                for e in bm.edges:
                    e.select_set(False)
                for v in bm.verts:
                    v.select_set(v.index in selected)
                bm.select_flush_mode()
                state.preview=token
            bmesh.update_edit_mesh(obj.data,loop_triangles=False,destructive=False)
        except ValueError as exc:
            self.report({"ERROR"},str(exc))
            return {"CANCELLED"}
        self.report({"INFO"},f"上唇 {len(paths[0])} 个顶点，下唇 {len(paths[1])} 个顶点")
        return {"FINISHED"}


class FACEAUTO_PT_lips(bpy.types.Panel):
    bl_space_type="VIEW_3D"
    bl_region_type="UI"
    bl_category="FaceAuto"
    bl_label="唇缘定位"
    bl_order=0

    @classmethod
    def poll(cls,context):
        obj=context.active_object
        return obj is not None and obj.type=="MESH" and obj.faceauto_parts.task=="MOUTH" and (obj.faceauto_parts.mouth!="READY" or obj.faceauto_parts.oral=="ADD" or obj.faceauto_parts.teeth or obj.faceauto_parts.tongue)

    def draw(self,context):
        ui=self.layout
        if not FACEAUTO_OT_lip_seed.poll(context):
            ui.label(text="选中一个网格，Tab 编辑，切到顶点选择。")
            return
        state=context.edit_object.faceauto_lips
        ui.prop(state,"method")
        ui.label(text="分别选上、下唇中央点。" if state.method=="AUTO" else "每条唇分别选两端，可加中间点。")
        for side,label in (("upper","上唇"),("lower","下唇")):
            raw=getattr(state,side)
            n=len(json.loads(raw)["ids"]) if raw else 0
            ui.operator("faceauto.lip_seed",text=f"记录{label}选点（已记 {n}）").side=side
        row=ui.row(align=True)
        for action,label in (("upper","预览上唇"),("lower","预览下唇"),("both","预览全部")):
            row.operator("faceauto.lip_trace",text=label).action=action
        ui.operator("faceauto.lip_trace",text="写入 / 更新两个唇缘组").action="apply"


classes=(FaceAutoLipState,FACEAUTO_OT_lip_seed,FACEAUTO_OT_lip_trace,FACEAUTO_PT_lips)



