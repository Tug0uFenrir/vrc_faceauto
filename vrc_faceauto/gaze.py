"""Gaze endpoints for the indexed ICT sclera and iris meshes."""
from math import radians
import bpy
import numpy as np
from mathutils import Matrix
from .calibration import coords

DIRECTIONS={"Left":((0,0,1),-20),"Right":((0,0,1),20),
            "Up":((1,0,0),-12),"Down":((1,0,0),12)}
LABELS=("向画面左","向画面右","向上","向下")
EYES=((21451,22221,23021),(23021,23791,24591))


def ict_mesh(obj):
    if not obj or obj.type!="MESH":
        return False
    mesh=obj.data
    return (len(mesh.vertices)==26719 and len(mesh.polygons)==26384
            and all(mesh.materials.get(n) is not None for n in ("left sclera","right sclera","left iris","right iris")))


def centers(mesh,p):
    result=[]
    for (start,iris,end),side in zip(EYES,("left","right")):
        # Verify actual face membership, not just vertex count.
        expected={mesh.materials.find(side+" sclera"):set(range(start,iris)),
                  mesh.materials.find(side+" iris"):set(range(iris,end))}
        actual={idx:set() for idx in expected}
        for poly in mesh.polygons:
            if poly.material_index in actual:
                actual[poly.material_index].update(poly.vertices)
        if actual!=expected:
            raise ValueError("眼球索引与 ICT 原始分区不匹配。")
        ball=p[start:iris]
        mean=ball.mean(axis=0)
        centered=ball-mean
        coef,_,rank,_=np.linalg.lstsq(np.column_stack((2*centered,np.ones(len(ball)))),
                                      np.sum(centered**2,axis=1),rcond=None)
        if rank!=4:
            raise ValueError("巩膜点不足以拟合球心。")
        result.append(mean+coef[:3])
    return result


class FACEAUTO_OT_gaze(bpy.types.Operator):
    bl_idname="faceauto.create_gaze"
    bl_label="生成双眼视线形态键"
    bl_options={"REGISTER","UNDO"}

    @classmethod
    def poll(cls,context):
        obj=context.active_object
        return (context.mode=="OBJECT" and ict_mesh(obj) and obj.data.users==1
                and (not obj.data.shape_keys or (obj.data.shape_keys.use_relative and not any(
                    "FaceAuto_Gaze"+n in obj.data.shape_keys.key_blocks for n in DIRECTIONS))))

    def execute(self,context):
        obj=context.active_object
        mesh=obj.data
        p=coords(mesh.shape_keys.reference_key.data if mesh.shape_keys else mesh.vertices)
        try:
            pivots=centers(mesh,p)
        except ValueError as exc:
            self.report({"ERROR"},str(exc))
            return {"CANCELLED"}
        if not mesh.shape_keys:
            obj.shape_key_add(name="Basis",from_mix=False)
        for name,(axis,angle) in DIRECTIONS.items():
            points=p.copy()
            rot=np.array(Matrix.Rotation(radians(angle),3,axis))
            for (start,_,end),center in zip(EYES,pivots):
                points[start:end]=(p[start:end]-center)@rot.T+center
            k=obj.shape_key_add(name="FaceAuto_Gaze"+name,from_mix=False)
            k.data.foreach_set("co",points.astype("f").ravel())
            k.value=0
        return {"FINISHED"}


class FACEAUTO_PT_gaze(bpy.types.Panel):
    bl_space_type="VIEW_3D"
    bl_region_type="UI"
    bl_category="FaceAuto"
    bl_label="眼珠视线 · ICT"
    bl_order=3

    @classmethod
    def poll(cls,context):
        obj=context.active_object
        return obj is not None and obj.type=="MESH" and obj.faceauto_parts.task=="EYES" and ict_mesh(obj)

    def draw(self,context):
        ui=self.layout
        obj=context.active_object
        if not ict_mesh(obj):
            ui.label(text="此版眼珠控制适用于 ICT 完整实验脸。")
            return
        ui.operator("faceauto.create_gaze")
        keys=obj.data.shape_keys
        for name,label in zip(DIRECTIONS,LABELS):
            k=keys.key_blocks.get("FaceAuto_Gaze"+name) if keys else None
            if k:
                ui.prop(k,"value",text=label,slider=True)

