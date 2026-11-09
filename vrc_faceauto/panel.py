import bpy
import numpy as np

from .calibration import coords, grouping, layout, measurement
from .solver import bounded_weights
from . import lip_ui, seam, oral, opening, gaze, lining, parts, eyes, speech


class FaceAutoSettings(bpy.types.PropertyGroup):
    target: bpy.props.FloatVectorProperty(name="目标开度", size=3, default=(40,46,40), min=0,
        precision=2, description="上下唇间距占中性嘴宽的百分比")
    budget: bpy.props.EnumProperty(name="独立控制数", items=[
        ("AUTO","全部","使用当前模型已有的全部开口控制"),
        ("1","1 · 整体","三个区域联动"),("2","2 · 双侧 / 中央","左右联动，中央独立"),
        ("3","3 · 左 / 中 / 右","三区独立")], default="AUTO")
    result: bpy.props.StringProperty()
    error: bpy.props.StringProperty()


def budget(settings, names):
    return len(names) if settings.budget=="AUTO" else int(settings.budget)


def fit(obj, context):
    settings = obj.faceauto
    names = layout(obj)
    groups = grouping(len(names),budget(settings,names))
    keys = [obj.data.shape_keys.key_blocks[n] for n in names]
    if not obj.data.shape_keys.use_relative or any(k.mute for k in keys) or obj.show_only_shape_key:
        raise ValueError("请启用相对形态键、取消开口键静音，并关闭仅显示当前形态键。")
    if any(m.show_viewport and not (m.type=="NODES" and m.node_group and m.node_group.get(eyes.MASK)) for m in obj.modifiers):
        raise ValueError("当前版本需在无启用修改器的基础网格上测量。")
    measure = measurement(obj)
    def pose(w):
        for group, value in zip(groups,w):
            for i in group:
                keys[i].value = float(value)
        context.view_layer.update()
        if any(abs(keys[i].value-value)>1e-5 for group,value in zip(groups,w) for i in group):
            raise ValueError("滑块范围或动画驱动阻止了权重写入。")
        ev = obj.evaluated_get(context.evaluated_depsgraph_get())
        return measure(coords(ev.data.vertices))
    original = [k.value for k in keys]
    try:
        neutral = pose(np.zeros(len(groups)))
        response = np.column_stack([pose(w)-neutral for w in np.eye(len(groups))])
        w = bounded_weights(neutral,response,settings.target[:])
        actual = pose(w)
        settings.result = " / ".join(f"{v:.3f}" for v in actual)
        settings.error = f"{max(abs(actual-settings.target[:])):.5f} 个百分点"
    except Exception:
        for k,v in zip(keys, original):
            k.value = v
        context.view_layer.update()
        raise


class FACEAUTO_OT_fit(bpy.types.Operator):
    bl_idname = "faceauto.fit"
    bl_label = "计算权重并应用"
    bl_options = {"REGISTER","UNDO"}

    def execute(self, context):
        try:
            fit(context.active_object,context)
        except (RuntimeError,ValueError) as err:
            self.report({"ERROR"},str(err))
            return {"CANCELLED"}
        return {"FINISHED"}

    @classmethod
    def poll(cls, context):
        return context.mode=="OBJECT" and bool(layout(context.active_object))


def can_copy(obj,names):
    if not names or len(obj.data.vertices)!=26719 or obj.modifiers or obj.data.shape_keys.animation_data:
        return False
    return len(obj.data.shape_keys.key_blocks)==len(names)+1 and all(
        k.relative_key == obj.data.shape_keys.reference_key and not k.vertex_group
        for k in list(obj.data.shape_keys.key_blocks)[1:])


class FACEAUTO_OT_copy(bpy.types.Operator):
    bl_idname = "faceauto.compact_copy"
    bl_label = "生成精简 ICT 副本"
    bl_description = "在原模型旁生成只含所选开口键的新网格，并对它应用目标开度"
    bl_options = {"REGISTER","UNDO"}

    @classmethod
    def poll(cls,context):
        obj = context.active_object
        names = layout(obj)
        return context.mode=="OBJECT" and bool(names) and budget(obj.faceauto,names)<len(names) and can_copy(obj,names)

    def execute(self,context):
        src = context.active_object
        names = layout(src)
        count = budget(src.faceauto,names)
        copy = src.copy()
        copy.data = src.data.copy()
        context.collection.objects.link(copy)
        try:
            blocks = copy.data.shape_keys.key_blocks
            basis = coords(blocks[0].data)
            merged = [basis+sum(coords(blocks[names[i]].data)-basis for i in group)
                      for group in grouping(len(names),count)]
            for k in list(blocks)[1:]:
                copy.shape_key_remove(k)
            for name,positions in zip(("Open_All",) if count==1 else ("Open_Sides","Open_C"),merged):
                k=copy.shape_key_add(name=name,from_mix=False)
                k.data.foreach_set("co",positions.astype("f").ravel())
                k.value=0
            copy.name=f"{src.name} ({count}键)"
            copy.location.x += 1.15*src.dimensions.x
            fit(copy,context)
        except Exception as err:
            mesh=copy.data
            bpy.data.objects.remove(copy,do_unlink=True)
            bpy.data.meshes.remove(mesh)
            self.report({"ERROR"},str(err))
            return {"CANCELLED"}
        src.select_set(False)
        copy.select_set(True)
        context.view_layer.objects.active=copy
        return {"FINISHED"}


class FACEAUTO_PT_aperture(bpy.types.Panel):
    bl_space_type="VIEW_3D"
    bl_region_type="UI"
    bl_category="FaceAuto"
    bl_label="嘴部开度"
    bl_order=1

    @classmethod
    def poll(cls,context):
        obj=context.active_object
        return obj is not None and obj.type=="MESH" and obj.faceauto_parts.task=="MOUTH" and bool(layout(obj))

    def draw(self,context):
        ui=self.layout
        obj=context.active_object
        names=layout(obj)
        if not names:
            ui.label(text="已切开嘴缝？在「嘴缝连通」生成张嘴键。")
            return
        settings=obj.faceauto
        ui.label(text=obj.name,icon="MESH_DATA")
        ui.label(text="目标开度（占中性嘴宽的 %）")
        for i,label in enumerate(("左","中央","右")):
            ui.prop(settings,"target",index=i,text=label)
        ui.prop(settings,"budget")
        ui.operator("faceauto.fit")
        if budget(settings,names)<len(names) and can_copy(obj,names):
            ui.operator("faceauto.compact_copy")
        if settings.result:
            ui.label(text="上次实测 (%)   "+settings.result)
            ui.label(text="最大误差   "+settings.error)
        ui.separator()
        ui.label(text="当前形态键权重")
        for n in names:
            ui.prop(obj.data.shape_keys.key_blocks[n],"value",text=n,slider=True)


classes=lip_ui.classes+(speech.FACEAUTO_OT_speech_mark,speech.FACEAUTO_OT_speech_generate,speech.FACEAUTO_PT_speech,parts.Parts,parts.FACEAUTO_PT_parts,eyes.FACEAUTO_OT_eye_region,eyes.FACEAUTO_OT_eye_action,lining.FACEAUTO_OT_lining,gaze.FACEAUTO_OT_gaze,gaze.FACEAUTO_PT_gaze,opening.FACEAUTO_OT_opening,opening.FACEAUTO_OT_upper,oral.FACEAUTO_OT_oral,seam.FACEAUTO_OT_seam,seam.FACEAUTO_PT_seam,FaceAutoSettings,FACEAUTO_OT_fit,FACEAUTO_OT_copy,FACEAUTO_PT_aperture)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.faceauto_parts=bpy.props.PointerProperty(type=parts.Parts)
    bpy.types.Object.faceauto_lips=bpy.props.PointerProperty(type=lip_ui.FaceAutoLipState)
    bpy.types.Object.faceauto=bpy.props.PointerProperty(type=FaceAutoSettings)


def unregister():
    del bpy.types.Object.faceauto_parts
    del bpy.types.Object.faceauto_lips
    del bpy.types.Object.faceauto
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)











