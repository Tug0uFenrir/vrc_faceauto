"""FaceAuto entry points are chosen per mesh and per component."""
import bpy
from . import eyes,opening
from .calibration import layout

EYE_TYPES=[("EXISTING","已有眼球","指定现有独立对象；有眼皮即可制作眨眼"),
           ("SOCKET","有眼孔，缺眼球","选择一圈眼缘边并生成眼球"),
           ("PAINTED","贴图眼，没有眼孔","明确选中要隐藏的眼面，保留原始网格")]
def eye_object_filter(self,obj):
    return obj.type=="MESH" and obj!=self.id_data

class Parts(bpy.types.PropertyGroup):
    task:bpy.props.EnumProperty(name="编辑部位",items=[("KEYS","已有表情","直接预览模型上的键"),
        ("MOUTH","嘴部","按嘴部现状选择操作"),("EYES","眼部","分别设置两眼")])
    mouth:bpy.props.EnumProperty(name="嘴部现状",items=[("READY","已有张嘴键","直接试用现有表情"),
        ("OPEN","上下唇可分开","从唇缘制作张嘴键"),("SEALED","嘴缝黏合","检测并切开连通面")])
    oral:bpy.props.EnumProperty(name="口腔袋",items=[("KEEP","保留 / 跳过","保留现有口腔"),
        ("ADD","缺少口腔袋","选择需要的袋口结构")])
    speech_jaw:bpy.props.StringProperty(name="Lower jaw group")
    use_incisor:bpy.props.BoolProperty(name="FF 使用上门牙参考点")
    teeth:bpy.props.BoolProperty(name="缺少牙列")
    tongue:bpy.props.BoolProperty(name="缺少舌体")
    key_name:bpy.props.StringProperty(name="形态键")
    side:bpy.props.EnumProperty(name="眼睛",items=eyes.SIDES)
    route_NegX:bpy.props.EnumProperty(name="该眼现状",items=EYE_TYPES)
    route_PosX:bpy.props.EnumProperty(name="该眼现状",items=EYE_TYPES)
    eye_NegX:bpy.props.PointerProperty(name="眼球对象",type=bpy.types.Object,poll=eye_object_filter)
    eye_PosX:bpy.props.PointerProperty(name="眼球对象",type=bpy.types.Object,poll=eye_object_filter)


def head_for(context):
    obj=context.active_object
    return obj if obj and obj.type=="MESH" else None


def key_picker(ui,obj):
    shape=obj.data.shape_keys
    if not shape or len(shape.key_blocks)<2:
        ui.label(text="这个网格还没有表情键。")
        return
    cfg=obj.faceauto_parts
    ui.prop_search(cfg,"key_name",shape,"key_blocks",text="选择已有形态键")
    k=shape.key_blocks.get(cfg.key_name)
    if k and k!=shape.reference_key:
        ui.prop(k,"value",text=k.name,slider=True)


class FACEAUTO_PT_parts(bpy.types.Panel):
    bl_space_type="VIEW_3D"
    bl_region_type="UI"
    bl_category="FaceAuto"
    bl_label="模型与表情"
    bl_order=-10

    def draw(self,context):
        ui=self.layout
        obj=head_for(context)
        if not obj:
            ui.label(text="选中角色的脸部网格。")
            return
        s=obj.faceauto_parts
        ui.label(text=obj.name,icon="MESH_DATA")
        ui.prop(s,"task",expand=True)
        if s.task=="KEYS":
            key_picker(ui,obj)
        elif s.task=="MOUTH":
            ui.prop(s,"mouth")
            ui.prop(s,"oral")
            row=ui.row(align=True)
            row.prop(s,"teeth")
            row.prop(s,"tongue")
            if s.mouth=="READY":
                key_picker(ui,obj)
            elif s.mouth=="OPEN":
                ui.label(text="记录上下唇缘后即可生成张嘴键。")
                ui.operator("faceauto.create_opening")
                ui.operator("faceauto.create_upper")
                for name in (*(layout(obj) or ()),opening.UPPER_KEY):
                    if obj.data.shape_keys and name in obj.data.shape_keys.key_blocks:
                        ui.prop(obj.data.shape_keys.key_blocks[name],"value",text=name,slider=True)
            if s.oral=="ADD":
                ui.operator("faceauto.build_lining",text="从开放切边补建相连口腔袋")
                ui.operator("faceauto.oral_proxy",text="补独立口腔袋").part="CAVITY"
            if s.teeth:
                ui.operator("faceauto.oral_proxy",text="仅补牙列").part="TEETH"
            if s.tongue:
                ui.operator("faceauto.oral_proxy",text="仅补舌体").part="TONGUE"
        else:
            ui.prop(s,"side",expand=True)
            route="route_"+s.side
            ui.prop(s,route)
            mode=getattr(s,route)
            ui.prop(s,"eye_"+s.side)
            edit=context.mode=="EDIT_MESH"
            row=ui.column()
            row.enabled=edit
            if mode=="PAINTED":
                ui.label(text="编辑模式：选中一只眼的眼白和虹膜面。")
                row.operator("faceauto.eye_region",text="记录眼缘并隐藏选中眼面").painted=True
            else:
                ui.label(text="编辑模式：选中该眼的整圈眼睑边缘。")
                row.operator("faceauto.eye_region",text="记录选中的眼缘边环").painted=False
            eye=eyes.eye_ref(obj,s.side)
            if mode!="EXISTING" and not eye:
                row=ui.row()
                row.enabled=context.mode=="OBJECT"
                row.operator("faceauto.eye_action",text="生成独立眼球").action="BALL"
            if eye and eye.rotation_mode not in ("QUATERNION","AXIS_ANGLE"):
                ui.prop(eye,"rotation_euler",index=2,text="眼球水平转角")
                ui.prop(eye,"rotation_euler",index=0,text="眼球俯仰转角")
            elif eye:
                ui.prop(eye,"rotation_quaternion" if eye.rotation_mode=="QUATERNION" else "rotation_axis_angle",text="眼球旋转")
            name="FaceAuto_Blink_"+s.side
            if obj.data.shape_keys and name in obj.data.shape_keys.key_blocks:
                ui.prop(obj.data.shape_keys.key_blocks[name],"value",text="眨眼",slider=True)
            else:
                row=ui.row()
                row.enabled=context.mode=="OBJECT"
                row.operator("faceauto.eye_action",text="从眼睑生成眨眼形态键").action="BLINK"


