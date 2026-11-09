"""Six editable speech shapes, calibrated from neutral lip contours in face-local X/Z."""
import bmesh
import bpy
import numpy as np

from .calibration import coords
from .fields import extend_field
from .opening import rim_names

# width ratio, central gap / neutral width, forward travel / width, upper opening share
POSES = {
    "pp": (.99, .0, .008, .45),
    "ff": (1.01, .085, .004, .70),
    "aa": (1.01, .31, .005, .22),
    "ih": (1.14, .105, -.012, .42),
    "oh": (.78, .28, .072, .36),
    "ou": (.60, .145, .125, .46),
}
JAW_DROP={"pp":0., "ff":.035, "aa":.21, "ih":.05, "oh":.14, "ou":.06}
CORNERS="FaceAuto_SpeechCorners"
PREFIX="vrc.v_"


class FACEAUTO_OT_speech_mark(bpy.types.Operator):
    bl_idname="faceauto.speech_mark"
    bl_label="记录说话口型参考点"
    bl_options={"UNDO"}
    tooth:bpy.props.BoolProperty(default=False)

    def execute(self,context):
        obj=context.active_object
        if self.tooth:
            if not obj or obj.type!="MESH":
                return {"CANCELLED"}
            obj["faceauto_incisor"]=list(obj.matrix_world.inverted_safe()@context.scene.cursor.location)
            return {"FINISHED"}
        if context.mode!="EDIT_MESH" or len(context.objects_in_mode)!=1 or obj.data.users!=1:
            self.report({"ERROR"},"请单独编辑脸部网格，在左右嘴角各选一个顶点。")
            return {"CANCELLED"}
        bm=bmesh.from_edit_mesh(obj.data)
        ids=[v.index for v in bm.verts if v.select and not v.hide]
        if len(ids)!=2:
            self.report({"ERROR"},"只选两个顶点，左右嘴角各一个。")
            return {"CANCELLED"}
        group=obj.vertex_groups.get(CORNERS)
        if group and group.lock_weight:
            self.report({"ERROR"},"嘴角顶点组已锁定。")
            return {"CANCELLED"}
        if group is None:
            group=obj.vertex_groups.new(name=CORNERS)
        layer=bm.verts.layers.deform.verify()
        for v in bm.verts:
            if v.index in ids:
                v[layer][group.index]=1
            elif group.index in v[layer]:
                del v[layer][group.index]
        bmesh.update_edit_mesh(obj.data,destructive=False)
        return {"FINISHED"}


def landmarks(obj,p):
    groups=[obj.vertex_groups.get(n) for n in (*rim_names(obj),CORNERS)]
    if any(g is None for g in groups):
        raise ValueError("先写入上下唇缘组，再记录左右两个嘴角。")
    by_group={g.index:[] for g in groups}
    for v in obj.data.vertices:
        for w in v.groups:
            if w.group in by_group and w.weight>0:
                by_group[w.group].append(v.index)
    upper,lower,corners=[np.array(sorted(ids,key=lambda i:p[i,0]),dtype=int) for ids in by_group.values()]
    if len(corners)!=2 or min(len(upper),len(lower))<3:
        raise ValueError("需要两个嘴角，上下唇缘各至少三个顶点。")
    width=np.ptp(p[corners,0])
    if width<=0:
        raise ValueError("嘴角沿局部 X 的宽度为零。")
    for rim in (upper,lower):
        if np.max(np.linalg.norm(p[corners]-p[rim[[0,-1]]],axis=1))>.24*width:
            raise ValueError("嘴角离唇缘端点过远，请复核标记和模型朝向。")
    if set(upper[1:-1]) & set(lower[1:-1]):
        raise ValueError("上下唇中段共用顶点，请先分开嘴缝。")
    return upper,lower,corners,width


def generate(obj):
    mesh=obj.data
    keys=mesh.shape_keys
    if keys and not keys.use_relative:
        raise ValueError("请使用相对形态键。")
    available=[n for n in POSES if not keys or PREFIX+n not in keys.key_blocks]
    if not available:
        return []
    p=coords(keys.reference_key.data if keys else mesh.vertices)
    hi,lo,corners,width=landmarks(obj,p)
    start,end=p[corners,0]
    center=(start+end)/2
    tooth=obj.get("faceauto_incisor") if obj.faceauto_parts.use_incisor else None
    if obj.faceauto_parts.use_incisor and (tooth is None or
            np.linalg.norm(np.asarray(tooth)-p[hi[np.argmin(abs(p[hi,0]-center))]])>.65*width):
        raise ValueError("请将游标放在上门牙中央的下缘，重新记录牙齿参考点。")
    anchors=np.unique(np.r_[hi,lo,corners])
    x=p[anchors,0]
    t=np.clip((x-start)/width,0,1)
    arch=np.sin(np.pi*t)**.85
    upper=np.isin(anchors,hi)
    corner=np.isin(anchors,corners)
    deltas=[]
    def sample(rim,axis):
        return np.interp(x,p[rim,0],p[rim,axis])
    mid_z=(sample(hi,2)+sample(lo,2))*.5
    mid_y=(sample(hi,1)+sample(lo,1))*.5
    gap=sample(hi,2)-sample(lo,2)
    if gap[np.argmin(abs(x-center))]<-.02*width:
        raise ValueError("中央上唇低于下唇，请检查上下标签和局部 Z 轴。")
    for name in available:
        scale,height,forward,share=POSES[name]
        # At matching x: G = height*width*arch, dz_upper - dz_lower = G - gap.
        d=np.zeros((len(anchors),3))
        d[:,0]=(scale-1)*(x-center)
        d[:,1]=-width*forward*(.45+.55*arch)
        d[:,2]=(height*width*arch-gap)*np.where(upper,share,-(1-share))
        if name in ("oh","ou"):
            # Narrow the visible opening inside the mouth corners. Keep the side
            # seals slightly parted and retain a smooth, round central aperture.
            u=2*t-1
            breadth=.82 if name=="oh" else .60
            oval=np.exp(-1.45*(u/breadth)**4)*np.sqrt(np.maximum(0.,1-u*u))
            d[:,0]=(x-center)*(scale*(.76+.24*u*u)-1)
            d[:,2]=(height*width*oval-gap)*np.where(upper,share,-(1-share))
        if name=="pp":
            d[:,1]+=(mid_y-p[anchors,1])*arch
            d[:,2]=mid_z-p[anchors,2]+np.where(upper,1,-1)*np.clip(gap,0,.004*width)*.5*arch
        if name=="ff":
            if tooth is None:
                # Curl the lower lip inward and upward below the raised upper lip.
                d[~upper,1]+=.047*width*arch[~upper]
                d[~upper,2]+=.047*width*arch[~upper]
                d[upper,2]+=.036*width*arch[upper]
            else:
                # Target a point just in front of the marked upper incisor cutting edge.
                blend=arch[~upper]**3
                for axis,clearance in ((1,-.04),(2,-.006)):
                    d[~upper,axis]=(1-blend)*d[~upper,axis]+blend*(tooth[axis]+clearance*width-p[anchors[~upper],axis])
        d[corner,2]=0
        deltas.append(d)
    prescribed=np.column_stack(deltas)
    ids,field=extend_field(obj,p,dict(zip(anchors,prescribed)),width)
    jaw_ids=[]
    jaw_weights=[]
    if obj.faceauto_parts.speech_jaw:
        group=obj.vertex_groups.get(obj.faceauto_parts.speech_jaw)
        if group is None:
            raise ValueError("选定的下颌组不存在。")
        for v in mesh.vertices:
            for weight in v.groups:
                if weight.group==group.index and weight.weight>0:
                    jaw_ids.append(v.index)
                    jaw_weights.append(weight.weight)
        if not jaw_ids or set(jaw_ids).intersection(anchors):
            raise ValueError("下颌组应包含下牙或口内组织，并排除唇缘和嘴角。")
    created=[]
    try:
        if not keys:
            created.append(obj.shape_key_add(name="Basis",from_mix=False))
        for j,name in enumerate(available):
            shape=obj.shape_key_add(name=PREFIX+name,from_mix=False)
            created.append(shape)
            target=p.copy()
            target[ids]+=field[:,3*j:3*j+3]
            if jaw_ids:
                influence=np.array(jaw_weights)[:,None]
                jaw_delta=np.array([0.,.12,-1.])*width*JAW_DROP[name]
                target[jaw_ids]+=influence*(p[jaw_ids]+jaw_delta-target[jaw_ids])
            shape.data.foreach_set("co",target.astype("f").ravel())
            shape.value=0
    except Exception:
        for k in reversed(created):
            obj.shape_key_remove(k)
        raise
    return available


class FACEAUTO_OT_speech_generate(bpy.types.Operator):
    bl_idname="faceauto.speech_generate"
    bl_label="生成缺少的六个口型"
    bl_options={"REGISTER","UNDO"}

    @classmethod
    def poll(cls,context):
        obj=context.active_object
        return context.mode=="OBJECT" and obj and obj.type=="MESH" and obj.data.users==1

    def execute(self,context):
        try:
            added=generate(context.active_object)
        except ValueError as err:
            self.report({"ERROR"},str(err))
            return {"CANCELLED"}
        self.report({"INFO"},"新增 "+", ".join(added) if added else "六个口型均已存在。")
        return {"FINISHED"}


class FACEAUTO_PT_speech(bpy.types.Panel):
    bl_space_type="VIEW_3D"
    bl_region_type="UI"
    bl_category="FaceAuto"
    bl_label="说话口型"
    bl_order=2

    @classmethod
    def poll(cls,context):
        obj=context.active_object
        return obj and obj.type=="MESH" and obj.faceauto_parts.task=="MOUTH"

    def draw(self,context):
        obj=context.active_object
        ui=self.layout
        ui.label(text="标记内唇缘，编辑模式各选一个嘴角。")
        row=ui.row()
        row.enabled=context.mode=="EDIT_MESH"
        row.operator("faceauto.speech_mark",text="记录左右嘴角")
        ui.prop_search(obj.faceauto_parts,"speech_jaw",obj,"vertex_groups",text="下颌跟随组（可选）")
        ui.prop(obj.faceauto_parts,"use_incisor")
        if obj.faceauto_parts.use_incisor:
            ui.operator("faceauto.speech_mark",text="用游标记录上门牙下缘").tooth=True
        ui.operator("faceauto.speech_generate")
        keys=obj.data.shape_keys
        if keys and context.mode=="OBJECT":
            for n in POSES:
                k=keys.key_blocks.get(PREFIX+n)
                if k:
                    ui.prop(k,"value",text=n.upper(),slider=True)


