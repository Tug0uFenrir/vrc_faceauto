"""Detect a surface patch bounded by the two marked lip rims and split its midline."""
import heapq
import hashlib
import json
import math

import bpy
import bmesh
import numpy as np


def route(start, end, neighbors, cost):
    distances={start:0.}
    parent={}
    queue=[(0.,start.index,start)]
    while queue:
        d,_,v=heapq.heappop(queue)
        if d!=distances[v]:
            continue
        if v==end:
            path=[end]
            while path[-1]!=start:
                path.append(parent[path[-1]])
            return path[::-1]
        for n in neighbors(v):
            nd=d+cost(v,n)
            if nd<distances.get(n,math.inf):
                distances[n],parent[n]=nd,v
                heapq.heappush(queue,(nd,n.index,n))
    raise ValueError("局部表面没有可用的连通路径。")


def edges(bm,path):
    return {bm.edges.get((a,b)) for a,b in zip(path,path[1:])}


def analyse(obj,bm):
    deform=bm.verts.layers.deform.active
    if deform is None:
        raise ValueError("请先建立上下唇缘组。")
    groups=[obj.vertex_groups.get("Scan_"+side+"_lip") for side in ("upper","lower")]
    if any(g is None for g in groups):
        raise ValueError("请先建立上下唇缘组。")
    rims=[[],[]]
    for v in bm.verts:
        weights=v[deform]
        for rim,g in zip(rims,groups):
            if weights.get(g.index,0)>0:
                rim.append(v)
    for rim in rims:
        rim.sort(key=lambda v:v.co.x)
        if len(rim)<3 or None in edges(bm,rim):
            raise ValueError("唇缘组须是按局部 X 排列的连续边链。")
    upper,lower=rims
    width=min(r[-1].co.x for r in rims)-max(r[0].co.x for r in rims)
    if width<=0 or any(v.hide for rim in rims for v in rim):
        raise ValueError("请显示两条唇缘，并检查嘴宽。")
    rimverts=set(upper+lower)
    caps=[]
    for i in (0,-1):
        a,b=upper[i],lower[i]
        center=(a.co+b.co)*.5
        radius=max((a.co-b.co).length,.07*width)*1.4
        def adjacent(v):
            for e in v.link_edges:
                n=e.other_vert(v)
                if not e.hide and (n==b or n not in rimverts) and (n.co-center).length<radius:
                    yield n
        caps.append(route(a,b,adjacent,lambda v,n:(n.co-v.co).length))
    wall=set().union(*(edges(bm,r) for r in rims+caps))
    if any(e is None for e in wall):
        raise ValueError("唇缘或端点之间的路径不连续。")
    # A closed, bounded face strip must touch every perimeter edge and have no
    # other boundary. This rejects disconnected rims and an already opened slit.
    start=bm.edges.get((upper[len(upper)//2-1],upper[len(upper)//2]))
    xmin=min(v.co.x for v in rimverts)-.18*width
    xmax=max(v.co.x for v in rimverts)+.18*width
    origin=sum((v.co for v in rimverts),upper[0].co*0)/len(rimverts)
    def flood(seed):
        seen={seed}
        todo=[seed]
        while todo:
            f=todo.pop()
            c=f.calc_center_median()
            if f.hide or not xmin<c.x<xmax or (c-origin).length>width or len(seen)>15000:
                return set()
            for e in f.edges:
                if e not in wall:
                    for n in e.link_faces:
                        if n not in seen:
                            seen.add(n)
                            todo.append(n)
        boundary={e for f in seen for e in f.edges if len([n for n in e.link_faces if n in seen])==1}
        return seen if boundary==wall else set()
    patch=set()
    for seed in start.link_faces:
        patch=flood(seed)
        if patch:
            break
    if not patch:
        raise ValueError("没有找到由两条唇缘围住的完整连通面带，不能自动切开。")
    xs=[[v.co.x for v in r] for r in rims]
    zs=[[v.co.z for v in r] for r in rims]
    def mid(v):
        return np.mean([np.interp(v.co.x,x,z) for x,z in zip(xs,zs)])
    ends=[min(cap,key=lambda v:abs(v.co.z-mid(v))) for cap in caps]
    allowed={v for f in patch for v in f.verts}
    def adjacent(v):
        return [e.other_vert(v) for e in v.link_edges if e.other_vert(v) in allowed and e not in wall
                and len(e.link_faces)==2 and all(f in patch for f in e.link_faces)]
    def cost(v,n):
        return (n.co-v.co).length*(1+1800*((n.co.z-mid(n))/width)**2)
    try:
        seam=route(*ends,adjacent,cost)
    except ValueError:
        alternatives=[]
        for a in caps[0]:
            for b in caps[1]:
                try:
                    line=route(a,b,adjacent,cost)
                    score=sum(cost(v,n) for v,n in zip(line,line[1:]))+3*(abs(a.co.z-mid(a))+abs(b.co.z-mid(b)))
                    alternatives.append((score,line))
                except ValueError:
                    continue
        seam=min(alternatives,key=lambda item:item[0])[1] if alternatives else []
    return patch,seam


def token(patch,seam):
    data=[sorted((f.index,tuple((v.index,*v.co) for v in f.verts)) for f in patch),[v.index for v in seam]]
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


class FACEAUTO_OT_seam(bpy.types.Operator):
    bl_idname="faceauto.seam"
    bl_label="嘴缝连通检测与切开"
    bl_options={"REGISTER","UNDO"}
    cut:bpy.props.BoolProperty(default=False)

    @classmethod
    def poll(cls,context):
        return context.mode=="EDIT_MESH" and len(context.objects_in_mode)==1

    def execute(self,context):
        obj=context.edit_object
        bm=bmesh.from_edit_mesh(obj.data)
        bm.faces.index_update()
        bm.verts.index_update()
        try:
            patch,seam=analyse(obj,bm)
            stamp=token(patch,seam)
            if self.cut:
                if obj.get("faceauto_seam_preview")!=stamp or not seam:
                    raise ValueError("请先预览有效面带和切线。")
                if obj.data.shape_keys or obj.data.users!=1:
                    raise ValueError("请在无形态键、单用户的副本上切缝。")
                line=edges(bm,seam)
                if len(line)<2 or None in line or any(not e.is_manifold for e in line):
                    raise ValueError("切线须为连续的双侧面内边。")
                # Label the two banks using the pre-cut upper/lower face components.
                upper_group=obj.vertex_groups["Scan_upper_lip"].index
                dv=bm.verts.layers.deform.active
                left=set(patch)
                todo=[next(iter(patch))]
                component=set(todo)
                while todo:
                    f=todo.pop()
                    for e in f.edges:
                        if e not in line:
                            for n in e.link_faces:
                                if n in left and n not in component:
                                    component.add(n)
                                    todo.append(n)
                other=left-component
                if not other:
                    raise ValueError("候选线未将面带分成两侧。")
                def upper_score(fs):
                    return sum(v[dv].get(upper_group,0)>0 for v in {v for f in fs for v in f.verts})
                if upper_score(component)<upper_score(other):
                    component,other=other,component
                names=("FaceAuto_SeamUpper","FaceAuto_SeamLower")
                if any(obj.vertex_groups.get(n) for n in names):
                    raise ValueError("已有切缝边界组，请在未切开的副本上运行。")
                locations={tuple(v.co) for v in seam}
                bmesh.ops.split_edges(bm,edges=list(line))
                for name,fs in zip(names,(component,other)):
                    group=obj.vertex_groups.new(name=name)
                    for v in {v for f in fs for v in f.verts if tuple(v.co) in locations}:
                        v[dv][group.index]=1.
                obj.pop("faceauto_seam_preview",None)
            else:
                bpy.ops.mesh.select_all(action="DESELECT")
                context.tool_settings.mesh_select_mode=(False,False,True)
                for f in patch:
                    f.select_set(True)
                obj["faceauto_seam_preview"]=stamp
                self.report({"INFO"},f"高亮 {len(patch)} 个面，候选切线 {max(0,len(seam)-1)} 条边")
            bmesh.update_edit_mesh(obj.data,destructive=self.cut)
        except ValueError as exc:
            self.report({"WARNING"},str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class FACEAUTO_PT_seam(bpy.types.Panel):
    bl_space_type="VIEW_3D"
    bl_region_type="UI"
    bl_category="FaceAuto"
    bl_label="嘴缝连通"
    bl_order=1

    @classmethod
    def poll(cls,context):
        obj=context.active_object
        return obj is not None and obj.type=="MESH" and obj.faceauto_parts.task=="MOUTH" and obj.faceauto_parts.mouth=="SEALED"

    def draw(self,context):
        ui=self.layout
        ui.operator("faceauto.seam",text="检测并高亮跨缝面带").cut=False
        row=ui.row()
        obj=context.edit_object
        row.enabled=bool(obj and obj.get("faceauto_seam_preview"))
        row.operator("faceauto.seam",text="沿面带中央切开").cut=True
        from .calibration import layout
        head=context.active_object
        names=layout(head)
        if names and head.vertex_groups.get("FaceAuto_SeamLower") is not None:
            if context.mode=="OBJECT":
                for name in names:
                    ui.prop(head.data.shape_keys.key_blocks[name],"value",text="张嘴 "+name,slider=True)
            else:
                ui.label(text="按 Tab 切回物体模式调节张嘴。")
        else:
            ui.operator("faceauto.create_opening",text="生成张嘴形态键并预览")
        from .opening import UPPER_KEY
        blocks=head.data.shape_keys.key_blocks if head and head.type=="MESH" and head.data.shape_keys else ()
        if UPPER_KEY in blocks:
            if context.mode=="OBJECT":
                ui.prop(blocks[UPPER_KEY],"value",text="上唇上提",slider=True)
        else:
            ui.operator("faceauto.create_upper",text="生成上唇上提形态键")

