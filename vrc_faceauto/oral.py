"""Optional, editable oral placeholders fitted behind FaceAuto's split banks."""
from math import cos, sin, pi
import bpy
import numpy as np
from .opening import rim_names


class FACEAUTO_OT_oral(bpy.types.Operator):
    bl_idname="faceauto.oral_proxy"
    bl_label="生成简化口腔、牙列和舌体"
    bl_options={"REGISTER","UNDO"}

    part:bpy.props.EnumProperty(items=[("ALL","组合",""),("TEETH","牙列",""),("TONGUE","舌头",""),("CAVITY","口腔袋","")],default="ALL")

    @classmethod
    def poll(cls,context):
        obj=context.active_object
        return (context.mode=="OBJECT" and obj is not None and obj.type=="MESH"
                and all(obj.vertex_groups.get(n) for n in rim_names(obj)))

    def execute(self,context):
        head=context.active_object
        geom=head.data.shape_keys.reference_key.data if head.data.shape_keys else head.data.vertices
        groups=[head.vertex_groups[n].index for n in rim_names(head)]
        rims=[[],[]]
        for v in head.data.vertices:
            for g in v.groups:
                if g.group in groups and g.weight>0:
                    rims[groups.index(g.group)].append(v.index)
        if min(map(len,rims))<3:
            self.report({"ERROR"},"上下切缝边界组至少各需三个点。")
            return {"CANCELLED"}
        xs=[geom[i].co.x for i in rims[0]]
        width=max(xs)-min(xs)
        if width<=0:
            return {"CANCELLED"}
        cx=(max(xs)+min(xs))/2
        anchors=[min(ids,key=lambda i:abs(geom[i].co.x-cx)) for ids in rims]
        origin=np.array(geom[anchors[0]].co)
        origin[1] += .52*width  # Recess the fixture behind the lower lip and cheeks.
        generated=[]
        def attach(name,verts,faces,bank,color):
            kind="CAVITY" if name.endswith("cavity") else "TONGUE" if name.endswith("Tongue") else "TEETH"
            if self.part not in ("ALL",kind) or any(o.get("faceauto_oral") and o.name.startswith(name) for o in head.children):
                return
            generated.append(name)
            anchor=anchors[bank]
            anchor_co=np.asarray(geom[anchor].co)
            mesh=bpy.data.meshes.new(name)
            mesh.from_pydata((origin+width*np.asarray(verts)-anchor_co).tolist(),[],faces)
            mesh.update()
            obj=bpy.data.objects.new(name,mesh)
            context.collection.objects.link(obj)
            obj.parent=head
            if bank:
                obj.parent_type="VERTEX"
                obj.parent_vertices=(anchor,0,0)
            else:
                obj.location=anchor_co
            obj["faceauto_oral"]=True
            mat=bpy.data.materials.new(name)
            mat.diffuse_color=(*color,1)
            mesh.materials.append(mat)
            obj.color=(*color,1)
            for f in mesh.polygons:
                f.use_smooth=True
        def ellipsoid(verts,faces,center,scale,exponent):
            # Squared-off crowns, rounded tongue, with a single vertex at each pole.
            offset=len(verts)
            signed=lambda t:np.sign(t)*abs(t)**exponent
            verts.append((center[0],center[1],center[2]-scale[2]))
            count=16
            for j in range(1,10):
                a=-pi/2+pi*j/10
                for k in range(count):
                    b=2*pi*k/count
                    verts.append(np.array(center)+scale*np.array([signed(cos(a)*cos(b)),signed(cos(a)*sin(b)),signed(sin(a))]))
            top=len(verts)
            verts.append((center[0],center[1],center[2]+scale[2]))
            for k in range(count):
                i=offset+1+k
                n=offset+1+(k+1)%count
                faces.append((offset,n,i))
                for j in range(8):
                    a,b=i+j*count,n+j*count
                    faces.append((a,b,b+count,a+count))
                faces.append((i+8*count,n+8*count,top))
        verts=[];faces=[]
        for depth,rx,rz in ((.10,.69,.57),(.35,.66,.57),(.65,.54,.51),(.84,.20,.30)):
            for k in range(32):
                a=2*pi*k/32
                verts.append((rx*cos(a),depth,-.12+rz*sin(a)))
        for j in range(3):
            for k in range(32):
                i=j*32+k;n=j*32+(k+1)%32
                faces.append((i,n,n+32,i+32))
        faces.append(tuple(range(96,128)))
        if head.vertex_groups.get("FaceAuto_OralLining") is None:
            attach("FaceAuto Oral cavity",verts,faces,0,(.16,.037,.046))
        for bank,name in enumerate(("FaceAuto Upper teeth","FaceAuto Lower teeth")):
            verts=[];faces=[]
            for j in range(10):
                x=(j-4.5)*.085
                ellipsoid(verts,faces,(x,.14+.7*x*x,(.025 if bank==0 else -.115)+.13*x*x),
                          np.array([.041,.034,.061]),.55)
            attach(name,verts,faces,bank,(.86,.77,.62))
        verts=[];faces=[]
        ellipsoid(verts,faces,(0,.42,-.09),np.array([.27,.15,.065]),.82)
        attach("FaceAuto Tongue",verts,faces,1,(.62,.17,.21))
        if not generated:
            self.report({"INFO"},"所选部件已存在。")
        return {"FINISHED"}





