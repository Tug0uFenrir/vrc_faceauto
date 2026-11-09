"""Calibration of the prepared ICT and Ada aperture meshes."""
import numpy as np

# Inner lip loops of ICT-FaceKit in native OBJ vertex order, traced by the ICT experiment.
ICT_UPPER = [5674,5646,5915,5596,5595,5579,5578,5589,5606,5630,5657,5656,5711,5712,5736,6037,6023,5781,5796,5533,6301,6286,6527,6541,6241,6217,6216,6161,6162,6135,6111,6094,6083,6084,6100,6101,6420,6151,6179]
ICT_LOWER = [5700,5727,5730,5995,5754,5895,5802,5820,5836,5847,5858,5867,5966,5952,5981,6009,5880,5938,5517,6443,6385,6513,6485,6456,6470,6372,6363,6352,6341,6325,6307,6400,6259,6499,6235,6232,6205]
LAYOUTS = [
    ("Open_L","Open_C","Open_R"), ("LowerLip_NegX","LowerLip_Center","LowerLip_PosX"),
    ("Open_Sides","Open_C"), ("Open_L_R","Open_C"), ("Open_All",), ("Open_L_C_R",), ("FaceAuto_TestOpen",),
]


def layout(obj):
    if obj and obj.type == "MESH" and obj.data.shape_keys:
        blocks = obj.data.shape_keys.key_blocks
        return next((names for names in LAYOUTS if all(n in blocks for n in names)), ())


def grouping(n, budget):
    if budget > n:
        raise ValueError(f"此文件只有 {n} 个独立开口键，请调低控制数或打开三键版。")
    if budget == 1:
        return [list(range(n))]
    if budget == 2 and n == 3:
        return [[0,2],[1]]
    return [[i] for i in range(n)]


def coords(vertices):
    data = np.empty(3*len(vertices), dtype="f")
    vertices.foreach_get("co", data)
    return data.reshape(-1,3).astype(float)


def measurement(obj):
    p = coords(obj.data.shape_keys.reference_key.data)
    seam_groups=[obj.vertex_groups.get(n) for n in ("FaceAuto_SeamUpper","FaceAuto_SeamLower")]
    if all(g is not None for g in seam_groups):
        rims=[np.array(sorted((v.index for v in obj.data.vertices if any(w.group==g.index and w.weight>0 for w in v.groups)),key=lambda i:p[i,0])) for g in seam_groups]
        if min(map(len,rims))<3:
            raise ValueError("切边组至少各需三个顶点。")
        left=max(p[r[0],0] for r in rims)
        width=min(p[r[-1],0] for r in rims)-left
        if width<=0:
            raise ValueError("切边没有共同嘴宽。")
        sites=left+width*np.array([.25,.5,.75])
        def seam_gaps(xyz):
            hi,lo=[np.interp(sites,p[r,0],xyz[r,2]) for r in rims]
            return 100*(hi-lo)/width
        return seam_gaps
    groups = [obj.vertex_groups.get(name) for name in ("Scan_upper_lip","Scan_lower_lip")]
    if all(g is not None for g in groups):
        rims = [np.array([v.index for v in obj.data.vertices if any(g.group==group.index for g in v.groups)]) for group in groups]
        if min(map(len,rims)) < 3:
            raise ValueError("上下唇顶点组都需要至少三个顶点。")
        native = (len(p)==26719 and len(obj.data.polygons)==26384 and
                  all(set(ids)==set(ref) for ids,ref in zip(rims,(ICT_UPPER,ICT_LOWER))))
        if not native:
            xs = p[rims[1],0]
            width = np.ptp(xs)
            if width <= 0:
                raise ValueError("下唇顶点组沿 X 轴的宽度为零。")
            sites = xs.min() + width*np.array([.25,.5,.75])
            hi,lo = [r[np.abs(p[r,0,None]-sites).argmin(axis=0)] for r in rims]
            return lambda xyz: 100*(xyz[hi,2]-xyz[lo,2])/width
    if len(p) != 26719 or len(obj.data.polygons) != 26384:
        raise ValueError("无法定位唇缘。请选择 ICT 完整拓扑或带上下唇扫描组的模型。")
    edges = {tuple(sorted(e.vertices)) for e in obj.data.edges}
    for rim in (ICT_UPPER,ICT_LOWER):
        if any(tuple(sorted(pair)) not in edges for pair in zip(rim,rim[1:])):
            raise ValueError("ICT 唇缘顶点顺序不匹配。")
    upper,lower = [np.array(r)[np.argsort(p[r,0])] for r in (ICT_UPPER,ICT_LOWER)]
    left = max(p[upper[0],0],p[lower[0],0])
    width = min(p[upper[-1],0],p[lower[-1],0])-left
    if width <= 0:
        raise ValueError("无法测量中性嘴宽。")
    sites = left + width*np.array([.25,.5,.75])
    def gaps(xyz):
        hi,lo = [np.interp(sites,p[r,0],xyz[r,2]) for r in (upper,lower)]
        return 100*(hi-lo)/width
    return gaps


