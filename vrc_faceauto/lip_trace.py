"""Trace inner lip edges from user-labelled mesh vertices in local X/Z coordinates."""
from heapq import heappop, heappush
import numpy as np


def topology(faces, count):
    adjacent = [set() for _ in range(count)]
    turns = {}
    for f in faces:
        for i, tip in enumerate(f):
            a, b = f[i-1], f[(i+1) % len(f)]
            adjacent[tip].update((a,b))
            for prev, nxt in ((a,b),(b,a)):
                turns.setdefault((prev,tip),set()).add(nxt)
    return adjacent, turns


def automatic(p, seed, adjacent, turns):
    branches = []
    for sign in (-1,1):
        if not adjacent[seed]:
            raise ValueError("种子是孤立顶点。")
        nxt = max(adjacent[seed], key=lambda v: sign*(p[v,0]-p[seed,0]))
        if sign*(p[nxt,0]-p[seed,0]) <= 0:
            raise ValueError("请在唇缘中央取种子点。")
        path = [seed,nxt]
        while True:
            prev,tip = path[-2:]
            options = adjacent[tip] - turns[(prev,tip)] - {prev}
            if len(options)!=1:
                raise ValueError("边流在此处分叉，请改用含两端点的多点模式。")
            v = options.pop()
            if sign*(p[v,0]-p[tip,0]) <= 0:
                break
            path.append(v)
        branches.append(path)
    return branches[0][:0:-1]+branches[1]


def segment(p, adjacent, start, end, blocked):
    # Positive-X paths stay on connected mesh edges. Penalize departures from the
    # 3D chord to avoid cutting across a distant fold between two user anchors.
    # cost(i,j) = ||p_j-p_i||*(1+4*d_j/L), with perpendicular distance to the anchor line.
    axis = p[end]-p[start]
    length = np.linalg.norm(axis)
    if length == 0 or axis[0] <= 0:
        raise ValueError("同一条唇缘上的引导点需要沿局部 X 轴分开。")
    unit = axis/length
    distance, parent = {start:0.0}, {}
    queue = [(0.0,start)]
    while queue:
        cost, a = heappop(queue)
        if cost != distance[a]:
            continue
        if a == end:
            path=[end]
            while path[-1]!=start:
                path.append(parent[path[-1]])
            return path[::-1]
        for b in adjacent[a]:
            if b in blocked or not p[a,0] < p[b,0] <= p[end,0]:
                continue
            offset=p[b]-p[start]
            deviation=np.linalg.norm(offset-unit*np.dot(offset,unit))
            score=cost+np.linalg.norm(p[b]-p[a])*(1+4*deviation/length)
            if score < distance.get(b,np.inf):
                distance[b],parent[b]=score,a
                heappush(queue,(score,b))
    raise ValueError("引导点之间没有沿局部 X 递增的连通边，请检查唇缘或补点。")


def trace(points, faces, upper, lower, guided):
    p=np.asarray(points)
    adjacent,turns=topology(faces,len(p))
    paths=[]
    for seeds,other in ((upper,lower),(lower,upper)):
        seeds=sorted(set(seeds),key=lambda v:p[v,0])
        if guided:
            if len(seeds)<2:
                raise ValueError("多点模式需要在每条唇缘记录两个端点，可加中间引导点。")
            path=[seeds[0]]
            for a,b in zip(seeds,seeds[1:]):
                blocked=set(other)-{a,b}
                path.extend(segment(p,adjacent,a,b,blocked)[1:])
        else:
            if len(seeds)!=1:
                raise ValueError("自动模式请分别记录一个上唇中央点和一个下唇中央点。")
            path=automatic(p,seeds[0],adjacent,turns)
        if len(path)<2:
            raise ValueError("找到的唇缘太短。")
        paths.append(path)
    up,lo=paths
    if (set(up)&set(lo)) - ({up[0],up[-1]} & {lo[0],lo[-1]}):
        raise ValueError("上下路径在嘴角以外重合，请检查种子点。")
    left=max(p[up[0],0],p[lo[0],0])
    right=min(p[up[-1],0],p[lo[-1],0])
    if right <= left:
        raise ValueError("上下路径没有共同嘴宽。")
    sites=np.linspace(left,right,11)[1:-1]
    gap=np.interp(sites,p[up,0],p[up,2])-np.interp(sites,p[lo,0],p[lo,2])
    if gap.min() < -(right-left)*.002:
        raise ValueError("上下路径交叉或标签颠倒，请检查种子和局部 Z 朝向。")
    return paths
