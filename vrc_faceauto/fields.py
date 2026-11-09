"""Extend prescribed vertex displacements along connected surface edges."""
import heapq
from collections import defaultdict

import numpy as np


def extend_field(obj,p,seeds,width):
    # D_i(0) = D_nearest*(1-d_i/r)^2; d_i is the shortest mesh-edge distance.
    # Free vertices: D_next = .35*D_i + .65*sum(w_ij*D_j)/sum(w_ij).
    r=.55*width
    rim=p[list(seeds)]
    near=((p>=rim.min(axis=0)-r)&(p<=rim.max(axis=0)+r)).all(axis=1)
    pairs=np.empty(len(obj.data.edges)*2,dtype="i")
    obj.data.edges.foreach_get("vertices",pairs)
    pairs=pairs.reshape(-1,2)
    pairs=pairs[near[pairs].all(axis=1)]
    adj=defaultdict(list)
    for (a,b),length in zip(pairs,np.linalg.norm(p[pairs[:,0]]-p[pairs[:,1]],axis=1)):
        adj[a].append((b,length))
        adj[b].append((a,length))
    best={v:(0.,v) for v in seeds}
    todo=[(0.,v,v) for v in seeds]
    heapq.heapify(todo)
    while todo:
        d,v,source=heapq.heappop(todo)
        if (d,source)!=best[v]:
            continue
        for n,dist in adj[v]:
            nd=d+dist
            if nd<r and nd<best.get(n,(np.inf,None))[0]:
                best[n]=(nd,source)
                heapq.heappush(todo,(nd,n,source))
    ids=np.array(list(best))
    local={v:i for i,v in enumerate(ids)}
    field=np.zeros((len(ids),len(next(iter(seeds.values())))))
    rows=[];cols=[];weights=[]
    denom=np.zeros(len(ids))
    for i,v in enumerate(ids):
        d,source=best[v]
        field[i]=seeds[source]*(1-d/r)**2
        for n,length in adj[v]:
            w=1/max(length,1e-10)
            denom[i]+=w
            if n in local:
                rows.append(i);cols.append(local[n]);weights.append(w)
    rows=np.array(rows)
    cols=np.array(cols)
    weights=np.array(weights)
    fixed=[local[v] for v in seeds]
    for _ in range(48):
        avg=np.column_stack([np.bincount(rows,weights*field[cols,j],minlength=len(ids)) for j in range(field.shape[1])])
        new=avg/np.maximum(denom[:,None],1e-10)
        new[fixed]=field[fixed]
        field=.35*field+.65*new
    return ids,field
