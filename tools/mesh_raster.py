"""Fixed-camera, full-mesh z-buffer used only for scientific report figures."""
import numpy as np
from numba import njit

@njit
def raster(vertices,faces,P,width,height,scale_x,scale_y):
    points=np.empty((len(vertices),3),np.float32)
    for i in range(len(vertices)):
        x,y,z=vertices[i]
        w=P[2,0]*x+P[2,1]*y+P[2,2]*z+P[2,3]
        points[i,0]=(P[0,0]*x+P[0,1]*y+P[0,2]*z+P[0,3])/w*scale_x
        points[i,1]=(P[1,0]*x+P[1,1]*y+P[1,2]*z+P[1,3])/w*scale_y
        points[i,2]=w
    depth=np.full((height,width),np.inf,np.float32)
    gray=np.full((height,width),np.nan,np.float32)
    # View direction from projection's third row; consistent two-sided headlight.
    light=P[2,:3];light=light/np.sqrt((light*light).sum())
    for t in range(len(faces)):
        ia,ib,ic=faces[t];a=points[ia];b=points[ib];c=points[ic]
        if a[2]<=0 or b[2]<=0 or c[2]<=0:continue
        xlo=max(0,int(np.floor(min(a[0],b[0],c[0]))));xhi=min(width-1,int(np.ceil(max(a[0],b[0],c[0]))))
        ylo=max(0,int(np.floor(min(a[1],b[1],c[1]))));yhi=min(height-1,int(np.ceil(max(a[1],b[1],c[1]))))
        if xhi<xlo or yhi<ylo:continue
        denom=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(denom)<1e-12:continue
        e1=vertices[ib]-vertices[ia];e2=vertices[ic]-vertices[ia]
        normal=np.cross(e1,e2);norm=np.sqrt((normal*normal).sum())
        if norm<1e-12:continue
        shade=.25+.72*abs((normal*light).sum()/norm)
        for yy in range(ylo,yhi+1):
            for xx in range(xlo,xhi+1):
                px=xx+.5;py=yy+.5
                wa=((b[1]-c[1])*(px-c[0])+(c[0]-b[0])*(py-c[1]))/denom
                wb=((c[1]-a[1])*(px-c[0])+(a[0]-c[0])*(py-c[1]))/denom
                wc=1-wa-wb
                if min(wa,wb,wc)<-1e-6:continue
                inv=wa/a[2]+wb/b[2]+wc/c[2]
                if inv<=0:continue
                zz=1/inv
                if zz<depth[yy,xx]:depth[yy,xx]=zz;gray[yy,xx]=shade
    return gray
