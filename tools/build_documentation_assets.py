"""Generate photo-free documentation figures from the archived sparse run."""
import sys,json,csv
from pathlib import Path
sys.path.insert(0,'/tmp/factored_neus_slide_deps')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
import trimesh
from mesh_raster import raster

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/assets';OUT.mkdir(exist_ok=True)
result=json.load(open(ROOT/'results/landmarks.json'));vals=json.load(open(ROOT/'results/validation_statistics.json'))
folder=ROOT/'exp/122_1_masked/staged_10_corrected/geometry'
names=['Nose tip','Outer eye 33','Mouth corner 291','Outer eye 263','Inner eye 133','Inner eye 362','Mouth corner 61','Forehead','Nose bridge','Chin']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

fig,ax=plt.subplots(figsize=(10,5));rows=result['reprojection']['per_landmark']
for i,r in enumerate(rows):
    if r['mean_error'] is None:ax.text(.5,i,'Unresolved',va='center',color='#b04452')
    else:ax.barh(i,r['mean_error'],color='#178777');ax.text(r['mean_error']+.4,i,f"{r['mean_error']:.2f} px",va='center')
ax.set_yticks(range(10),names);ax.invert_yaxis();ax.set_xlim(0,36);ax.set_xlabel('Mean native-image reprojection error (px)');ax.set_title('Corrected staged softmax: 8/10 recovered, 19.31 px conditional mean');ax.grid(axis='x',alpha=.15)
save(fig,'final_landmark_errors')

curves={}
for row in csv.DictReader(open(ROOT/'results/curve_windows.csv')):curves.setdefault(row['tag'],[]).append(row)
fig,axes=plt.subplots(2,2,figsize=(11,7))
for ax,tag,title in zip(axes.flat,['Loss/landmark_heatmap','Loss/nonlandmark_surface_ce','LandmarkGradients/head','LandmarkGradients/sdf'],['Heatmap L1','Non-landmark surface CE','Head gradient norm','SDF gradient norm from landmarks']):
    rows=curves[tag];x=[(int(r['start'])+int(r['end']))/2 for r in rows];y=[float(r['mean']) for r in rows]
    ax.plot(x,y,color='#178777',lw=2)
    if tag=='Loss/landmark_heatmap':ax.plot(x,[float(r['nonzero_mean']) for r in rows],ls='--',color='#647d92',label='Nonzero supervision batches');ax.legend(fontsize=8)
    else:ax.set_yscale('log')
    ax.axvspan(10000,15000,color='#eef3f6');ax.axvline(15000,color='#8c9dab',ls='--');ax.set_title(title);ax.set_xlabel('Iteration');ax.grid(alpha=.15)
fig.suptitle('Original run: 500-step means; shaded interval is frozen geometry',fontsize=15);fig.tight_layout();save(fig,'training_curves')

fig,ax=plt.subplots(figsize=(10,5));matrix=np.array([[v['rows'][k]['maximum'] for v in vals] for k in range(10)])
im=ax.imshow(np.maximum(matrix,1e-7),norm=LogNorm(1e-7,1),cmap='viridis',aspect='auto');ax.set_yticks(range(10),names);ax.set_xticks(range(8),[f"{v['iteration']/1000:g}k" for v in vals]);ax.set_title('All intermediate peak responses: fixed validation camera 1');fig.colorbar(im,ax=ax,label='Response maximum (log scale)');save(fig,'intermediate_responses')

# Only model heatmaps are included; no photographs or photo overlays are copied.
for start in [0,5]:
    fig,axes=plt.subplots(5,8,figsize=(15,14))
    for row,k in enumerate(range(start,start+5)):
        for col,v in enumerate(vals):
            ax=axes[row,col];ax.imshow(Image.open(folder/f'landmark_validation/{v["iteration"]:08d}_001/{k}_prediction.png'));ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f"{v['iteration']/1000:g}k")
            if col==0:ax.set_ylabel(names[k])
    fig.suptitle('Saved landmark responses: fixed 0–1 scale, camera 1',fontsize=16);fig.tight_layout(rect=(0,0,1,.96));save(fig,f'all_heatmaps_{start//5+1}')

fig,axes=plt.subplots(5,4,figsize=(11,15))
for k in range(10):
    row,col=divmod(k,2)
    for offset,kind in enumerate(['prediction','target']):
        ax=axes[row,col*2+offset];ax.imshow(Image.open(folder/f'landmark_validation/00030000_001/{k}_{kind}.png'));ax.set_xticks([]);ax.set_yticks([]);ax.set_title(names[k]+'\n'+kind,fontsize=9)
fig.suptitle('Final predicted and target heatmaps; excluded view targets are zero',fontsize=15);fig.tight_layout(rect=(0,0,1,.96));save(fig,'final_heatmap_comparison')

camera=json.load(open(ROOT/'results/render_camera6.json'));P=np.asarray(camera['P_world'],dtype=np.float32);width,height=480,720
crop=tuple(int(v*(width/1728 if i%2==0 else height/2592)) for i,v in enumerate(camera['crop_native']))
fig,axes=plt.subplots(1,4,figsize=(12,5))
for ax,step in zip(axes,[15000,20000,25000,30000]):
    mesh=trimesh.load(folder/f'meshes/{step:08d}.ply',process=False);g=raster(np.asarray(mesh.vertices,dtype=np.float32),np.asarray(mesh.faces,dtype=np.int32),P,width,height,width/1728,height/2592)
    image=np.full((height,width,3),[243,247,250],np.uint8);visible=np.isfinite(g)
    for ch,base in enumerate([180,200,220]):image[:,:,ch][visible]=np.clip(g[visible]*base+23,0,255).astype(np.uint8)
    ax.imshow(Image.fromarray(image).crop(crop));ax.axis('off');ax.set_title(f'{step/1000:g}k')
fig.suptitle('Intermediate surfaces: same camera, crop and shading; full saved meshes',fontsize=15);fig.tight_layout(rect=(0,0,1,.95));save(fig,'surface_progress')
print(OUT)
