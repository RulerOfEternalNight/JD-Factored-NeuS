"""Plot existing detector annotations for the selected ten landmarks; no training."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from pyhocon import ConfigFactory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.landmarks import LandmarkAnnotations

root=Path(__file__).resolve().parents[1];out=root/'deliverables/staged10_multiview';out.mkdir(parents=True,exist_ok=True)
c=ConfigFactory.parse_file(str(root/'confs/staged_10_sigma20.conf'))['landmarks']
paths=sorted((root/'public_data/122_1_masked/image').glob('*.png'))
labels=LandmarkAnnotations(str(root/c['annotation_path'].replace('CASE_NAME','122_1_masked')),paths,1728,2592,c['ids'],c['names'])
view=6;xy=labels.xy[view];assert labels.valid[view].all()
photo=Image.open(paths[view]);photo.save(out/'sample_photo_camera006.png')
short=['Nose tip','Outer eye corner · MP33','Mouth corner · MP291','Outer eye corner · MP263','Inner eye corner · MP133','Inner eye corner · MP362','Mouth corner · MP61','Forehead center','Nose bridge','Chin']
colors=plt.get_cmap('tab10')(np.arange(10))
# Matching crop in both panels, in native coordinates; saved original remains full-size.
lo=xy.min(0);hi=xy.max(0);pad=np.array([210,180]);bounds=[max(0,lo[0]-pad[0]),min(1728,hi[0]+pad[0]),min(2592,hi[1]+pad[1]),max(0,lo[1]-pad[1])]
fig=plt.figure(figsize=(16,8),facecolor='white');gs=fig.add_gridspec(1,3,width_ratios=[1,1,1.05],wspace=.12)
for j,title in enumerate(['Sample input image','The ten selected landmarks']):
 ax=fig.add_subplot(gs[j]);ax.imshow(photo);ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.set_axis_off();ax.set_title(title,fontweight='bold',fontsize=16,pad=14)
 if j:
  # Place numbers beside points with leaders, avoiding eye-corner overlaps.
  offsets=[(35,5),(-115,-42),(65,35),(65,-48),(-125,20),(70,15),(-125,35),(40,-30),(45,-45),(50,25)]
  for k,(point,col,offset) in enumerate(zip(xy,colors,offsets)):
   ax.scatter(*point,s=65,c=[col],edgecolors='white',linewidths=1.5,zorder=5)
   ax.annotate(str(k+1),point,xytext=tuple(point+offset),textcoords='data',ha='center',va='center',fontweight='bold',fontsize=13,color='white',bbox={'boxstyle':'circle,pad=.25','fc':col,'ec':'white','lw':1},arrowprops={'arrowstyle':'-','color':col,'lw':1.8},zorder=6)
ax=fig.add_subplot(gs[2]);ax.set_axis_off();ax.text(0,1,'Landmark key',fontweight='bold',fontsize=17,va='top',transform=ax.transAxes)
for k,(name,lid,col) in enumerate(zip(short,c['ids'],colors)):
 y=.90-k*.079
 ax.text(.035,y,str(k+1),color='white',ha='center',va='center',fontweight='bold',fontsize=12,bbox={'boxstyle':'circle,pad=.27','fc':col,'ec':'none'},transform=ax.transAxes)
 ax.text(.115,y,f'{name}  [ID {lid}]' if 'MP' not in name else name,fontsize=12,va='center',transform=ax.transAxes)
ax.text(0,.035,'Forehead and nose bridge replace\nthe closely spaced lip centers.',fontsize=12,color='#566577',transform=ax.transAxes)
fig.suptitle('Ten landmark locations for the staged experiment',fontsize=22,fontweight='bold',y=.99)
fig.text(.5,.045,'Camera 006 · existing MediaPipe detections, not model predictions or verified anatomical ground truth.\nThe photo panels use the same display crop; the original full-resolution sample is also included.',ha='center',fontsize=11,color='#566577')
fig.subplots_adjust(top=.89,bottom=.13,left=.025,right=.98);fig.savefig(out/'selected_10_landmarks.png',dpi=180);fig.savefig(out/'selected_10_landmarks.pdf');plt.close(fig)
points=[]
for k,(lid,name,point) in enumerate(zip(c['ids'],c['names'],xy)):points.append({'display_number':k+1,'mediapipe_id':lid,'name':name,'xy_native_pixels':point.tolist()})
separation=[];overlap=[]
for points_view,valid in zip(labels.xy,labels.valid):
 if valid.all():
  distances=np.linalg.norm(points_view[:,None]-points_view[None],axis=-1);np.fill_diagonal(distances,np.inf);separation.append(float(distances.min()));overlap.extend(np.exp(-((points_view[:,None]-points_view[None])**2).sum(-1)/(2*20**2)).sum(-1).tolist())
(out/'selected_landmarks.json').write_text(json.dumps({'image':paths[view].name,'native_size':[1728,2592],'landmarks':points,'valid_views_per_landmark':labels.valid.sum(0).tolist(),'median_nearest_pair_separation_native_px':float(np.median(separation)),'fraction_annotation_centers_target_sum_gt_1_5':float(np.mean(np.array(overlap)>1.5))},indent=2))
print(out/'selected_10_landmarks.png')
