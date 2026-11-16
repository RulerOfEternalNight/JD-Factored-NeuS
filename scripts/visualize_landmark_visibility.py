"""Render the actual retained/excluded pseudo-labels for manual inspection."""
import sys, json, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from models.landmarks import LandmarkAnnotations


def main():
    root = Path(__file__).resolve().parents[1]
    review = json.loads((root/'annotations/122_1_masked/staged10_visibility_baseline10k.json').read_text())
    data = root/'public_data/122_1_masked/image'
    paths = [data/n for n in review['image_names']]
    labels = LandmarkAnnotations(root/'annotations/122_1_masked/mediapipe_478_v3.json', paths,
                                 *review['native_size'], review['ids'], review['names'])
    allowed = np.asarray(review['supervision_valid'], dtype=bool)
    out = root/'deliverables/landmark_corrections'
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), facecolor='white')
    for ax, i in zip(axes.flat, [6, 1, 16, 42]):
        ax.imshow(Image.open(paths[i]))
        xy = labels.xy[i, labels.valid[i]]
        for k in np.flatnonzero(labels.valid[i]):
            u, v = labels.xy[i, k]
            color = '#10c982' if allowed[i, k] else '#ff9c32'
            ax.scatter(u, v, c=color, s=48, edgecolors='black', linewidths=.7)
            ax.annotate(str(review['ids'][k]), (u, v), xytext=(7, 5), textcoords='offset points',
                        fontsize=9, color='white', bbox=dict(facecolor='black', alpha=.65, pad=1.5))
        ax.set_xlim(max(0, xy[:, 0].min()-160), min(review['native_size'][0], xy[:, 0].max()+160))
        ax.set_ylim(min(review['native_size'][1], xy[:, 1].max()+160), max(0, xy[:, 1].min()-160))
        ax.set_title(f"Camera {i:02d} · {allowed[i].sum()}/10 labels retained", fontsize=14)
        ax.axis('off')
    fig.suptitle('Visibility review of the 10 landmark annotations', fontsize=20, y=.98)
    fig.text(.5, .025, 'Green: retained for training    Orange: excluded from supervision\n'
             'Numbers are MediaPipe IDs. Mesh-inferred review flags require visual inspection; they are not ground truth.',
             ha='center', fontsize=11)
    fig.tight_layout(rect=(0, .065, 1, .95))
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'visibility_review.{ext}', dpi=180)
    with (out/'visibility_counts.csv').open('w') as f:
        writer = csv.writer(f); writer.writerow(['id', 'name', 'original_views', 'retained_views'])
        for a in review['anchors']:
            writer.writerow([a['id'], a['name'], a['original_views'], a['retained_views']])
    print(out/'visibility_review.png')


if __name__ == '__main__':
    main()
