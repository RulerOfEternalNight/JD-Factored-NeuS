"""Build a photo-free PDF record from the branch's method/results documentation."""
import sys,re,html
from pathlib import Path
sys.path.insert(0,'/tmp/factored_neus_slide_deps')
import matplotlib
from PIL import Image as PILImage
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,Preformatted,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1];font=Path(matplotlib.get_data_path())/'fonts/ttf'
pdfmetrics.registerFont(TTFont('DV',str(font/'DejaVuSans.ttf')));pdfmetrics.registerFont(TTFont('DVB',str(font/'DejaVuSans-Bold.ttf')))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='B',fontName='DV',fontSize=9.5,leading=13.5,spaceAfter=8))
styles.add(ParagraphStyle(name='H',fontName='DVB',fontSize=17,leading=23,spaceAfter=13,textColor=colors.HexColor('#153b52'),keepWithNext=True))
styles.add(ParagraphStyle(name='S',fontName='DVB',fontSize=12,leading=16,spaceAfter=8,spaceBefore=10,textColor=colors.HexColor('#178777'),keepWithNext=True))
styles.add(ParagraphStyle(name='T',fontName='DV',fontSize=8,leading=11))
def clean(s):return re.sub(r'\[([^\]]+)\]\([^)]+\)',r'\1',s).replace('**','').replace('`','')
def p(s,style='B'):return Paragraph(html.escape(clean(s)),styles[style])
story=[Spacer(1,50),p('Corrected staged softmax landmark learning','H'),p('Working implementation record: 10 anatomical landmarks','S'),p('Frozen branch: softmax_10landmark_working'),p('Original experiment: staged_10_corrected'),Spacer(1,20),p('8/10 selected; 19.3065-pixel mean reprojection error on selected landmarks.'),p('Forehead and chin unresolved. Training-view MediaPipe agreement, not independently validated 3D anatomy or geometry improvement.'),Spacer(1,20),p('This record includes method, final metrics, all intermediate landmark responses, fixed-camera surface progression and reproducibility limitations. Source photographs and photographic overlays are not included.'),p('For setup, exact artifact restoration and repeat training, follow README.md and docs/REPRODUCIBILITY.md in the branch.')]
for name in ['METHOD.md','RESULTS.md','INTERMEDIATE_RESULTS.md','KNOWN_LIMITATIONS.md']:
    story.append(PageBreak());lines=(ROOT/'docs'/name).read_text().splitlines();i=0;paragraph=[]
    def flush():
        if paragraph:story.append(p(' '.join(paragraph)));paragraph.clear()
    while i<len(lines):
        line=lines[i]
        if not line.strip():flush();i+=1;continue
        if line.startswith('# '):flush();story.append(p(line[2:],'H'));i+=1;continue
        if line.startswith('## '):flush();story.append(p(line[3:],'S'));i+=1;continue
        if line.startswith('!['):
            flush();path=ROOT/'docs'/re.search(r'\(([^)]+)\)',line).group(1);w,h=PILImage.open(path).size;scale=min(500/w,490/h);story.append(Image(str(path),width=w*scale,height=h*scale));story.append(Spacer(1,12));i+=1;continue
        if line.startswith('|'):
            flush();rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[s.strip() for s in lines[i].strip('|').split('|')]
                if not all(set(s)<=set('-: ') for s in cells):rows.append([p(s,'T') for s in cells])
                i+=1
            n=len(rows[0]);widths=[150,350] if n==2 else [500/n]*n;t=Table(rows,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.3,colors.HexColor('#cbdbe5')),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf4f7')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.extend([t,Spacer(1,10)]);continue
        if line.startswith('```'):
            flush();code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
            story.append(Preformatted('\n'.join(code),styles['Code']));i+=1;continue
        paragraph.append(line);i+=1
    flush()
def footer(canvas,doc):
    canvas.setFont('DV',7);canvas.setFillColor(colors.HexColor('#5a7385'));canvas.drawString(45,25,'Softmax 10-landmark working snapshot | private source photos excluded');canvas.drawRightString(550,25,str(doc.page))
out=ROOT/'docs/WORKING_IMPLEMENTATION.pdf'
SimpleDocTemplate(str(out),pagesize=(595,842),leftMargin=47,rightMargin=47,topMargin=45,bottomMargin=45,title='Corrected staged 10-landmark Factored-NeuS working implementation').build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
