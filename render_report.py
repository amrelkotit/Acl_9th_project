"""Render the evidence-backed Markdown report to a review-ready PDF."""
from pathlib import Path
import re
import sys
from xml.sax.saxutils import escape
root = Path(__file__).resolve().parent
sys.path.insert(0,str(root/'.deps'))
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
                               PageBreak, KeepTogether)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

source = root/'outputs'/'batch2'/'milestone1_report.md'
target = root/'output'/'pdf'/'ACL_Milestone_1_Review_Report.pdf'
target.parent.mkdir(parents=True,exist_ok=True)
font_dir = Path('C:/Windows/Fonts')
for name,filename in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf'),('Arial-Italic','ariali.ttf'),('Arial-BoldItalic','arialbi.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(font_dir/filename)))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-BoldItalic')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyAcademic',fontName='Arial',fontSize=9.5,leading=13.5,spaceAfter=8))
styles.add(ParagraphStyle(name='CellAcademic',fontName='Arial',fontSize=7.1,leading=9.3,wordWrap='CJK'))
styles.add(ParagraphStyle(name='CaptionAcademic',fontName='Arial-Italic',fontSize=8,leading=11,textColor=colors.HexColor('#475569'),spaceAfter=10))
for key in ['Title','Heading1','Heading2','Heading3']:
    styles[key].fontName = 'Arial-Bold'
styles['Title'].fontSize=23
styles['Title'].leading=28
styles['Heading1'].fontSize=15
styles['Heading1'].leading=19
styles['Heading1'].spaceBefore=15
styles['Heading2'].fontSize=11.5
styles['Heading2'].leading=15
styles['Heading2'].spaceBefore=10

def text_markup(text):
    text = text.replace('—',' - ').replace('–','-').replace('−','-').replace('’',"'").replace('“','"').replace('”','"')
    text = escape(text)
    text = re.sub(r'\[([^]]+)\]\((https?://[^)]+)\)',r'<a href="\2" color="#0f766e">\1</a>',text)
    text = re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
    return text

WIDTH = A4[0]-90
story = [Paragraph('FORMULA 1<br/>PODIUM PREDICTION',styles['Title']),Spacer(1,18),
    Paragraph('CSEN 903 - Advanced Computer Lab',styles['Heading1']),
    Paragraph('Milestone 1 | Integrated analytical report',styles['Heading2']),Spacer(1,20),
    Paragraph('Complete local review package - plan rows 1-40 addressed',styles['BodyAcademic']),
    Paragraph('Prepared 14 October 2026. Submission deadline: 18 October 2026, 11:59 PM.',styles['BodyAcademic']),
    Paragraph('Includes audit, cleaning, historical questions, leakage-safe features, three-model comparison, controlled experiments, explanations, and inference demonstrations.',styles['BodyAcademic']),
    Paragraph('<b>Pending:</b> all-member review/access verification, Kaggle synchronization and runtime verification, historical source ambiguities, and official submission.',styles['BodyAcademic']),
    Spacer(1,22),Paragraph('Amr Hazem Mohamed Mohamed Hassan Elkotit<br/>Mark Bassem Mourad Samoel Emira<br/>Loay Mohamed Sabry Farag Ismail<br/>Karim Ahmed Mahmoud Abdelmageed',styles['BodyAcademic']),
    Spacer(1,20),Paragraph('Implementation assistance does not establish member contribution sign-off. All members must explain and defend the complete work.',styles['CaptionAcademic']),PageBreak()]
lines = source.read_text(encoding='utf-8').splitlines()
i=0
while i<len(lines):
    line = lines[i].strip()
    if not line:
        i+=1; continue
    if line.startswith('# '):
        i+=1; continue
    if line.startswith('## '):
        title = line[3:]
        if title in ['Three-model training, selection and comparison','Controlled experiments','Inference function and missing-data handling','Final acceptance addendum']:
            story.append(PageBreak())
        story.append(Paragraph(text_markup(title),styles['Heading1']))
        i+=1; continue
    if line.startswith('### '):
        story.append(Paragraph(text_markup(line[4:]),styles['Heading2']))
        i+=1; continue
    if line.startswith('|'):
        raw_rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            row=[c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[:\- ]+',c) for c in row): raw_rows.append(row)
            i+=1
        count=len(raw_rows[0])
        if count>=8:
            widths=[35,49]+[(WIDTH-84)/(count-2)]*(count-2)
        elif count==2:
            widths=[WIDTH*.72,WIDTH*.28]
        elif count==3:
            widths=[WIDTH*.19,WIDTH*.29,WIDTH*.52]
        else:
            widths=[WIDTH/count]*count
        data=[[Paragraph(text_markup(c),styles['CellAcademic']) for c in row] for row in raw_rows]
        table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dbeafe')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#cbd5e1')),
            ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        if len(raw_rows)<=11:
            story.append(KeepTogether([table,Spacer(1,12)]))
        else:
            story.extend([table,Spacer(1,12)])
        continue
    image_match=re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)',line)
    if image_match:
        caption,path=image_match.groups()
        image_path=source.parent/path
        with PILImage.open(image_path) as im: w,h=im.size
        # Confusion matrices remain legible at this size and can share a page.
        max_width=340 if 'confusion' in path else WIDTH
        max_height=300 if 'confusion' in path else 540
        scale=min(max_width/w,max_height/h)
        figure=Image(str(image_path),width=w*scale,height=h*scale)
        story.append(KeepTogether([figure,Spacer(1,5),Paragraph(text_markup(caption),styles['CaptionAcademic'])]))
        i+=1
        # Caption text repeats the Markdown alt text; render it once.
        while i<len(lines) and not lines[i].strip(): i+=1
        if i<len(lines) and lines[i].strip()==caption: i+=1
        continue
    paragraph=[line]
    i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#','|','![')):
        paragraph.append(lines[i].strip()); i+=1
    body=' '.join(paragraph)
    if body.startswith('- '): body=body[2:]
    story.append(Paragraph(text_markup(body),styles['BodyAcademic']))

def footer(canvas,doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#cbd5e1'))
    canvas.line(45,38,A4[0]-45,38)
    canvas.setFont('Arial',7)
    canvas.setFillColor(colors.HexColor('#64748b'))
    canvas.drawString(45,26,'CSEN 903 | Milestone 1 | Review draft')
    canvas.drawRightString(A4[0]-45,26,str(doc.page))
    canvas.restoreState()

doc=SimpleDocTemplate(str(target),pagesize=A4,leftMargin=45,rightMargin=45,topMargin=45,bottomMargin=50,
    title='ACL Milestone 1 - Formula 1 Podium Prediction',author='ACL Project Team')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(target)
