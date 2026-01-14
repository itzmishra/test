"""
Alternative PDF Documentation Generator
Uses markdown2 and reportlab as alternative method
"""

import os
import sys
from pathlib import Path

def generate_pdf_alternative():
    """
    Alternative method using markdown2 and reportlab
    """
    try:
        import markdown2
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from html.parser import HTMLParser
        import re
        
        print("Using markdown2 + reportlab method...")
        
    except ImportError:
        print("Installing required packages...")
        os.system(f"{sys.executable} -m pip install markdown2 reportlab")
        import markdown2
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from html.parser import HTMLParser
        import re
    
    current_dir = Path(__file__).parent
    markdown_file = current_dir / "documentation.md"
    output_pdf = current_dir / "Engine_Fault_Detection_System_Documentation.pdf"
    
    if not markdown_file.exists():
        print(f"❌ Error: {markdown_file} not found!")
        return False
    
    # Read markdown
    with open(markdown_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown2.markdown(md_content, extras=['fenced-code-blocks', 'tables'])
    
    # Create PDF
    doc = SimpleDocTemplate(str(output_pdf), pagesize=A4,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=72)
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='#2c3e50',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='#34495e',
        spaceAfter=12,
        spaceBefore=20
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor='#555',
        spaceAfter=10,
        spaceBefore=15
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Code'],
        fontSize=9,
        fontName='Courier',
        leftIndent=20,
        rightIndent=20,
        backColor='#f4f4f4',
        borderPadding=10
    )
    
    # Simple HTML to ReportLab converter
    def html_to_paragraphs(html_text):
        """Convert HTML to ReportLab paragraphs"""
        # Remove HTML tags and convert to plain text with basic formatting
        text = re.sub(r'<h1>(.*?)</h1>', r'<b><font size="18">\1</font></b>', html_text)
        text = re.sub(r'<h2>(.*?)</h2>', r'<b><font size="14">\1</font></b>', text)
        text = re.sub(r'<h3>(.*?)</h3>', r'<b><font size="12">\1</font></b>', text)
        text = re.sub(r'<code>(.*?)</code>', r'<font name="Courier" size="9">\1</font>', text)
        text = re.sub(r'<strong>(.*?)</strong>', r'<b>\1</b>', text)
        text = re.sub(r'<em>(.*?)</em>', r'<i>\1</i>', text)
        text = re.sub(r'<.*?>', '', text)  # Remove remaining HTML tags
        return text
    
    # Parse content (simplified)
    lines = html_content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue
        
        if line.startswith('<h1>'):
            text = re.sub(r'<.*?>', '', line)
            story.append(Paragraph(text, title_style))
        elif line.startswith('<h2>'):
            text = re.sub(r'<.*?>', '', line)
            story.append(Paragraph(text, heading1_style))
        elif line.startswith('<h3>'):
            text = re.sub(r'<.*?>', '', line)
            story.append(Paragraph(text, heading2_style))
        elif line.startswith('<pre>') or line.startswith('<code'):
            text = re.sub(r'<.*?>', '', line)
            story.append(Preformatted(text, code_style))
        else:
            if line:
                para_text = html_to_paragraphs(line)
                story.append(Paragraph(para_text, normal_style))
    
    # Build PDF
    try:
        doc.build(story)
        print(f"✅ PDF generated successfully: {output_pdf}")
        return True
    except Exception as e:
        print(f"❌ Error building PDF: {e}")
        return False

if __name__ == "__main__":
    generate_pdf_alternative()




