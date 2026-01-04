"""
PDF Documentation Generator
Converts markdown documentation to PDF format
"""

import os
import sys
from pathlib import Path

try:
    from markdown import markdown
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
except ImportError:
    print("Required packages not installed. Installing...")
    os.system(f"{sys.executable} -m pip install markdown weasyprint")
    from markdown import markdown
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration

def markdown_to_pdf(markdown_file, output_pdf, css_style=None):
    """
    Convert markdown file to PDF
    
    Args:
        markdown_file: Path to input markdown file
        output_pdf: Path to output PDF file
        css_style: Optional custom CSS string
    """
    # Read markdown file
    with open(markdown_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown(md_content, extensions=['extra', 'codehilite', 'tables'])
    
    # Default CSS styling for professional PDF
    default_css = """
    @page {
        size: A4;
        margin: 2cm;
        @top-center {
            content: "Engine Fault Detection System - Documentation";
            font-size: 10pt;
            color: #666;
        }
        @bottom-center {
            content: "Page " counter(page) " of " counter(pages);
            font-size: 10pt;
            color: #666;
        }
    }
    
    body {
        font-family: 'Arial', 'Helvetica', sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #333;
        max-width: 100%;
    }
    
    h1 {
        font-size: 24pt;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
        page-break-after: avoid;
    }
    
    h2 {
        font-size: 18pt;
        color: #34495e;
        border-bottom: 2px solid #95a5a6;
        padding-bottom: 8px;
        margin-top: 25px;
        margin-bottom: 15px;
        page-break-after: avoid;
    }
    
    h3 {
        font-size: 14pt;
        color: #555;
        margin-top: 20px;
        margin-bottom: 10px;
        page-break-after: avoid;
    }
    
    h4 {
        font-size: 12pt;
        color: #666;
        margin-top: 15px;
        margin-bottom: 8px;
        page-break-after: avoid;
    }
    
    p {
        margin-bottom: 12px;
        text-align: justify;
    }
    
    code {
        background-color: #f4f4f4;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
        font-size: 10pt;
        color: #c7254e;
    }
    
    pre {
        background-color: #f8f8f8;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        overflow-x: auto;
        page-break-inside: avoid;
        margin: 15px 0;
    }
    
    pre code {
        background-color: transparent;
        padding: 0;
        color: #333;
    }
    
    blockquote {
        border-left: 4px solid #3498db;
        padding-left: 15px;
        margin-left: 0;
        color: #555;
        font-style: italic;
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
        page-break-inside: avoid;
    }
    
    th, td {
        border: 1px solid #ddd;
        padding: 10px;
        text-align: left;
    }
    
    th {
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
    
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    ul, ol {
        margin: 15px 0;
        padding-left: 30px;
    }
    
    li {
        margin-bottom: 8px;
    }
    
    a {
        color: #3498db;
        text-decoration: none;
    }
    
    a:hover {
        text-decoration: underline;
    }
    
    .toc {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
        page-break-inside: avoid;
    }
    
    .toc ul {
        list-style-type: none;
        padding-left: 0;
    }
    
    .toc li {
        margin: 5px 0;
    }
    
    .toc a {
        color: #2c3e50;
        text-decoration: none;
    }
    
    hr {
        border: none;
        border-top: 2px solid #ecf0f1;
        margin: 30px 0;
    }
    
    strong {
        color: #2c3e50;
        font-weight: bold;
    }
    
    em {
        font-style: italic;
        color: #555;
    }
    
    /* Code blocks */
    .codehilite {
        background-color: #f8f8f8;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        overflow-x: auto;
        page-break-inside: avoid;
    }
    
    /* Diagrams and flowcharts */
    pre.flowchart {
        background-color: #f0f8ff;
        border-left: 4px solid #3498db;
        font-family: 'Courier New', monospace;
        white-space: pre;
    }
    """
    
    # Use custom CSS if provided, otherwise use default
    css = CSS(string=css_style if css_style else default_css)
    
    # Create full HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Engine Fault Detection System - Documentation</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Generate PDF
    print(f"Converting {markdown_file} to PDF...")
    HTML(string=full_html).write_pdf(output_pdf, stylesheets=[css])
    print(f"✅ PDF generated successfully: {output_pdf}")

if __name__ == "__main__":
    # Get current directory
    current_dir = Path(__file__).parent
    
    # Input and output paths
    markdown_file = current_dir / "documentation.md"
    output_pdf = current_dir / "Engine_Fault_Detection_System_Documentation.pdf"
    
    # Check if markdown file exists
    if not markdown_file.exists():
        print(f"❌ Error: {markdown_file} not found!")
        sys.exit(1)
    
    try:
        # Convert to PDF
        markdown_to_pdf(str(markdown_file), str(output_pdf))
        print(f"\n📄 Documentation PDF created: {output_pdf}")
        print(f"📊 File size: {output_pdf.stat().st_size / 1024:.2f} KB")
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        print("\nTrying alternative method...")
        
        # Alternative: Try with reportlab if weasyprint fails
        try:
            print("Attempting alternative PDF generation method...")
            # You can add alternative PDF generation here if needed
            raise e
        except Exception as e2:
            print(f"❌ All PDF generation methods failed: {e2}")
            print("\n💡 Alternative: You can use online tools like:")
            print("   - https://www.markdowntopdf.com/")
            print("   - https://dillinger.io/ (export as PDF)")
            print("   - Or use pandoc: pandoc documentation.md -o output.pdf")
            sys.exit(1)


