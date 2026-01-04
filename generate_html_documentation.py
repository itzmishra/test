"""
HTML Documentation Generator
Converts markdown to beautifully formatted HTML that can be printed to PDF
"""

import os
import sys
from pathlib import Path

try:
    from markdown import markdown
except ImportError:
    print("Installing markdown package...")
    os.system(f"{sys.executable} -m pip install markdown")
    from markdown import markdown

def markdown_to_html(markdown_file, output_html):
    """
    Convert markdown file to HTML with professional styling
    """
    # Read markdown file
    with open(markdown_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown(md_content, extensions=['extra', 'codehilite', 'tables', 'fenced_code'])
    
    # Professional HTML template with CSS
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Engine Fault Detection System - Complete Documentation</title>
    <style>
        @media print {{
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                margin: 0;
                padding: 0;
            }}
            .no-print {{
                display: none;
            }}
            h1, h2, h3 {{
                page-break-after: avoid;
            }}
            pre, code {{
                page-break-inside: avoid;
            }}
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        
        .document-container {{
            background: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        
        h1 {{
            font-size: 32px;
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
            margin-top: 0;
            margin-bottom: 30px;
        }}
        
        h2 {{
            font-size: 24px;
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 10px;
            margin-top: 40px;
            margin-bottom: 20px;
        }}
        
        h3 {{
            font-size: 18px;
            color: #555;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        
        h4 {{
            font-size: 16px;
            color: #666;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        
        p {{
            margin-bottom: 15px;
            text-align: justify;
        }}
        
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #c7254e;
        }}
        
        pre {{
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-left: 4px solid #3498db;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
            margin: 20px 0;
        }}
        
        pre code {{
            background-color: transparent;
            padding: 0;
            color: #333;
            border: none;
        }}
        
        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin-left: 0;
            color: #555;
            font-style: italic;
            background-color: #f9f9f9;
            padding: 15px 20px;
            margin: 20px 0;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        
        tr:hover {{
            background-color: #f1f1f1;
        }}
        
        ul, ol {{
            margin: 15px 0;
            padding-left: 40px;
        }}
        
        li {{
            margin-bottom: 8px;
        }}
        
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 40px 0;
        }}
        
        strong {{
            color: #2c3e50;
            font-weight: bold;
        }}
        
        em {{
            font-style: italic;
            color: #555;
        }}
        
        .codehilite {{
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-left: 4px solid #3498db;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
            margin: 20px 0;
        }}
        
        .header-info {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        
        .header-info h1 {{
            color: white;
            border: none;
            margin: 0;
            padding: 0;
        }}
        
        .header-info p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        
        .print-button {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: #3498db;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            z-index: 1000;
        }}
        
        .print-button:hover {{
            background: #2980b9;
        }}
        
        @media print {{
            .print-button {{
                display: none;
            }}
            .document-container {{
                box-shadow: none;
                padding: 0;
            }}
            body {{
                background: white;
                padding: 0;
            }}
        }}
    </style>
    <script>
        function printPDF() {{
            window.print();
        }}
    </script>
</head>
<body>
    <button class="print-button no-print" onclick="printPDF()">Print to PDF</button>
    <div class="document-container">
        <div class="header-info">
            <h1>Sound-Based Engine Fault Detection System</h1>
            <p>Complete Technical Documentation | Version 1.0 | December 2024</p>
        </div>
        {content}
    </div>
    <div style="text-align: center; padding: 20px; color: #666; font-size: 12px;" class="no-print">
        <p>To save as PDF: Click the "Print to PDF" button above, or use Ctrl+P and select "Save as PDF"</p>
    </div>
</body>
</html>"""
    
    # Insert content into template
    full_html = html_template.format(content=html_content)
    
    # Write HTML file
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"HTML documentation generated: {output_html}")
    print(f"Open the file in a browser and use 'Print to PDF' to create PDF")
    return output_html

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    markdown_file = current_dir / "documentation.md"
    output_html = current_dir / "Engine_Fault_Detection_System_Documentation.html"
    
    if not markdown_file.exists():
        print(f"Error: {markdown_file} not found!")
        sys.exit(1)
    
    try:
        html_file = markdown_to_html(str(markdown_file), str(output_html))
        print(f"\nDocumentation HTML created: {html_file}")
        print(f"Instructions:")
        print(f"   1. Open {html_file} in your web browser")
        print(f"   2. Click the 'Print to PDF' button, or")
        print(f"   3. Press Ctrl+P and select 'Save as PDF'")
        print(f"   4. Choose 'Save as PDF' as the printer")
        print(f"   5. Save the file")
    except Exception as e:
        print(f"Error generating HTML: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

