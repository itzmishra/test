# Documentation Generation Guide

This guide explains how to generate the PDF documentation for the Engine Fault Detection System.

## Files Created

1. **documentation.md** - Complete markdown documentation with all technical details
2. **Engine_Fault_Detection_System_Documentation.html** - Formatted HTML version (ready for PDF conversion)
3. **generate_html_documentation.py** - Script to convert markdown to HTML
4. **generate_pdf_documentation.py** - Script to convert markdown to PDF (requires WeasyPrint)

## Method 1: HTML to PDF (Recommended - Easiest)

The HTML file has been generated and is ready to use:

1. **Open the HTML file**: `Engine_Fault_Detection_System_Documentation.html`
2. **Print to PDF**:
   - Click the "Print to PDF" button in the top-right corner, OR
   - Press `Ctrl+P` (Windows) or `Cmd+P` (Mac)
   - Select "Save as PDF" or "Microsoft Print to PDF" as the printer
   - Click "Save" and choose your location

This method works on all systems and produces high-quality PDFs.

## Method 2: Regenerate HTML

If you need to regenerate the HTML file:

```bash
python generate_html_documentation.py
```

## Method 3: Direct PDF Generation (Advanced)

If you have WeasyPrint installed (requires system libraries on Windows):

```bash
python generate_pdf_documentation.py
```

**Note**: WeasyPrint requires additional system libraries on Windows. For most users, Method 1 is recommended.

## Method 4: Using Pandoc (Alternative)

If you have Pandoc installed:

```bash
pandoc documentation.md -o Engine_Fault_Detection_System_Documentation.pdf
```

## Method 5: Online Tools

You can also use online markdown to PDF converters:

1. Upload `documentation.md` to https://www.markdowntopdf.com/
2. Or use https://dillinger.io/ and export as PDF

## Documentation Contents

The documentation includes:

- **Executive Summary**: Project overview and key features
- **System Overview**: Problem statement and solution approach
- **Architecture**: High-level and detailed architecture diagrams
- **System Components**: Detailed explanation of all components
- **Data Flow and Methodology**: Complete workflow and methodology
- **Feature Extraction Pipeline**: All 44 features explained
- **Machine Learning Model**: Model architecture and training details
- **User Interface**: Streamlit app structure and components
- **Installation and Setup**: Step-by-step installation guide
- **Usage Guide**: How to use the application
- **Technical Specifications**: All technical details
- **Performance Metrics**: Model performance and benchmarks
- **Future Enhancements**: Roadmap and improvements
- **References**: Libraries, papers, and standards

## File Structure

```
project-root/
├── documentation.md                                    # Source markdown
├── Engine_Fault_Detection_System_Documentation.html   # Generated HTML
├── generate_html_documentation.py                     # HTML generator
├── generate_pdf_documentation.py                      # PDF generator
└── DOCUMENTATION_README.md                            # This file
```

## Troubleshooting

### HTML file not opening properly
- Ensure you're using a modern browser (Chrome, Firefox, Edge)
- Check that the file encoding is UTF-8

### PDF generation issues
- Use Method 1 (Print to PDF from browser) - most reliable
- Ensure browser supports PDF printing

### Markdown syntax errors
- Check that `documentation.md` is valid markdown
- Verify all code blocks are properly formatted

## Support

For issues or questions about the documentation:
1. Check the troubleshooting section above
2. Verify all files are present
3. Try regenerating the HTML file

---

**Last Updated**: December 2024







