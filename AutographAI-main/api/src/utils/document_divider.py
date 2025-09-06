#!/usr/bin/env python3
import csv
import os
from fpdf import FPDF

# Path to your CSV and to the font file
CSV_FILE = "products.csv"                  # <--- Change if needed
OUTPUT_PDF = "output.pdf"              # <--- PDF output name
DEJAVU_TTF = "DejaVuSans.ttf"          # <--- Make sure this file is in the same folder

# How many rows to include in the PDF
MAX_ITEMS = 50

class PDFUnicode(FPDF):
    """A simple subclass of FPDF that adds a Unicode-capable TTF font."""
    def __init__(self):
        super().__init__()
        # Enable auto page break
        self.set_auto_page_break(auto=True, margin=15)

        # Add the font. 'uni=True' is crucial for handling wide Unicode ranges
        if not os.path.isfile(DEJAVU_TTF):
            raise FileNotFoundError(
                f"Cannot find {DEJAVU_TTF} in the current directory. "
                "Please place a TTF font file here or update the path."
            )
        self.add_font("DejaVu", "", DEJAVU_TTF, uni=True)

        # Set the default font/size
        self.set_font("DejaVu", size=12)

def main():
    pdf = PDFUnicode()

    # Read the CSV, parse up to MAX_ITEMS
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            if i == MAX_ITEMS:
                break  # Stop after N rows

            pdf.add_page()

            # Build a multi-line string from the row's columns
            # Adjust keys below to match your column headers
            text_paragraph = (
                "=== Product Node ===\n\n"
                f"ID: {row.get('id','')}\n"
                f"Product ID: {row.get('product_id','')}\n"
                f"Name: {row.get('name','')}\n"
                f"Brand: {row.get('brand_name','')}\n\n"
                "--- Properties ---\n"
                f"brand_id: {row.get('brand_id','')}\n"
                f"sell_price: {row.get('sell_price','')}\n"
                f"list_price: {row.get('list_price','')}\n"
                f"discount: {row.get('discount','')}\n"
                f"discount_val: {row.get('discount_val','')}\n"
                f"color_css: {row.get('color_css','')}\n"
                f"url: {row.get('url','')}\n"
                f"ingredients: {row.get('ingredients','')}\n"
                f"batch_id: {row.get('batch_id','')}\n"
                f"created_at: {row.get('created_at','')}\n"
                f"updated_at: {row.get('updated_at','')}\n"
                f"category: {row.get('category','')}\n"
            )

            pdf.multi_cell(0, 8, text_paragraph)

    # Save the PDF
    pdf.output(OUTPUT_PDF)
    print(f"Done! Created PDF with up to {MAX_ITEMS} items => {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
