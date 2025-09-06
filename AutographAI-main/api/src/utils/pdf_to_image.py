import fitz  # PyMuPDF
import os

def pdf_to_images(pdf_path, output_dir):
    """
    Convert each page of the PDF to a high-resolution JPEG image.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Path to the directory to save images.

    Returns:
        None
    """
    # Check if output directory exists, create if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Iterate over each page in the PDF
    for page_num in range(len(pdf_document)):
        # Get the page
        page = pdf_document[page_num]

        # Define high resolution (e.g., 300 DPI)
        zoom_x = 3.0  # Horizontal zoom
        zoom_y = 3.0  # Vertical zoom
        matrix = fitz.Matrix(zoom_x, zoom_y)

        # Render page to a pixmap
        pix = page.get_pixmap(matrix=matrix)

        # Define output image path
        output_path = os.path.join(output_dir, f"page_{page_num + 1}.jpeg")

        # Save the image as JPEG
        pix.save(output_path)

    # Close the PDF document
    pdf_document.close()

    print(f"PDF converted to images and saved in: {output_dir}")

# Example usage
pdf_path = "patent-1.pdf"  # Path to your PDF file
output_dir = "./result"  # Directory to save images
pdf_to_images(pdf_path, output_dir)