from pdf2image import convert_from_path
from PIL import Image
import os

class PDFCombiner:
    def __init__(self, dpi, size_mm):
        self.dpi = dpi
        self.size_pixels = (int(size_mm[0] * dpi / 25.4), int(size_mm[1] * dpi / 25.4))

    def convert_pdf_page_to_image(self, pdf_path):
        # Convert the first page of the PDF to an image
        images = convert_from_path(pdf_path, dpi=self.dpi, size=self.size_pixels)
        return images[0]

    def combine_pdfs(self, folder_path):
        for root, dirs, files in os.walk(folder_path):
            if "confusion_matrix.pdf" in files and "ROC.pdf" in files:
                confusion_matrix_path = os.path.join(root, "confusion_matrix.pdf")
                roc_path = os.path.join(root, "ROC.pdf")

                confusion_matrix_img = self.convert_pdf_page_to_image(confusion_matrix_path)
                roc_img = self.convert_pdf_page_to_image(roc_path)

                # Combine the images side by side
                combined_img_width = confusion_matrix_img.width + roc_img.width
                combined_img_height = max(confusion_matrix_img.height, roc_img.height)
                combined_img = Image.new('RGB', (combined_img_width, combined_img_height))
                combined_img.paste(confusion_matrix_img, (0, 0))
                combined_img.paste(roc_img, (confusion_matrix_img.width, 0))

                # Save the combined image
                combined_img_path = os.path.join(root, 'combined_image.jpg')
                combined_img.save(combined_img_path, 'JPEG', quality=100)

                print(f"Combined image saved at {combined_img_path}")

# Example usage
# combiner = PDFCombiner(900, (800, 600))
# combiner.combine_pdfs('/path/to/parent/folder')


# Example usage
combiner = PDFCombiner(900, (800, 600))
combiner.combine_pdfs(r'F:\360MoveData\Users\Administrator\Desktop\data\命名')

