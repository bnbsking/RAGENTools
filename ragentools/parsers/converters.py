import os
from typing import List

import pymupdf


class Pdf2TxtConverter:
    def __init__(self, pdf_path_list: List[str], output_folder: str):
        self.pdf_path_list = pdf_path_list
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def convert(self):
        for pdf_path in self.pdf_path_list:
            with pymupdf.open(pdf_path) as document:
                save_path = os.path.join(self.output_folder, os.path.basename(pdf_path) + ".txt")
                with open(save_path, 'w', encoding='utf-8') as txt_file:
                    for page in document:
                        text = page.get_text()
                        txt_file.write(text + "\n")
