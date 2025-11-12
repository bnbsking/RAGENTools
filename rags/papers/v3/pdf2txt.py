import glob
from ragentools.parsers.converters import Pdf2TxtConverter


obj = Pdf2TxtConverter(
    pdf_path_list=glob.glob("/app/rags/papers/data/*.pdf"),
    output_folder="/app/rags/papers/v3/input"
)
obj.convert()
