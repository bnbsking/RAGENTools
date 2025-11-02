import pandas as pd

from ragentools.parsers.pdf_parser import PDFParser


class TestPDFParser:
    def test_parse(self):
        parser = PDFParser(
            input_path_list=["/app/tests/parsers/data/qap.pdf"],
            output_folder="/app/tests/parsers/output"
        )
        parser.parse()

        df_out = pd.read_csv("/app/tests/parsers/output/qap.pdf.csv")
        df_exp = pd.read_csv("/app/tests/parsers/ground_truth/qap.pdf.csv")
        pd.testing.assert_frame_equal(df_out, df_exp)


if __name__ == "__main__":
    test_parser = TestPDFParser()
    test_parser.test_parse()
