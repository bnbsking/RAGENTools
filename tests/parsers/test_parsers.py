import glob
import os

import pandas as pd
from ragentools.parsers import Document
from ragentools.parsers.readers import ListReader, PDFReader
from ragentools.parsers.chunkers import OverlapChunker
from ragentools.parsers.savers import ListSaver, PDFSaver
from ragentools.parsers.parsers import BaseParser


class TestBaseParser:
    def test_run_list_eager(self):
        # inputs
        text_list =["Hello", "World !!"]
        meta_list = [{"id": 1}, {"id": 2}]
        save_folder = "/app/tests/parsers/output"

        # expected outputs
        target_list = [
            Document(page_content='Hel', metadata={'id': 1}),
            Document(page_content='llo', metadata={'id': 1}),
            Document(page_content='o', metadata={'id': 1}),
            Document(page_content='Wor', metadata={'id': 2}),
            Document(page_content='rld', metadata={'id': 2}),
            Document(page_content='d !', metadata={'id': 2}),
            Document(page_content='!!', metadata={'id': 2})
        ]
        gt_csv_path = "/app/tests/parsers/ground_truth/parsed_lists.csv"

        # run
        reader = ListReader(text_list=text_list, meta_list=meta_list)
        chunker = OverlapChunker(chunk_size=3, overlap_size=1)
        saver = ListSaver(save_folder=save_folder)
        parser = BaseParser(reader=reader, chunker=chunker, saver=saver)
        result = parser.run()
        print(result)

        # verify
        assert len(result) == len(target_list)
        for res_doc, tgt_doc in zip(result, target_list):
            assert res_doc.page_content == tgt_doc.page_content
            assert res_doc.metadata == tgt_doc.metadata
        df1 = pd.read_csv(f"{save_folder}/parsed_lists.csv")
        df2 = pd.read_csv(gt_csv_path)
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_run_pdf_lazy(self):
        # inputs
        pdf_paths = glob.glob("/app/tests/parsers/data/*.pdf")
        save_folder = "/app/tests/parsers/output"
        
        # expected outputs
        gt_csv_folder = "/app/tests/parsers/ground_truth/"
        target = Document(
            page_content='Medicine \nDefinition:\u200b\n Medicine is the science and practice of diagnosing, treating, and preventing disease and \ninjury, as well as maintaining and promoting overall health. It encompasses a wide ran',
            metadata={'source_path': '/app/tests/parsers/data/medicine.pdf', 'page': 1}
        )

        # run
        reader = PDFReader(pdf_paths=pdf_paths)
        chunker = OverlapChunker(chunk_size=200, overlap_size=10)
        saver = PDFSaver(save_folder=save_folder)
        parser = BaseParser(reader=reader, chunker=chunker, saver=saver)
        result = parser.run(lazy=True)
        first_chunk = next(result)
        print(first_chunk)

        # verify
        assert first_chunk.page_content == target.page_content
        assert first_chunk.metadata == target.metadata
        for pdf_path in pdf_paths:
            df1 = pd.read_csv(f"{save_folder}/{os.path.basename(pdf_path)}.csv")
            df2 = pd.read_csv(f"{gt_csv_folder}/{os.path.basename(pdf_path)}.csv")
            pd.testing.assert_frame_equal(df1, df2)


if __name__ == "__main__":
    test_parser = TestBaseParser()
    test_parser.test_run_list_eager()
    test_parser.test_run_pdf_lazy()