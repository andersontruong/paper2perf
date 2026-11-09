from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from paper2perf.documents import extract_pages
from paper2perf.store import Store


class DocumentTests(unittest.TestCase):
    def test_text_page_numbers(self):
        suffix, pages = extract_pages("paper.txt", b"First page\fSecond page")
        self.assertEqual(suffix, ".txt")
        self.assertEqual(pages, ["First page", "Second page"])

    def test_pdf_original_page_numbers(self):
        import pymupdf as fitz
        with fitz.open() as pdf:
            for text in ["First original page", "Second original page"]:
                page = pdf.new_page()
                page.insert_text((40, 60), text)
            _, pages = extract_pages("source.pdf", pdf.tobytes())
        self.assertIn("First original", pages[0])
        self.assertIn("Second original", pages[1])

    def test_image_only_pdf_is_not_silently_accepted(self):
        import pymupdf as fitz
        with fitz.open() as pdf:
            pdf.new_page()
            with self.assertRaisesRegex(ValueError, "OCR"):
                extract_pages("scan.pdf", pdf.tobytes())

    def test_invalid_formats_and_page_limits(self):
        for name, data in [("file.py", b"print(1)"), ("empty.txt", b""), ("binary.txt", b"\xff")]:
            with self.assertRaises(ValueError):
                extract_pages(name, data)
        with self.assertRaises(ValueError):
            extract_pages("large.txt", b"a\fb", max_pages=1)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.store = Store(Path(self.temp.name))
        self.a, self.b = self.store.new_session(), self.store.new_session()
        self.da = self.store.add_document(self.a, "A.txt", ".txt", b"A memory", ["A memory 8 GB"])
        self.db = self.store.add_document(self.b, "B.txt", ".txt", b"B memory", ["B memory 999 GB"])

    def tearDown(self):
        self.temp.cleanup()

    def test_search_is_scoped(self):
        results = self.store.search([self.da], "memory OR 999")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["document"], self.da)

    def test_duplicate_upload_is_idempotent(self):
        again = self.store.add_document(self.a, "renamed.txt", ".txt", b"A memory", ["A memory 8 GB"])
        self.assertEqual(again, self.da)
        self.assertEqual(len(self.store.documents(self.a)), 1)

    def test_query_punctuation_does_not_change_fts_syntax(self):
        self.store.search([self.da], "memory\"; DROP TABLE documents; --")
        self.assertEqual(len(self.store.documents(self.a)), 1)

    def test_run_access_requires_session(self):
        rid = self.store.start_run(self.a, {"message": "test"})
        self.assertIsNone(self.store.run(self.b, rid))
        self.assertIsNotNone(self.store.run(self.a, rid))

    def test_artifact_paths_cannot_escape(self):
        rid = self.store.start_run(self.a, {})
        with self.assertRaises(ValueError):
            self.store.artifact(rid, "../bad.txt", "data")

    def test_persistence(self):
        self.store.message(self.a, "user", "Question")
        loaded = Store(Path(self.temp.name))
        self.assertEqual(loaded.history(self.a)[0]["content"], "Question")


if __name__ == "__main__":
    unittest.main()


class PdfTypographyTests(unittest.TestCase):
    def test_raised_footnote_is_not_concatenated_into_a_number(self):
        import pymupdf
        from paper2perf.documents import extract_pages
        from paper2perf.evidence import numbers
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_text((60, 72), '4', fontsize=12)
            page.insert_text((66.8, 68), '2', fontsize=7)
            page.insert_text((100, 72), '42', fontsize=12)
            self.assertIn('42', page.get_text(sort=True))
            _, pages = extract_pages('table.pdf', doc.tobytes())
        self.assertIn('4²', pages[0])
        self.assertIn('42', pages[0])  # The real baseline number is unchanged.
        self.assertEqual(numbers(pages[0]), [4, 42])

    def test_ordinary_baseline_digits_are_not_rewritten(self):
        import pymupdf
        from paper2perf.documents import extract_pages
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_text((60, 72), '42 6243 108 128')
            _, pages = extract_pages('ordinary.pdf', doc.tobytes())
        self.assertEqual(pages[0].strip(), '42 6243 108 128')
