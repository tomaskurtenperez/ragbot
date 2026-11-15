"""Unit tests for the plain-text (RFC) document splitter."""

import tempfile
import unittest
from pathlib import Path

from app.services.document_splitter import find_documents, split_text_file


PAGINATED_RFC = """\
Internet Engineering Task Force (IETF)                    D. Hardt, Ed.
Request for Comments: 6749                                     Microsoft

                 The OAuth 2.0 Authorization Framework

Abstract

   The OAuth 2.0 authorization framework enables a third-party
   application to obtain limited access to an HTTP service.

Table of Contents

   1. Introduction ....................................................4
   2. Client Registration .............................................9

1.  Introduction

   The client MUST NOT use the authorization code more than once.

Hardt                        Standards Track                    [Page 4]
\f
RFC 6749                        OAuth 2.0                   October 2012


1.1.  Roles

   OAuth defines four roles.

Hardt                        Standards Track                    [Page 5]
"""


class SplitTextFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.path = self.root / "rfc6749-oauth2.txt"
        self.path.write_text(PAGINATED_RFC, encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_chunks_are_tagged_with_their_section(self) -> None:
        chunks = split_text_file(self.path, self.root)
        sections = [chunk["metadata"]["section"] for chunk in chunks]

        self.assertEqual(sections, ["Front matter", "1. Introduction", "1.1. Roles"])
        self.assertTrue(
            all(chunk["metadata"]["filename"] == self.path.name for chunk in chunks)
        )

    def test_page_headers_and_footers_are_removed(self) -> None:
        text = "\n".join(c["content"] for c in split_text_file(self.path, self.root))

        self.assertNotIn("[Page", text)
        self.assertNotIn("October 2012", text)
        self.assertNotIn("\f", text)

    def test_normative_keywords_are_preserved(self) -> None:
        chunks = split_text_file(self.path, self.root)
        introduction = next(
            c for c in chunks if c["metadata"]["section"] == "1. Introduction"
        )

        self.assertIn("MUST NOT use the authorization code", introduction["content"])

    def test_table_of_contents_is_dropped_but_abstract_kept(self) -> None:
        front_matter = split_text_file(self.path, self.root)[0]["content"]

        self.assertIn("Abstract", front_matter)
        self.assertNotIn("Table of Contents", front_matter)

    def test_find_documents_only_returns_supported_formats(self) -> None:
        (self.root / "notes.md").write_text("ignored", encoding="utf-8")

        self.assertEqual(find_documents(self.root), [self.path])


if __name__ == "__main__":
    unittest.main()
