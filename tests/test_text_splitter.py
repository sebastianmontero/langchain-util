import unittest
import pytest
from typing import List
from langchain_util.text_splitter import RecursiveCharacterTextSplitterWithContext
from langchain.docstore.document import Document

TEXT_TO_SPLIT_1 = """Hi.\n\nI'm Harrison.\n\nHow? Are? You?\nOkay then f f f f.
This is a weird text to write, but gotta test the splittingggg some how.

Bye!\n\n-H."""


class TestRecursiveCharacterTextSplitterWithContext(unittest.TestCase):

    def test_init_validations(self):
        with pytest.raises(ValueError, match="context_perc_of_chunk_size must be greater than 0 and less than 90 percent"):
            RecursiveCharacterTextSplitterWithContext(
                context_perc_of_chunk_size=-1)
        with pytest.raises(ValueError, match="context_perc_of_chunk_size must be greater than 0 and less than 90 percent"):
            RecursiveCharacterTextSplitterWithContext(
                context_perc_of_chunk_size=-90.1)

    def test_split_documents_without_context(self):
        docs = [Document(page_content=TEXT_TO_SPLIT_1)]

        splitter = RecursiveCharacterTextSplitterWithContext(
            chunk_size=10, chunk_overlap=1)
        results = splitter.split_documents(docs)

        expected_docs = [
            Document(page_content="Hi.", metadata={}),
            Document(page_content="I'm", metadata={}),
            Document(page_content="Harrison.", metadata={}),
            Document(page_content="How? Are?", metadata={}),
            Document(page_content="You?", metadata={}),
            Document(page_content="Okay then", metadata={}),
            Document(page_content="f f f f.", metadata={}),
            Document(page_content="This is a", metadata={}),
            Document(page_content="a weird", metadata={}),
            Document(page_content="text to", metadata={}),
            Document(page_content="write, but", metadata={}),
            Document(page_content="gotta test", metadata={}),
            Document(page_content="the", metadata={}),
            Document(page_content="splittingg", metadata={}),
            Document(page_content="ggg", metadata={}),
            Document(page_content="some how.", metadata={}),
            Document(page_content="Bye!\n\n-H.", metadata={}),
        ]
        assert results == expected_docs

    def test_split_documents_with_context(self):
        docs = [
            Document(page_content=TEXT_TO_SPLIT_1,
                     metadata={"chunk-context": "Title"}),
            Document(page_content="Hola.\n\nQue tal?",
                     metadata={"chunk-context": "Context", "source":"s1"})
        ]

        splitter = RecursiveCharacterTextSplitterWithContext(
            chunk_size=17, chunk_overlap=1, context_perc_of_chunk_size=53)
        results = splitter.split_documents(docs)

        expected_docs = [
            Document(page_content="Title\n\nHi.", metadata={}),
            Document(page_content="Title\n\nI'm", metadata={}),
            Document(page_content="Title\n\nHarrison.", metadata={}),
            Document(page_content="Title\n\nHow? Are?", metadata={}),
            Document(page_content="Title\n\nYou?", metadata={}),
            Document(page_content="Title\n\nOkay then", metadata={}),
            Document(page_content="Title\n\nf f f f.", metadata={}),
            Document(page_content="Title\n\nThis is a", metadata={}),
            Document(page_content="Title\n\na weird", metadata={}),
            Document(page_content="Title\n\ntext to", metadata={}),
            Document(page_content="Title\n\nwrite, but", metadata={}),
            Document(page_content="Title\n\ngotta test", metadata={}),
            Document(page_content="Title\n\nthe", metadata={}),
            Document(page_content="Title\n\nsplittingg", metadata={}),
            Document(page_content="Title\n\nggg", metadata={}),
            Document(page_content="Title\n\nsome how.", metadata={}),
            Document(page_content="Title\n\nBye!\n\n-H.", metadata={}),
            Document(page_content="Context\n\nHola.", metadata={"source":"s1"}),
            Document(page_content="Context\n\nQue tal?", metadata={"source":"s1"}),
            
            
        ]
        assert results == expected_docs

    def test_split_documents_should_fail_for_context_too_long(self):
        docs = [Document(page_content=TEXT_TO_SPLIT_1,
                         metadata={"chunk-context": "Title"})]

        splitter = RecursiveCharacterTextSplitterWithContext(
            chunk_size=17, chunk_overlap=1, context_perc_of_chunk_size=41)
        with pytest.raises(RuntimeError, match="Chunk context is too long: Title\n\n"):
            splitter.split_documents(docs)
