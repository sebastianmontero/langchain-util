import copy
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Iterable, Sequence, Any
from langchain.docstore.document import Document
from langchain.schema import BaseDocumentTransformer


class TextSplitterWithContext(BaseDocumentTransformer, ABC):
    """Interface for splitting text into chunks. A context to add to the chunk can be provided via
       document metadata, the size of the context will be considered when splitting the text"""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        context_key: str = "chunk-context",
        context_separator: str = "\n\n",
        context_perc_of_chunk_size: float = 20
    ):
        """Create a new TextSplitter."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        if context_perc_of_chunk_size < 0 or context_perc_of_chunk_size > 90:
            raise ValueError(
                f"context_perc_of_chunk_size must be greater than 0 and less than 90 percent"
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._context_key = context_key
        self._context_separator = context_separator
        self._context_perc_of_chunk_size = context_perc_of_chunk_size / 100

    @abstractmethod
    def split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into multiple components."""

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            context_length = 0
            context = ""
            if _metadatas[i].get(self._context_key):
                context = _metadatas[i].pop(
                    self._context_key) + self._context_separator
                context_length = self._length_function(context)
                if context_length / self._chunk_size > self._context_perc_of_chunk_size:
                    raise RuntimeError(f"Chunk context is too long: {context}")

            for chunk in self.split_text(text, self._chunk_size - context_length):
                new_doc = Document(
                    page_content=context + chunk, metadata=copy.deepcopy(_metadatas[i])
                )
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.create_documents(texts, metadatas=metadatas)

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: Iterable[str], separator: str, chunk_size: int) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > chunk_size
            ):
                if total > chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len +
                            (separator_len if len(current_doc) > 0 else 0)
                        > chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform sequence of documents by splitting them."""
        return self.split_documents(list(documents))

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Asynchronously transform a sequence of documents by splitting them."""
        raise NotImplementedError


class RecursiveCharacterTextSplitterWithContext(TextSplitterWithContext):
    """Implementation of splitting text that looks at characters.

    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(self, separators: Optional[List[str]] = None, **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = self._separators[-1]
        for _s in self._separators:
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                break
        # Now that we have the separator, split the text
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        for s in splits:
            if self._length_function(s) < chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(
                        _good_splits, separator, chunk_size)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                other_info = self.split_text(s, chunk_size)
                final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(
                _good_splits, separator, chunk_size)
            final_chunks.extend(merged_text)
        return final_chunks
