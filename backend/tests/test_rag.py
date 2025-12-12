from langchain.schema import Document

from app.rag.retriever import chunk_text, docs_to_context


def test_chunk_text_produces_chunks():
    text = "This is a test document. " * 50  # long enough to be chunked
    docs = chunk_text(text=text, base_metadata={"source": "unit-test"}, namespace="test-ns")

    assert len(docs) > 0
    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.page_content.strip() != ""
        assert doc.metadata.get("source") == "unit-test"
        assert doc.metadata.get("namespace") == "test-ns"


def test_docs_to_context_includes_metadata_and_content():
    docs = [
        Document(
            page_content="First doc content.",
            metadata={"source": "doc1", "title": "First"},
        ),
        Document(
            page_content="Second doc content.",
            metadata={"source": "doc2", "title": "Second"},
        ),
    ]

    context = docs_to_context(docs)

    assert "First doc content." in context
    assert "Second doc content." in context
    # Metadata such as source/title should appear in the context header
    assert "source=doc1" in context or "title=First" in context
    assert "source=doc2" in context or "title=Second" in context
