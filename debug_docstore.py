"""
Debugging script to check document store configuration for BM25 compatibility
"""
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, EmbeddingRetriever
from haystack.schema import Document

def check_docstore_bm25_compatibility():
    """
    Test if document stores are properly configured for BM25 retrieval
    """
    print("Testing document store BM25 compatibility...")
    
    # Create a test document
    test_doc = Document(content="This is a test document for BM25 retrieval")
    
    # Test 1: Document store with BM25 enabled
    print("\nTest 1: InMemoryDocumentStore with use_bm25=True")
    try:
        doc_store_bm25 = InMemoryDocumentStore(use_bm25=True)
        doc_store_bm25.write_documents([test_doc])
        retriever = BM25Retriever(document_store=doc_store_bm25)
        results = retriever.retrieve("test document")
        print("✅ Success! BM25 retrieval works with use_bm25=True")
    except Exception as e:
        print(f"❌ Error: {e}")
        
    # Test 2: Document store without BM25 enabled
    print("\nTest 2: InMemoryDocumentStore with use_bm25=False (should fail)")
    try:
        doc_store_no_bm25 = InMemoryDocumentStore(use_bm25=False)
        doc_store_no_bm25.write_documents([test_doc])
        retriever = BM25Retriever(document_store=doc_store_no_bm25)
        results = retriever.retrieve("test document")
        print("❓ Unexpected: BM25 retrieval works without use_bm25=True")
    except Exception as e:
        print(f"✅ Expected error: {e}")
        
    # Test 3: Embedding retriever (doesn't need BM25)
    print("\nTest 3: EmbeddingRetriever with regular document store")
    try:
        doc_store_embed = InMemoryDocumentStore()
        doc_store_embed.write_documents([test_doc])
        retriever = EmbeddingRetriever(
            document_store=doc_store_embed,
            embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
            model_format="sentence_transformers"
        )
        doc_store_embed.update_embeddings(retriever)
        results = retriever.retrieve("test document")
        print("✅ Success! Embedding retrieval works with regular document store")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_docstore_bm25_compatibility()