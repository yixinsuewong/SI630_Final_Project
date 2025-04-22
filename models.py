from haystack import Pipeline
from haystack.schema import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, SentenceTransformersRanker, EmbeddingRetriever
from haystack.nodes.prompt import PromptNode
from haystack.nodes.prompt.prompt_model import PromptModel

import torch
from typing import List

default_prompt = """Answer the question using the provided context. Please answer as concisely as possible.
{meta['few_shot_example']}


Context: {join(documents, delimiter=new_line)}

Question: {query}

Answer: """


def baseline(
    documents: List[Document], use_gpu: bool = False, top_k: int = 1
) -> Pipeline:
    """
    Baseline RAG pipeline with BM25 retriever, MiniLM reranker, and LaMini-Flan-T5 prompter
    """
    # Ensure use_bm25=True for BM25Retriever compatibility
    document_store = InMemoryDocumentStore(use_gpu=use_gpu, use_bm25=True)
    document_store.write_documents(documents)
    retriever = BM25Retriever(document_store=document_store, top_k=100)

    # Explicitly specify only the parameters the reranker accepts
    reranker = SentenceTransformersRanker(
        model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2",
        top_k=top_k,
    )

    prompter = PromptNode(
        model_name_or_path="MBZUAI/LaMini-Flan-T5-783M",
        default_prompt_template=default_prompt,
        model_kwargs={"model_max_length": 2048, "torch_dtype": torch.bfloat16},
        use_gpu=use_gpu,
    )

    p = Pipeline()
    p.add_node(component=retriever, name="Retriever", inputs=["Query"])
    # Try explicitly removing any extra kwargs when adding to pipeline
    p.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
    p.add_node(component=prompter, name="prompt_node", inputs=["Reranker"])
    return p


def embed_retriever(
    documents: List[Document], use_gpu: bool = False, top_k: int = 1
) -> Pipeline:
    """
    RAG pipeline with embedding-based retriever instead of BM25
    """
    # For embedding retriever, we don't need use_bm25=True
    document_store = InMemoryDocumentStore(use_gpu=use_gpu, use_bm25=True)
    document_store.write_documents(documents)

    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        model_format="sentence_transformers",
        top_k=20,
        use_gpu=use_gpu,
    )
    document_store.update_embeddings(retriever=retriever)

    # Remove use_gpu parameter from SentenceTransformersRanker
    reranker = SentenceTransformersRanker(
        model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2",
        top_k=top_k,
    )

    prompter = PromptNode(
        model_name_or_path="MBZUAI/LaMini-Flan-T5-783M",
        default_prompt_template=default_prompt,
        model_kwargs={"model_max_length": 2048, "torch_dtype": torch.bfloat16},
        use_gpu=use_gpu,
    )

    p = Pipeline()
    p.add_node(component=retriever, name="Retriever", inputs=["Query"])
    p.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
    p.add_node(component=prompter, name="prompt_node", inputs=["Reranker"])
    return p


def embed_meta(
    documents: List[Document], use_gpu: bool = False, top_k: int = 1
) -> Pipeline:
    """
    RAG pipeline with embedding-based retriever that also embeds metadata fields
    """
    # For embedding retriever, we don't need use_bm25=True
    document_store = InMemoryDocumentStore(use_gpu=use_gpu, use_bm25=True)
    document_store.write_documents(documents)

    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        model_format="sentence_transformers",
        top_k=20,
        embed_meta_fields=["title"],
        use_gpu=use_gpu,
    )
    document_store.update_embeddings(retriever=retriever)

    # Remove use_gpu parameter from SentenceTransformersRanker
    reranker = SentenceTransformersRanker(
        model_name_or_path="cross-encoder/qnli-distilroberta-base",
        top_k=top_k,
    )

    prompter = PromptNode(
        model_name_or_path="MBZUAI/LaMini-Flan-T5-783M",
        default_prompt_template=default_prompt,
        model_kwargs={"model_max_length": 2048, "torch_dtype": torch.bfloat16},
        use_gpu=use_gpu,
    )

    p = Pipeline()
    p.add_node(component=retriever, name="Retriever", inputs=["Query"])
    p.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
    p.add_node(component=prompter, name="prompt_node", inputs=["Reranker"])
    return p


def different_reranker(
    documents: List[Document], use_gpu: bool = False, top_k: int = 1
) -> Pipeline:
    """
    RAG pipeline with different reranker model (QNLI DistilRoBERTa)
    """
    # Since we use BM25Retriever, ensure use_bm25=True
    document_store = InMemoryDocumentStore(use_gpu=use_gpu, use_bm25=True)
    document_store.write_documents(documents)
    retriever = BM25Retriever(document_store=document_store, top_k=100)

    # Remove use_gpu parameter from SentenceTransformersRanker
    reranker = SentenceTransformersRanker(
        model_name_or_path="cross-encoder/qnli-distilroberta-base",
        top_k=top_k,
    )

    prompter = PromptNode(
        model_name_or_path="MBZUAI/LaMini-Flan-T5-783M",
        default_prompt_template=default_prompt,
        model_kwargs={"model_max_length": 2048, "torch_dtype": torch.bfloat16},
        use_gpu=use_gpu,
    )

    p = Pipeline()
    p.add_node(component=retriever, name="Retriever", inputs=["Query"])
    p.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
    p.add_node(component=prompter, name="prompt_node", inputs=["Reranker"])
    return p


def squad(
    documents: List[Document], use_gpu: bool = False, top_k: int = 1
) -> Pipeline:
    """
    RAG pipeline similar to SQuAD configuration
    """
    # Since we use BM25Retriever, ensure use_bm25=True
    document_store = InMemoryDocumentStore(use_gpu=use_gpu, use_bm25=True)
    document_store.write_documents(documents)
    retriever = BM25Retriever(document_store=document_store, top_k=100)

    # Remove use_gpu parameter from SentenceTransformersRanker
    reranker = SentenceTransformersRanker(
        model_name_or_path="cross-encoder/qnli-distilroberta-base",
        top_k=top_k,
    )

    prompter = PromptNode(
        model_name_or_path="MBZUAI/LaMini-Flan-T5-783M",
        default_prompt_template=default_prompt,
        model_kwargs={"model_max_length": 2048, "torch_dtype": torch.bfloat16},
        use_gpu=use_gpu,
    )

    p = Pipeline()
    p.add_node(component=retriever, name="Retriever", inputs=["Query"])
    p.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
    p.add_node(component=prompter, name="prompt_node", inputs=["Reranker"])
    return p


def few_shot_pipeline(
    questions: List[str],
    answers: List[str],
    n_examples: int = 1,
    use_gpu: bool = False,
) -> Pipeline:
    """
    Pipeline to retrieve few-shot examples for the prompter
    """
    docs = []
    for q, a in zip(questions, answers):
        docs.append(Document(content=q, meta={"answer": a}))

    # Make sure BM25 is explicitly enabled for the few-shot document store
    document_store = InMemoryDocumentStore(use_gpu=use_gpu, use_bm25=True)
    document_store.write_documents(docs)
    
    # Create the retriever after writing documents to ensure BM25 is properly initialized
    retriever = BM25Retriever(document_store=document_store, top_k=n_examples)

    p = Pipeline()
    p.add_node(component=retriever, name="Retriever", inputs=["Query"])
    return p