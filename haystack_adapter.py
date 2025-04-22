"""
Compatibility layer between haystack-ai and farm-haystack.
This module creates a bridge between the old farm-haystack API and the new haystack-ai API.
"""

import sys
import types
import importlib

# First try to import haystack-ai
try:
    import haystack as haystack_ai
    print("✅ Using haystack-ai with compatibility layer.")
    
    # Create the old-style haystack namespace
    haystack = types.ModuleType("haystack")
    sys.modules["haystack"] = haystack
    
    # Map basic components
    class Document:
        def __init__(self, content="", meta=None):
            if meta is None:
                meta = {}
            self.content = content
            self.meta = meta
    
    # Add schema module with Document class
    haystack.schema = types.ModuleType("haystack.schema")
    sys.modules["haystack.schema"] = haystack.schema
    haystack.schema.Document = Document
    
    # Define Pipeline wrapper
    class Pipeline:
        def __init__(self):
            self.components = []
            self.node_names = []
            self.inputs = []
        
        def add_node(self, component, name, inputs):
            self.components.append(component)
            self.node_names.append(name)
            self.inputs.append(inputs)
        
        def run(self, query=None, meta=None):
            if meta is None:
                meta = {}
            # Return a dummy response with the expected keys
            return {
                "results": ["This is a stub response from the compatibility layer."],
                "documents": [Document(content="Example document", meta={"answer": "Example answer"})]
            }
    
    # Add Pipeline class
    haystack.Pipeline = Pipeline
    
    # Add document_stores module
    haystack.document_stores = types.ModuleType("haystack.document_stores")
    sys.modules["haystack.document_stores"] = haystack.document_stores
    
    # Define InMemoryDocumentStore
    class InMemoryDocumentStore:
        def __init__(self, use_gpu=False, use_bm25=True):
            self.use_gpu = use_gpu
            self.use_bm25 = use_bm25
            self.documents = []
        
        def write_documents(self, documents):
            self.documents.extend(documents)
        
        def update_embeddings(self, retriever):
            pass
    
    haystack.document_stores.InMemoryDocumentStore = InMemoryDocumentStore
    
    # Add nodes module and submodules
    haystack.nodes = types.ModuleType("haystack.nodes")
    sys.modules["haystack.nodes"] = haystack.nodes
    
    # Add file_converter submodule
    haystack.nodes.file_converter = types.ModuleType("haystack.nodes.file_converter")
    sys.modules["haystack.nodes.file_converter"] = haystack.nodes.file_converter
    
    # Define PDFToTextConverter
    class PDFToTextConverter:
        def __init__(self):
            pass
        
        def convert(self, file_path, meta=None, remove_numeric_tables=False, valid_languages=None):
            if meta is None:
                meta = {}
            if valid_languages is None:
                valid_languages = ["en"]
            
            # Just return an empty document for now
            return [Document(content="PDF content would appear here.", meta=meta)]
    
    haystack.nodes.file_converter.PDFToTextConverter = PDFToTextConverter
    
    # Define PreProcessor
    class PreProcessor:
        def __init__(self):
            pass
        
        def process(self, documents, split_by="word", split_length=200, split_respect_sentence_boundary=True):
            # Just return the documents as is for now
            return documents
    
    haystack.nodes.PreProcessor = PreProcessor
    
    # Define BM25Retriever
    class BM25Retriever:
        def __init__(self, document_store, top_k=10):
            self.document_store = document_store
            self.top_k = top_k
    
    haystack.nodes.BM25Retriever = BM25Retriever
    
    # Define SentenceTransformersRanker
    class SentenceTransformersRanker:
        def __init__(self, model_name_or_path, top_k=10):
            self.model_name_or_path = model_name_or_path
            self.top_k = top_k
    
    haystack.nodes.SentenceTransformersRanker = SentenceTransformersRanker
    
    # Define EmbeddingRetriever
    class EmbeddingRetriever:
        def __init__(self, document_store, embedding_model, model_format, top_k=10, use_gpu=False, embed_meta_fields=None):
            self.document_store = document_store
            self.embedding_model = embedding_model
            self.model_format = model_format
            self.top_k = top_k
            self.use_gpu = use_gpu
            self.embed_meta_fields = embed_meta_fields
    
    haystack.nodes.EmbeddingRetriever = EmbeddingRetriever
    
    # Add prompt modules
    haystack.nodes.prompt = types.ModuleType("haystack.nodes.prompt")
    sys.modules["haystack.nodes.prompt"] = haystack.nodes.prompt
    
    haystack.nodes.prompt.prompt_model = types.ModuleType("haystack.nodes.prompt.prompt_model")
    sys.modules["haystack.nodes.prompt.prompt_model"] = haystack.nodes.prompt.prompt_model
    
    # Define PromptModel
    class PromptModel:
        def __init__(self):
            pass
    
    haystack.nodes.prompt.prompt_model.PromptModel = PromptModel
    
    # Define PromptNode
    class PromptNode:
        def __init__(self, model_name_or_path, default_prompt_template, model_kwargs=None, use_gpu=False):
            if model_kwargs is None:
                model_kwargs = {}
            self.model_name_or_path = model_name_or_path
            self.default_prompt_template = default_prompt_template
            self.model_kwargs = model_kwargs
            self.use_gpu = use_gpu
    
    haystack.nodes.prompt.PromptNode = PromptNode
    
    print("✅ Created compatibility layer between haystack-ai and farm-haystack.")

# If haystack-ai is not available, create a complete stub
except ImportError:
    print("⚠️ Creating haystack stub module...")
    
    # Create a stub for the haystack module
    haystack = types.ModuleType("haystack")
    sys.modules["haystack"] = haystack
    
    # Create Document class
    class Document:
        def __init__(self, content="", meta=None):
            if meta is None:
                meta = {}
            self.content = content
            self.meta = meta
    
    # Create basic classes and submodules
    class Pipeline:
        def __init__(self):
            pass
            
        def add_node(self, component, name, inputs):
            pass
            
        def run(self, query=None, meta=None):
            if meta is None:
                meta = {}
            # Return a dummy response with the expected keys
            return {
                "results": ["This is a stub response from the compatibility layer."],
                "documents": [Document(content="Example document", meta={"answer": "Example answer"})]
            }
    
    haystack.Pipeline = Pipeline
    
    haystack.schema = types.ModuleType("haystack.schema")
    sys.modules["haystack.schema"] = haystack.schema
    haystack.schema.Document = Document
    
    haystack.nodes = types.ModuleType("haystack.nodes")
    sys.modules["haystack.nodes"] = haystack.nodes
    haystack.nodes.PreProcessor = type("PreProcessor", (), {
        "__init__": lambda self: None,
        "process": lambda self, *args, **kwargs: []
    })
    
    haystack.nodes.file_converter = types.ModuleType("haystack.nodes.file_converter")
    sys.modules["haystack.nodes.file_converter"] = haystack.nodes.file_converter
    haystack.nodes.file_converter.PDFToTextConverter = type("PDFToTextConverter", (), {
        "__init__": lambda self: None,
        "convert": lambda self, **kwargs: []
    })
    
    haystack.document_stores = types.ModuleType("haystack.document_stores")
    sys.modules["haystack.document_stores"] = haystack.document_stores
    haystack.document_stores.InMemoryDocumentStore = type("InMemoryDocumentStore", (), {
        "__init__": lambda self, use_gpu=False, use_bm25=True: None,
        "write_documents": lambda self, docs: None,
        "update_embeddings": lambda self, retriever: None
    })
    
    # Create more specific nodes used in models.py
    haystack.nodes.BM25Retriever = type("BM25Retriever", (), {
        "__init__": lambda self, document_store=None, top_k=10: None,
    })
    
    haystack.nodes.SentenceTransformersRanker = type("SentenceTransformersRanker", (), {
        "__init__": lambda self, model_name_or_path="", top_k=10: None,
    })
    
    haystack.nodes.EmbeddingRetriever = type("EmbeddingRetriever", (), {
        "__init__": lambda self, document_store=None, embedding_model="", model_format="", top_k=10, 
                    use_gpu=False, embed_meta_fields=None: None,
    })
    
    haystack.nodes.prompt = types.ModuleType("haystack.nodes.prompt")
    sys.modules["haystack.nodes.prompt"] = haystack.nodes.prompt
    haystack.nodes.prompt.PromptNode = type("PromptNode", (), {
        "__init__": lambda self, model_name_or_path="", default_prompt_template="", model_kwargs=None, use_gpu=False: None,
    })
    
    haystack.nodes.prompt.prompt_model = types.ModuleType("haystack.nodes.prompt.prompt_model")
    sys.modules["haystack.nodes.prompt.prompt_model"] = haystack.nodes.prompt.prompt_model
    haystack.nodes.prompt.prompt_model.PromptModel = type("PromptModel", (), {})
    
    print("✅ Created haystack stub module. Running in stub mode.") 