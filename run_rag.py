#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import importlib.util
import types

def create_haystack_stub():
    """Create a stub for the haystack module to allow the script to run.
    This is a temporary solution to avoid having to install haystack."""
    
    print("⚠️ Creating haystack stub module...")
    
    # Create a stub for the haystack module
    haystack = types.ModuleType("haystack")
    sys.modules["haystack"] = haystack
    
    # Create basic classes and submodules
    haystack.Pipeline = type("Pipeline", (), {
        "__init__": lambda self: None,
        "add_node": lambda self, component, name, inputs: None,
        "run": lambda self, **kwargs: {"results": ["This is a stub response. Haystack is not installed properly."]}
    })
    
    haystack.schema = types.ModuleType("haystack.schema")
    sys.modules["haystack.schema"] = haystack.schema
    haystack.schema.Document = type("Document", (), {
        "__init__": lambda self, content="", meta=None: None,
    })
    
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
    return True

def check_and_install_dependencies():
    """Check if haystack is installed and install if needed."""
    try:
        import haystack
        print("✅ Haystack is already installed.")
        return True
    except ImportError:
        print("⚠️ Haystack not found. Installing farm-haystack...")
        
        # Try to install farm-haystack
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "farm-haystack==1.6.0", "-q"])
            print("✅ Successfully installed farm-haystack!")
            
            # Apply any necessary patches
            try:
                # Import and apply the patches from fix_bm25.py
                spec = importlib.util.spec_from_file_location("fix_bm25", "fix_bm25.py")
                fix_bm25 = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(fix_bm25)
                
                fix_bm25.patch_docstore()
                fix_bm25.patch_ranker()
                fix_bm25.patch_component()
                print("✅ Applied fixes from fix_bm25.py")
            except Exception as e:
                print(f"⚠️ Error applying patches: {e}")
                
            return True
        except Exception as e:
            print(f"⚠️ Error installing farm-haystack: {e}")
            
            # Try to install haystack-ai as a fallback
            try:
                print("⚠️ Trying to install haystack-ai as a fallback...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "haystack-ai", "-q"])
                print("⚠️ Installed haystack-ai, but this might require code changes.")
                print("⚠️ You may need to modify imports in your code.")
                return True
            except Exception as e2:
                print(f"⚠️ Failed to install any version of haystack: {e2}")
                print("⚠️ Creating a stub module to allow the script to run in demo mode.")
                return create_haystack_stub()

def run_main_script(args):
    """Run the main.py script with the provided arguments."""
    cmd = [sys.executable, "main.py"] + args
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    # Get all arguments passed to this script
    script_args = sys.argv[1:]
    
    # Check and install dependencies
    if check_and_install_dependencies():
        # Run the main script with all arguments
        run_main_script(script_args)
    else:
        print("❌ Cannot run main.py due to dependency installation issues.")
        sys.exit(1) 