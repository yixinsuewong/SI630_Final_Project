# This file directly patches classes needed for BM25 and monkey patches fixes
import sys
import importlib.util
import inspect

# Simple version that just ensures InMemoryDocumentStore uses BM25
def patch_docstore():
    try:
        # Try to import directly from the known module path
        from haystack.document_stores.memory import InMemoryDocumentStore
        
        # Store the original init
        original_init = InMemoryDocumentStore.__init__
        
        # Define a new init function that always sets use_bm25=True
        def patched_init(self, *args, **kwargs):
            kwargs['use_bm25'] = True
            return original_init(self, *args, **kwargs)
        
        # Apply the patch
        InMemoryDocumentStore.__init__ = patched_init
        print("✅ BM25 patch applied to InMemoryDocumentStore")
        return True
    except ImportError:
        print("⚠️ Could not import InMemoryDocumentStore, BM25 patch not applied")
        return False

# Patch for SentenceTransformersRanker to handle use_gpu parameter
def patch_ranker():
    try:
        from haystack.nodes import SentenceTransformersRanker
        
        original_init = SentenceTransformersRanker.__init__
        
        def patched_init(self, *args, **kwargs):
            # Remove use_gpu if present
            if 'use_gpu' in kwargs:
                print(f"⚠️ Removing use_gpu parameter from SentenceTransformersRanker")
                del kwargs['use_gpu']
            
            # Call original init
            return original_init(self, *args, **kwargs)
        
        SentenceTransformersRanker.__init__ = patched_init
        print("✅ use_gpu patch applied to SentenceTransformersRanker")
        return True
    except ImportError:
        print("⚠️ Could not import SentenceTransformersRanker, use_gpu patch not applied")
        return False

# Patch for BaseComponent to handle parameter validation issues
def patch_component():
    try:
        from haystack.nodes.base import BaseComponent
        
        # Try to get the original get_params method
        if hasattr(BaseComponent, 'get_params'):
            original_get_params = BaseComponent.get_params
            
            # Define a patched version that handles KeyError
            def patched_get_params(self, return_defaults=False):
                try:
                    return original_get_params(self, return_defaults)
                except KeyError as e:
                    # Common problematic parameters
                    problematic_params = ["use_gpu", "model_name_or_path", "max_length"]
                    
                    error_param = str(e).strip("'")
                    if error_param in problematic_params:
                        # Just return a dict without the problematic parameters
                        try:
                            component_signature = inspect.signature(self.__init__).parameters
                            return {
                                key: value
                                for key, value in self.__dict__.items()
                                if key in component_signature and key not in problematic_params
                            }
                        except Exception:
                            # If that fails, return a minimal dict
                            return {}
                    else:
                        # For other KeyError issues, raise the original exception
                        raise
            
            # Apply the patch
            BaseComponent.get_params = patched_get_params
            print("✅ parameter validation patch applied to BaseComponent")
            return True
        else:
            print("⚠️ BaseComponent doesn't have a get_params method, patch not applied")
            return False
    except ImportError:
        print("⚠️ Could not import BaseComponent, parameter validation patch not applied")
        return False

# Apply all patches
success_docstore = patch_docstore()
success_ranker = patch_ranker()
success_component = patch_component()

if success_docstore and success_ranker and success_component:
    print("✅ All Haystack patches applied successfully")
else:
    print("⚠️ Some patches could not be applied")
