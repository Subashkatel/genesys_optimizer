# The caching system in our optimizer is designed to avoid redundant work by recognizing when layers have similar structures, which is common in modern neural networks. Let me explain with some examples:

# Consider a ResNet model with multiple residual blocks that have identical structure but different names:

# Without caching:

# Optimizer would generate tiling configurations for layer1_conv1
# Compile and simulate each configuration (~10 compilations/simulations)
# Repeat the exact same process for layer2_conv1 and layer3_conv1
# Total: ~30 compilations and simulations
# With caching:

# Optimizer generates a fingerprint for layer1_conv1 based on:
# Optimizer compiles and simulates configurations for layer1_conv1
# Finds the best configuration: {"OC": 32, "N": 1, "IC": 1, "KH": 1, "KW": 1, "OH": 14, "OW": 14}
# Caches this result with the fingerprint as key
# When processing layer2_conv1:
# Generates the same fingerprint (since the structure is identical)
# Finds a match in the cache
# Reuses the result without any compilation or simulation
# Same for layer3_conv1
# Total: ~10 compilations and simulations (just for the first layer)

import os
import json
import logging
import hashlib
from utils.logging_utils import setup_logging

logger = setup_logging("genesys_optimizer.cache")

class OptimizationCache:
    """
    Caches optimization results for similar layers to avoid redundant optimization.
    Similarity is determined by layer operation type, input/output shapes, and parameters.
    """
    
    def __init__(self, model_name, cache_dir="layer_cache"):
        """
        Initialize the cache.
        
        Args:
            model_name: Name of the model
            cache_dir: Directory to store cache files
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, f"{model_name}_layer_cache.json")
        self.cache = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing cache if available
        self._load_cache()
    
    def _load_cache(self):
        """Load existing cache if available."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded layer cache with {len(self.cache)} entries")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache file: {str(e)}")
                self.cache = {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            # Ensure cache directory exists
            os.makedirs(self.cache_dir, exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            logger.info(f"Cache saved with {len(self.cache)} entries")
            return True
        except (IOError, OSError) as e:
            logger.error(f"Failed to save cache: {str(e)}")
            return False
    
    def generate_layer_fingerprint(self, layer_info):
        """
        Generate a unique fingerprint for a layer based on its characteristics.
        
        Args:
            layer_info: Dictionary containing layer information
            
        Returns:
            String fingerprint that uniquely identifies layers with the same structure
        """
        # Extract key characteristics that determine layer similarity
        operation = layer_info.get("operation", "")
        dimensions = layer_info.get("dimensions", {})
        
        # Create a dictionary of the layer's key characteristics
        # This is what determines if two layers are "similar enough" to share optimizations
        layer_characteristics = {
            "operation": operation,
            "dimensions": dimensions,
            # Exclude instance_id to match similar layers with different IDs
        }
        
        # Convert to a stable string representation and hash it
        char_str = json.dumps(layer_characteristics, sort_keys=True)
        fingerprint = hashlib.md5(char_str.encode()).hexdigest()
        
        return fingerprint
    
    def get_cached_result(self, layer_info):
        """
        Get cached optimization result for a similar layer if available.
        
        Args:
            layer_info: Dictionary containing layer information
            
        Returns:
            Cached optimization result or None if not found
        """
        fingerprint = self.generate_layer_fingerprint(layer_info)
        if fingerprint in self.cache:
            cached_entry = self.cache[fingerprint]
            logger.info(f"Cache hit for layer with fingerprint {fingerprint[:8]}...")
            return cached_entry
        
        logger.debug(f"Cache miss for layer with fingerprint {fingerprint[:8]}...")
        return None
    
    def add_to_cache(self, layer_info, optimization_result):
        """
        Add a layer's optimization result to the cache.
        
        Args:
            layer_info: Dictionary containing layer information
            optimization_result: Result of layer optimization
            
        Returns:
            True if successfully added to cache, False otherwise
        """
        fingerprint = self.generate_layer_fingerprint(layer_info)
        
        # Store in memory cache
        self.cache[fingerprint] = optimization_result
        
        # Save updated cache to disk
        return self._save_cache()
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache = {}
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
                logger.info("Cache file cleared")
                return True
            except OSError as e:
                logger.error(f"Failed to delete cache file: {str(e)}")
                return False
        return True