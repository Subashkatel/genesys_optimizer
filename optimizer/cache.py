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
        logger.info(f"Cache initialized for model {model_name} with {len(self.cache)} entries")
    
    def _load_cache(self):
        """Load existing cache if available."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded layer cache with {len(self.cache)} entries from {self.cache_file}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache file ({self.cache_file}): {str(e)}")
                self.cache = {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            # Ensure cache directory exists
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Write to temporary file first
            temp_file = f"{self.cache_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            
            # Replace old cache with new one (atomic operation on most filesystems)
            os.replace(temp_file, self.cache_file)
            
            logger.info(f"Cache saved with {len(self.cache)} entries to {self.cache_file}")
            return True
        except (IOError, OSError) as e:
            logger.error(f"Failed to save cache to {self.cache_file}: {str(e)}")
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
        if not layer_info:
            logger.warning("Cannot get cached result for empty layer_info")
            return None
            
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
        if not layer_info or not optimization_result:
            logger.warning("Cannot add empty layer_info or optimization_result to cache")
            return False
            
        fingerprint = self.generate_layer_fingerprint(layer_info)
        
        # Store in memory cache
        self.cache[fingerprint] = optimization_result
        logger.info(f"Added result to cache with fingerprint {fingerprint[:8]}")
        
        # Save updated cache to disk
        return self._save_cache()
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache = {}
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
                logger.info(f"Cache file cleared: {self.cache_file}")
                return True
            except OSError as e:
                logger.error(f"Failed to delete cache file {self.cache_file}: {str(e)}")
                return False
        return True