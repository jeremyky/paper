import os
import sys
import logging
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the complete analysis pipeline"""
    try:
        # Setup paths
        input_dir = os.path.join(project_root, "data", "raw", "mm_unique_objects")
        feature_dir = os.path.join(project_root, "src", "unique_objects_features")
        
        # Ensure directories exist
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(feature_dir, exist_ok=True)
        
        # 1. Run unique-objects-analysis.py
        logger.info("Running original analysis...")
        import src.unique_objects_analysis
        
        # 2. Run unique-objects-analysis-global.py
        logger.info("Running global analysis...")
        import src.unique_objects_analysis_global
        
        logger.info("Analysis complete!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 