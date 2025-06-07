import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration settings loaded from environment variables"""
    
    # Gemini AI API settings
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Azure Cohere settings
    AZURE_COHERE_API_KEY = os.getenv("AZURE_COHERE_API_KEY", "")
    AZURE_COHERE_ENDPOINT = os.getenv("AZURE_COHERE_ENDPOINT", "")
    COHERE_MODEL_NAME = os.getenv("COHERE_MODEL_NAME", "embed-v-4-0")
    
    # Qdrant database settings
    QDRANT_URL = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    
    # Application settings
    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("APP_PORT", "8000"))
    DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"
    
    @classmethod
    def validate_config(cls):
        """Check if all required configuration is present"""
        required_vars = [
            ("GEMINI_API_KEY", cls.GEMINI_API_KEY),
            ("AZURE_COHERE_API_KEY", cls.AZURE_COHERE_API_KEY),
            ("AZURE_COHERE_ENDPOINT", cls.AZURE_COHERE_ENDPOINT),
            ("QDRANT_URL", cls.QDRANT_URL),
            ("QDRANT_API_KEY", cls.QDRANT_API_KEY),
        ]
        
        missing_vars = []
        for var_name, var_value in required_vars:
            if not var_value:
                missing_vars.append(var_name)
        
        if missing_vars:
            print(f"WARNING: Missing environment variables: {', '.join(missing_vars)}")
            print("Please check your .env file")
            return False
        

        return True
    
    @classmethod
    def get_cohere_headers(cls):
        """Get headers for Cohere API requests"""
        return {
            "Content-Type": "application/json",
            "api-key": cls.AZURE_COHERE_API_KEY
        }
    
    @classmethod
    def get_cohere_endpoint_url(cls):
        """Get full URL for Cohere embeddings endpoint"""
        return f"{cls.AZURE_COHERE_ENDPOINT}/embeddings" 