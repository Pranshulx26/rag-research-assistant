from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # App
    app_name: str = 'RAG Research Assistant'
    app_version: str = '1.0.0'
    debug: bool = False 

    # Gemini
    gemini_api_key: str 
    gemini_model: str = 'gemini-2.5-flash'

    # RAG parambers 
    chunk_size: int = 1000
    chunk_overlap: int = 200 
    top_k_results: int = 4 

    # Storage paths 
    upload_dir: Path = Path('data/uploads')
    vectorstore_dir: Path = Path('data/vectorstore')

    model_config = SettingsConfigDict(
        env_file='.env',
        case_sensitive=False
    )

    def model_post_init(self, __context):
        # Create directories if they don't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)

@lru_cache()
def get_settings() -> Settings:
    return Settings()