from pathlib import Path

from pydantic_settings import BaseSettings

FILE = Path(__file__)
ROOT = FILE.parent.parent


class Settings(BaseSettings):
    # PROJECT INFORMATION
    CORS_ORIGINS: list
    CORS_HEADERS: list

    class Config:
        env_file = ROOT / '.env'


settings = Settings()
