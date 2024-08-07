from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class AppSettings(BaseSettings):
    app_env: str
    app_host: str = "0.0.0.0"
    app_port: int = 8087
    model_config = SettingsConfigDict()


settings = AppSettings()
