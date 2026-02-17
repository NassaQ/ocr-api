import os
from urllib.parse import quote_plus
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- OCR ---
    OUTPUT_DIR: str = "documents"
    GPU: bool = False

    # --- Message Broker ---
    MESSAGE_BROKER_URL: str = "amqp://guest:guest@localhost:5672/"
    OCR_QUEUE_NAME: str = "ocr_queue"

    # --- SQL Server ---
    SQL_SERVER: str
    SQL_DB_NAME: str
    SQL_USER: str
    SQL_PASS: str
    SQL_DRIVER: str = "ODBC Driver 18 for SQL Server"
    SQL_CONNECT_TIMEOUT: int = 60
    SQL_MAX_RETRIES: int = 3
    SQL_RETRY_DELAY_BASE: int = 2

    # --- Azure Blob Storage ---
    BLOB_CONNECTION_STR: str
    BLOB_STORAGE_CONTAINER_NAME: str

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def SQL_CONNECTION_STRING(self) -> str:
        encoded_pass = quote_plus(self.SQL_PASS)
        return (
            f"mssql+aioodbc://{self.SQL_USER}:{encoded_pass}@"
            f"{self.SQL_SERVER}/{self.SQL_DB_NAME}"
            f"?driver={quote_plus(self.SQL_DRIVER)}"
            "&TrustServerCertificate=yes"
        )


settings = Settings()  # type: ignore

os.makedirs(settings.OUTPUT_DIR, exist_ok=True)