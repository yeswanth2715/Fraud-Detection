from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Finance Risk Engine"

    MODEL_PATH: str = "models/model.joblib"

    HIGH_RISK_THRESHOLD: float = 0.85
    MEDIUM_RISK_THRESHOLD: float = 0.60
    LOW_RISK_THRESHOLD: float = 0.30

    class Config:
        env_file = ".env"


settings = Settings()