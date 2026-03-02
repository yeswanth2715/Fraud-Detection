from fastapi import APIRouter
from app.api.schemas import TransactionRequest, PredictionResponse
from app.core.logger import logger
from app.core.exceptions import PredictionException
from app.models.predict import predict_transaction

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    try:
        logger.info(f"Received transaction: {transaction.dict()}")

        result = predict_transaction(transaction.dict())

        logger.info(
            f"Prediction complete | probability={result['fraud_probability']} | risk={result['risk_level']}"
        )

        return PredictionResponse(
            fraud_probability=result["fraud_probability"],
            risk_level=result["risk_level"],
        )

    except PredictionException:
        raise
    except Exception as e:
        logger.error(f"Unexpected failure: {str(e)}")
        raise PredictionException("Prediction processing failed")