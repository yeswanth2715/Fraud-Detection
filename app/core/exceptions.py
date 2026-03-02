class FinanceRiskException(Exception):
    """Base exception for the Finance Risk Engine"""

    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class PredictionException(FinanceRiskException):
    """Raised when prediction fails"""
    pass


class ValidationException(FinanceRiskException):
    """Raised for custom validation errors"""
    pass