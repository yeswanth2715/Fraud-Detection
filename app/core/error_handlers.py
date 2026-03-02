from fastapi import Request
from fastapi.responses import JSONResponse
from app.core.exceptions import FinanceRiskException
from app.core.logger import logger


async def finance_exception_handler(request: Request, exc: FinanceRiskException):
    logger.error(f"FinanceRiskException: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message},
    )


async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled Exception occurred")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error"},
    )