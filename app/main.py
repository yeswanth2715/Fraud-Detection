from fastapi import FastAPI
from app.api.routes import router
from app.core.exceptions import FinanceRiskException
from app.core.error_handlers import (
    finance_exception_handler,
    generic_exception_handler,
)


app = FastAPI(title="Finance Risk Engine")

# Include API routes
app.include_router(router)
