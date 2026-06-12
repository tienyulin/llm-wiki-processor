import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api import api_router
from core.deps import get_processor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the dependency singletons so misconfiguration fails at boot,
    not on the first request. Tests use lazy construction instead (their
    TestClient does not run the lifespan)."""
    if not os.getenv("PROCESSOR_API_KEY"):
        logger.warning(
            "PROCESSOR_API_KEY not set — /process is UNAUTHENTICATED (dev mode). "
            "Set it before exposing this service beyond localhost."
        )
    get_processor()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Wiki Processor", version="0.1.0", lifespan=lifespan)
    app.include_router(api_router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
