from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.inference import router as inference_router
from src.api.healthcheck import router as healthcheck_router

def init_app():
    router = APIRouter()
    router.include_router(healthcheck_router)
    router.include_router(inference_router)

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = init_app()
