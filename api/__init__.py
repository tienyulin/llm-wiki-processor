from fastapi import APIRouter

from api.routers import admin, process, system

api_router = APIRouter()
api_router.include_router(process.router)
api_router.include_router(admin.router)
api_router.include_router(system.router)
