from fastapi import APIRouter

from soundspace.api.routes import health, recommend

router = APIRouter()
router.include_router(health.router)
router.include_router(recommend.router)
