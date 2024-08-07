
from fastapi import APIRouter

router = APIRouter()

@router.get("/inference")
async def inference():
    return "Success!"