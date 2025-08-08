from fastapi import APIRouter

router = APIRouter()

# 確認テスト諸々

@router.get("/test")
async def test_endpoint():
    return { "test": "success" }
