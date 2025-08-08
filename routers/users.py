from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from datetime import timedelta

from database import database
from models import UserCreate, UserLogin, UserProfile, users
from dependencies import get_current_user
from security import create_access_token, verify_password, get_password_hash, ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter()

@router.post("/create")
async def create_account(user: UserCreate):
    # Check if user already exists
    query = users.select().where(users.c.email == user.email)
    if await database.fetch_one(query):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # Hash password and create user
    hashed_password = get_password_hash(user.password)
    query = users.insert().values(email=user.email, hashed_password=hashed_password)
    user_id = await database.execute(query)
    
    return {"message": "Account created successfully", "user_id": user_id, "email": user.email}

@router.post("/login")
async def login(user: UserLogin):
    # Find user in database
    query = users.select().where(users.c.email == user.email)
    db_user = await database.fetch_one(query)

    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user["email"], "id": db_user["id"]}, expires_delta=access_token_expires
    )

    response = JSONResponse(content={"message": "Login successful"})
    response.set_cookie(
        key="token",
        value=access_token,
        httponly=True,
        max_age=access_token_expires.total_seconds(),
        samesite="lax",
    )
    return response

@router.get("/profile", response_model=UserProfile)
async def get_profile(current_user: UserProfile = Depends(get_current_user)):
    return current_user
