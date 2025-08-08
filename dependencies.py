from fastapi import Depends, HTTPException, status, Cookie
from jose import JWTError, jwt
from typing import Optional

from security import SECRET_KEY, ALGORITHM
from models import UserProfile

async def get_current_user(token: Optional[str] = Cookie(None)):
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # In a real app, you'd fetch the user from the database here.
    # For now, we'll create a UserProfile from the token data.
    user = UserProfile(id=payload.get("id", 1), email=email)
    if user is None:
        raise credentials_exception
    return user
