"""FastAPI 依存関係（現在ユーザーの取得）。"""

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from models import UserProfile
from security import ALGORITHM, SECRET_KEY
from errors import unauthorized

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login", auto_error=False)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserProfile:
    if not token:
        unauthorized("Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    
    def cred_exc():
        unauthorized("Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str | None = payload.get("sub")
        if email is None:
            cred_exc()
    except JWTError:
        cred_exc()

    return UserProfile(id=payload.get("id", 1), email=email)
