"""アプリ共通のエラー型と簡易ライザー群。"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from fastapi import status


class AppError(Exception):
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, detail: Any = None, *, headers: Optional[Mapping[str, str]] = None):
        super().__init__(str(detail) if detail is not None else None)
        self.detail: Any = detail if detail is not None else "Internal Server Error"
        self.headers: Optional[dict[str, str]] = dict(headers) if headers else None


class BadRequestError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST


class UnauthorizedError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED


class ForbiddenError(AppError):
    status_code = status.HTTP_403_FORBIDDEN


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND


class ConflictError(AppError):
    status_code = status.HTTP_409_CONFLICT


class UnprocessableEntityError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY


class UpstreamError(AppError):
    """外部サービス（上流）が 4xx/5xx を返した場合に使用。"""

    status_code = status.HTTP_502_BAD_GATEWAY


class ServiceUnavailableError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class InternalServerError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


# 省略記法（例外送出ヘルパ）

def bad_request(detail: Any = "Bad Request") -> None:
    raise BadRequestError(detail)


def unauthorized(detail: Any = "Not authenticated", *, headers: Optional[Mapping[str, str]] = None) -> None:
    raise UnauthorizedError(detail, headers=headers)


def forbidden(detail: Any = "Forbidden") -> None:
    raise ForbiddenError(detail)


def not_found(detail: Any = "Not Found") -> None:
    raise NotFoundError(detail)


def conflict(detail: Any = "Conflict") -> None:
    raise ConflictError(detail)


def unprocessable(detail: Any = "Unprocessable Entity") -> None:
    raise UnprocessableEntityError(detail)


def upstream(detail: Any) -> None:
    raise UpstreamError(detail)


def internal(detail: Any = "Internal Server Error") -> None:
    raise InternalServerError(detail)
