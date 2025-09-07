"""Epson Connect API クライアント（最小限）。"""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()


class EpsonConfig:
    """Epson Connect API の設定（最小限: use_dummy / debug）。"""

    def __init__(
        self,
        use_dummy: Optional[bool] = None,
        debug: Optional[bool] = None,
        access_token: Optional[str] = None,
    ) -> None:
        self.use_dummy = bool(use_dummy) if use_dummy is not None else False
        self.debug = bool(debug) if debug is not None else True

        # use_dummy に応じてエンドポイント切替
        self.base_url = (
            "https://dummy-api.epsonconnect.com/api/2"
            if self.use_dummy
            else "https://api.epsonconnect.com/api/2"
        )
        self.auth_url = "https://auth.epsonconnect.com"

        # 認証情報は環境変数から取得
        self.api_key = os.getenv("EPSON_API_KEY")
        self.refresh_token = os.getenv("EPSON_REFRESH_TOKEN")
        self.client_id = os.getenv("EPSON_CLIENT_ID")
        self.client_secret = os.getenv("EPSON_CLIENT_SECRET")
        # アクセストークン（事前取得があれば優先）
        self.access_token = access_token or os.getenv("EPSON_ACCESS_TOKEN")

        # 必須: APIキー（他は必要時）
        if not self.api_key:
            raise ValueError("EPSON_API_KEY is required (set env)")


class EpsonToken:
    def __init__(self, access_token: str, expires_in: int) -> None:
        self.access_token = access_token
        # マージンを持って有効期限を計算
        self.expires_at = time.time() + max(0, int(expires_in) - 30)

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.expires_at


class EpsonClient:
    """Auth/Capability/ScanDestination/PrintJob/Upload/Start の最小実装。"""

    def __init__(self, config: Optional[EpsonConfig] = None, *, timeout: float = 30.0) -> None:
        self.config = config or EpsonConfig()
        self._token: Optional[EpsonToken] = None
        self._http = httpx.AsyncClient(timeout=timeout)
        # 起動時に事前トークン/キャッシュを適用
        if self.config.access_token:
            self._token = EpsonToken(self.config.access_token, 365 * 24 * 3600)
        else:
            cached = self._load_cached_token()
            if cached and not cached.is_expired:
                self._token = cached

    # --- ログ補助 ---
    def _log(self, *args: Any) -> None:
        if getattr(self.config, "debug", False):
            print("[EpsonClient]", *args)

    def _ensure_logs_dir(self) -> str:
        """プロジェクト直下の logs ディレクトリを用意し、そのパスを返す。"""
        project_root = os.path.dirname(os.path.dirname(__file__))
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        return logs_dir

    def _write_timestamped_log(self, prefix: str, content: str) -> str:
        """logs/にタイムスタンプ付きで書き出す。パスを返す。"""
        logs_dir = self._ensure_logs_dir()
        ts_epoch = f"{time.time():.6f}"
        filename = f"{ts_epoch}_{prefix}.log"
        path = os.path.join(logs_dir, filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            self._log(f"Failed writing log file {path}: {e}")
        else:
            self._log(f"Wrote log file: {path}")
        return path

    # --- トークンキャッシュ（ファイル） ---
    def _token_cache_path(self) -> str:
        logs_dir = self._ensure_logs_dir()
        return os.path.join(logs_dir, "epson_access_token.json")

    def _load_cached_token(self) -> Optional["EpsonToken"]:
        path = self._token_cache_path()
        try:
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            access_token = data.get("access_token")
            expires_at = float(data.get("expires_at", 0))
            if not access_token or time.time() >= expires_at:
                return None
            tok = EpsonToken(access_token, 0)
            tok.expires_at = expires_at
            self._log("Loaded access token from cache; expires_at=", expires_at)
            return tok
        except Exception as e:
            self._log(f"Failed to load token cache: {e}")
            return None

    def _save_cached_token(self, token: "EpsonToken") -> None:
        path = self._token_cache_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "access_token": token.access_token,
                    "expires_at": token.expires_at,
                }, f, ensure_ascii=False)
            self._log("Saved access token to cache; expires_at=", token.expires_at)
        except Exception as e:
            self._log(f"Failed to save token cache: {e}")

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        headers = kwargs.get("headers") or {}
        safe_headers = headers  # マスキングなし
        # curl風の再現コマンドを構築
        curl_parts = ["curl -sS", f"-X {method}", f"'{url}'"]
        for hk, hv in safe_headers.items():
            curl_parts.append(f"-H '{hk}: {hv}'")
        if "json" in kwargs and kwargs["json"] is not None:
            try:
                body_str = json.dumps(kwargs["json"], ensure_ascii=False)
                curl_parts.append(f"-d '{body_str}'")
            except Exception:
                curl_parts.append("# [json body omitted]")
        elif "data" in kwargs and kwargs["data"] is not None:
            try:
                if isinstance(kwargs["data"], dict):
                    form_str = "&".join(f"{k}={v}" for k, v in kwargs["data"].items())
                    curl_parts.append(f"-d '{form_str}'")
                else:
                    curl_parts.append(f"-d '{str(kwargs['data'])}'")
            except Exception:
                curl_parts.append("# [form data omitted]")
        elif "content" in kwargs and kwargs["content"] is not None:
            curl_parts.append("# [binary content omitted]")
        self._log("->", " ".join(curl_parts))
        self._log(f"-> {method} {url}", safe_headers)
        resp = await self._http.request(method, url, **kwargs)
        try:
            # 長大ログを避けるため先頭200文字だけ
            body_preview = resp.text[:200]
        except Exception:
            body_preview = None
        suffix = "..." if body_preview and len(resp.text) > 200 else ""
        self._log(f"<- {resp.status_code} {url}", (body_preview + suffix) if body_preview is not None else None)
        return resp

    async def _authed_request(self, method: str, url: str, *, retry_on_401: bool = True, **kwargs) -> httpx.Response:
        """認証付きリクエスト（常にBearer必須）。"""
        bearer = await self.ensure_token()
        headers = kwargs.pop("headers", {}) or {}
        headers.update(self._auth_headers(bearer))
        resp = await self._request(method, url, headers=headers, **kwargs)
        return resp

    async def aclose(self) -> None:
        await self._http.aclose()

    # --- Auth ---
    async def ensure_token(self) -> str:
        # メモリ内トークン
        if self._token and not self._token.is_expired:
            return self._token.access_token
        # 事前トークン（期限不明）
        if self.config.access_token:
            tok = EpsonToken(self.config.access_token, 365 * 24 * 3600)
            self._token = tok
            return tok.access_token
        # キャッシュトークン
        cached = self._load_cached_token()
        if cached and not cached.is_expired:
            self._token = cached
            return self._token.access_token
        # リフレッシュトークンがあれば更新
        if self.config.refresh_token:
            await self.refresh_access_token()
            if self._token and not self._token.is_expired:
                return self._token.access_token
        # なければ例外
        raise RuntimeError(
            "No Epson access token available (set EPSON_ACCESS_TOKEN, or EPSON_REFRESH_TOKEN(+CLIENT_ID/SECRET), or provide logs/epson_access_token.json)"
        )

    async def refresh_access_token(self) -> None:
        if not self.config.refresh_token:
            raise ValueError("EPSON_REFRESH_TOKEN is required to refresh token")

        url = f"{self.config.auth_url}/auth/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.config.refresh_token,
        }
        # client_id/secret があれば Basic を付与
        if self.config.client_id and self.config.client_secret:
            basic = base64.b64encode(f"{self.config.client_id}:{self.config.client_secret}".encode()).decode()
            headers["Authorization"] = f"Basic {basic}"

        # curlログ用
        curl_parts = [
            "curl -sS",
            "-X POST",
            f"'{url}'",
            "-H 'Content-Type: application/x-www-form-urlencoded'",
        ]
        if "Authorization" in headers:
            curl_parts.append(f"-H 'Authorization: {headers['Authorization']}'")
        form_str = "&".join(f"{k}={v}" for k, v in data.items())
        curl_parts.append(f"-d '{form_str}'")
        curl_cmd = " ".join(curl_parts)

        resp = await self._request("POST", url, headers=headers, data=data)
        resp.raise_for_status()
        payload = resp.json()
        self._token = EpsonToken(payload["access_token"], int(payload.get("expires_in", 3600)))
        # キャッシュへ保存
        self._save_cached_token(self._token)
        # refresh が返却されたら更新
        if "refresh_token" in payload:
            self.config.refresh_token = payload["refresh_token"]

        # 取得した refresh_token と curl をログ化
        try:
            returned_refresh = payload.get("refresh_token")
            log_lines = [
                f"time_iso: {time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime())}",
                f"endpoint: {url}",
                "--- curl ---",
                curl_cmd,
            ]
            if returned_refresh:
                log_lines.extend([
                    "--- returned_refresh_token ---",
                    returned_refresh,
                ])
            content = "\n".join(log_lines) + "\n"
            self._write_timestamped_log("auth_token", content)
        except Exception as e:
            self._log(f"Failed to write auth/token log: {e}")

    # --- Helpers ---
    def _auth_headers(self, bearer: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {bearer}",
            "x-api-key": self.config.api_key,
        }

    # --- API methods ---
    async def get_print_capability_document(self) -> Dict[str, Any]:
        url = f"{self.config.base_url}/printing/capability/document"
        resp = await self._authed_request("GET", url)
        resp.raise_for_status()
        return resp.json()

    async def create_scan_destination(self, alias_name: str, destination_url: str) -> Dict[str, str]:
        url = f"{self.config.base_url}/scanning/destinations"
        body = {
            "aliasName": alias_name,
            "destinationService": "url",
            "destination": destination_url,
        }
        resp = await self._authed_request("POST", url, json=body)
        resp.raise_for_status()
        return resp.json()

    async def create_print_job(
        self,
        job_name: str,
        print_mode: str,
        print_settings: Dict[str, Any],
    ) -> Dict[str, str]:
        url = f"{self.config.base_url}/printing/jobs"
        body = {
            "jobName": job_name,
            "printMode": print_mode,
            "printSettings": print_settings,
        }
        resp = await self._authed_request("POST", url, json=body)
        resp.raise_for_status()
        return resp.json()

    async def upload_job_file(self, upload_uri: str, filename: str, content: bytes, content_type: str) -> None:
        headers = {"content-type": content_type}
        self._log(f"Uploading file to {upload_uri} size={len(content)} bytes")
        resp = await self._request("POST", upload_uri, headers=headers, content=content)
        resp.raise_for_status()

    async def start_print(self, job_id: str) -> Dict[str, Any]:
        url = f"{self.config.base_url}/printing/jobs/{job_id}/print"
        resp = await self._authed_request("POST", url)
        resp.raise_for_status()
        return resp.json() if resp.content else {}
