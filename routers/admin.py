"""管理者向けAPI（再生成・分析・Epson連携の簡易テスト）。"""

import datetime
import os
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter
from errors import unauthorized, bad_request, not_found, upstream, internal
from pydantic import BaseModel

from database import database
from models import daily_summaries, AdminRegenerateResponse, AnyDictModel, EpsonPrintJobResponse, AnalysisResponse, EvaluateUrlsResponse, FeedbackResponse
from scheduler.summaries import regen_user_day
from services.daily_summary import run_eval_analysis, analyze_day as analyze_service, eval_all_urls, generate_llm_feedback
from services import EpsonClient, EpsonConfig
from utils.datetime_utils import parse_ymd_date
from utils.pdf_utils import render_markdown_to_pdf


class RegenerateRequest(BaseModel):
    password: str
    user_id: int
    date: str  # YYYY-MM-DD
    # 任意: URL評価とスキル分析も実行
    eval_and_analyze: bool = False


router = APIRouter(tags=["admin"])


@router.post("/admin/regenerate", response_model=AdminRegenerateResponse, operation_id="adminRegenerate")
async def admin_regenerate(body: RegenerateRequest) -> AdminRegenerateResponse:
    admin_pass = os.getenv("ADMIN_PASS")
    if not admin_pass or body.password != admin_pass:
        unauthorized("Unauthorized")
    try:
        target_date = parse_ymd_date(body.date)
    except ValueError:
        bad_request("Invalid date format. Use YYYY-MM-DD")
    result = await regen_user_day(body.user_id, target_date)
    extras = {}
    if body.eval_and_analyze:
        extras = await run_eval_analysis(body.user_id, target_date)
    return AdminRegenerateResponse(user_id=body.user_id, date=target_date.isoformat(), **result, **extras)


# --- 機能別エンドポイント ---

class SummaryRequest(BaseModel):
    password: str
    user_id: int
    date: str


@router.post("/admin/summary", response_model=AdminRegenerateResponse, operation_id="adminSummary")
async def admin_summary(body: SummaryRequest) -> AdminRegenerateResponse:
    admin_pass = os.getenv("ADMIN_PASS")
    if not admin_pass or body.password != admin_pass:
        unauthorized("Unauthorized")
    try:
        target_date = parse_ymd_date(body.date)
    except ValueError:
        bad_request("Invalid date format. Use YYYY-MM-DD")
    result = await regen_user_day(body.user_id, target_date)
    return AdminRegenerateResponse(user_id=body.user_id, date=target_date.isoformat(), **result)


class AnalysisRequest(BaseModel):
    password: str
    user_id: int
    date: str


@router.post("/admin/analysis", response_model=AnalysisResponse, operation_id="adminAnalysis")
async def admin_analysis(body: AnalysisRequest) -> AnalysisResponse:
    admin_pass = os.getenv("ADMIN_PASS")
    if not admin_pass or body.password != admin_pass:
        unauthorized("Unauthorized")
    try:
        target_date = parse_ymd_date(body.date)
    except ValueError:
        bad_request("Invalid date format. Use YYYY-MM-DD")
    result = await analyze_service(body.user_id, target_date)
    return AnalysisResponse(user_id=body.user_id, date=target_date.isoformat(), **result)


class EvaluateAllRequest(BaseModel):
    password: str


@router.post("/admin/evaluate-urls", response_model=EvaluateUrlsResponse, operation_id="adminEvaluateUrls")
async def admin_evaluate_all(body: EvaluateAllRequest) -> EvaluateUrlsResponse:
    admin_pass = os.getenv("ADMIN_PASS")
    if not admin_pass or body.password != admin_pass:
        unauthorized("Unauthorized")
    stats = await eval_all_urls()
    return EvaluateUrlsResponse(**stats)


class FeedbackRequest(BaseModel):
    password: str
    user_id: int
    date: str


@router.post("/admin/feedback", response_model=FeedbackResponse, operation_id="adminFeedback")
async def admin_feedback(body: FeedbackRequest) -> FeedbackResponse:
    admin_pass = os.getenv("ADMIN_PASS")
    if not admin_pass or body.password != admin_pass:
        unauthorized("Unauthorized")
    try:
        target_date = parse_ymd_date(body.date)
    except ValueError:
        bad_request("Invalid date format. Use YYYY-MM-DD")
    text = await generate_llm_feedback(body.user_id, target_date)
    return FeedbackResponse(user_id=body.user_id, date=target_date.isoformat(), feedback=text)


# ===== Epson 管理用お試しエンドポイント =====

class EpsonAuthBase(BaseModel):
    password: str
    use_dummy: bool = False

    def build_config(self) -> EpsonConfig:
    # 認証情報は環境変数から取得（ここで変更可能なのは use_dummy のみ）
        return EpsonConfig(use_dummy=self.use_dummy)


class EpsonCapabilityRequest(EpsonAuthBase):
    pass


@router.post("/admin/epson/capability", response_model=AnyDictModel, operation_id="adminEpsonCapability")
async def admin_eps_capability(body: EpsonCapabilityRequest) -> AnyDictModel:
    admin_pass = os.getenv("ADMIN_PASS")
    if not admin_pass or body.password != admin_pass:
        unauthorized("Unauthorized")
    client = EpsonClient(body.build_config())
    try:
        data = await client.get_print_capability_document()
        await client.aclose()
        return AnyDictModel(data)
    except httpx.HTTPStatusError as e:
        # 上流エラー内容をそのまま返して診断しやすくする
        detail = {
            "message": str(e),
            "url": str(getattr(e.request, "url", "")),
            "status_code": e.response.status_code if e.response else None,
            "body": e.response.text if e.response else None,
        }
        await client.aclose()
        upstream(detail)


class EpsonCreateDestinationRequest(EpsonAuthBase):
    alias_name: str = "Viofolio"
    user_id: int


@router.post("/admin/epson/scan-destination", response_model=AnyDictModel, operation_id="adminEpsonCreateDestination")
async def admin_eps_create_destination(body: EpsonCreateDestinationRequest) -> AnyDictModel:
    admin_pass = os.getenv("ADMIN_PASS")
    if not admin_pass or body.password != admin_pass:
        unauthorized("Unauthorized")
    client = EpsonClient(body.build_config())
    try:
        base = "https://tornado2025.chigayuki.com"
        dest_url = base.rstrip("/") + f"/upload/file/{body.user_id}"
        resp = await client.create_scan_destination(body.alias_name, dest_url)
        await client.aclose()
        return AnyDictModel(resp)
    except httpx.HTTPStatusError as e:
        detail = {
            "message": str(e),
            "url": str(getattr(e.request, "url", "")),
            "status_code": e.response.status_code if e.response else None,
            "body": e.response.text if e.response else None,
        }
        await client.aclose()
        upstream(detail)


class EpsonPrintSettings(BaseModel):
    paperSize: str = "ps_a4"
    paperType: str = "pt_plainpaper"
    borderless: bool = False
    printQuality: str = "normal"
    paperSource: str = "auto"
    colorMode: str = "mono"
    doubleSided: str = "none"
    reverseOrder: bool = False
    copies: int = 1
    collate: bool = True


class EpsonPrintFromMarkdownRequest(EpsonAuthBase):
    job_name: str = "daily"
    user_id: int
    date: Optional[str] = None  # YYYY-MM-DD, defaults to today
    filename: str = "summary.pdf"
    print_settings: Optional[Dict[str, Any]] = None


@router.post("/admin/epson/print", response_model=EpsonPrintJobResponse, operation_id="adminEpsonPrintFromMarkdown")
async def admin_eps_print_from_md(body: EpsonPrintFromMarkdownRequest) -> EpsonPrintJobResponse:
    admin_pass = os.getenv("ADMIN_PASS")
    if not admin_pass or body.password != admin_pass:
        unauthorized("Unauthorized")

    # 指定ユーザー/日のMarkdownをdaily_summariesから取得
    if body.date:
        try:
            target_date = parse_ymd_date(body.date)
        except ValueError:
            bad_request("Invalid date format. Use YYYY-MM-DD")
    else:
        target_date = datetime.date.today()
    row = await database.fetch_one(
        daily_summaries.select().where(
            (daily_summaries.c.user_id == body.user_id) & (daily_summaries.c.date == target_date)
        )
    )
    if not row:
        not_found("Daily summary not found for specified date")
    md_text = row["markdown"]

    # Markdown→PDF変換
    pdf_bytes = render_markdown_to_pdf(md_text, title=f"Summary {target_date.isoformat()}")

    # EPSONジョブ作成
    settings = body.print_settings or EpsonPrintSettings().model_dump()
    client = EpsonClient(body.build_config())
    try:
        job = await client.create_print_job(
            job_name=body.job_name,
            print_mode="document",
            print_settings=settings,
        )

        await client.upload_job_file(
            upload_uri= f'{job["uploadUri"]}&File=1.pdf',
            filename=body.filename,
            content=pdf_bytes,
            content_type="application/pdf",
        )

        printed = await client.start_print(job_id=job["jobId"])
        printed = bool(printed)
        return EpsonPrintJobResponse(job=job, printed=printed)
    except httpx.HTTPStatusError as e:
        detail = {
            "message": str(e),
            "url": str(getattr(e.request, "url", "")),
            "status_code": e.response.status_code if e.response else None,
            "body": e.response.text if e.response else None,
        }
        upstream(detail)
    finally:
        await client.aclose()
