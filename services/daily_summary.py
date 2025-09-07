"""日次要約と分析サービス（入力収集→要約→タグ抽出→保存）。"""

from __future__ import annotations

import datetime
import os
import re
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

from database import database
from models import (
    daily_summaries,
    daily_summary_tags,
    daily_feedback,
    file_summaries,
    files,
    search_histories,
    tags,
    url_evaluations,
    user_skill_features,
)
from utils.datetime_utils import day_range, naive_utc_now

from python.summary_function.cli import (
    summarize_multiple_inputs,
    extract_categorical_keywords,
    aggregate_category_summary,
)

# Analysis (skill/feedback) imports – do not modify python/ code
import pandas as pd
from python.analyzer_function.skillviz_ml import io as skill_io
from python.analyzer_function.skillviz_ml import llm as skill_llm
from python.analyzer_function.skillviz_ml import scoring as skill_scoring
from python.analyzer_function.skillviz_ml import analyzer as skill_analyzer


async def _ensure_file_summaries_for_user_date(user_id: int, start_dt: datetime.datetime, end_dt: datetime.datetime) -> int:
    """対象期間に要約のないファイルへプレースホルダーを作成。戻り値は作成数。"""
    created_file = 0
    f_query = (
        files.select()
        .with_only_columns(files.c.id)
        .where((files.c.user_id == user_id) & (files.c.created_at >= start_dt) & (files.c.created_at <= end_dt))
    )
    file_rows = await database.fetch_all(f_query)
    for row in file_rows:
        file_id = row[0]
        exists = await database.fetch_one(
            file_summaries.select().with_only_columns(file_summaries.c.id).where(file_summaries.c.file_id == file_id)
        )
        if exists:
            continue
        await database.execute(
            file_summaries.insert().values(
                file_id=file_id,
                summary=f"Auto summary for file {file_id}",
                created_at=naive_utc_now(),
            )
        )
        created_file += 1
    return created_file


async def _collect_sources(user_id: int, start_dt: datetime.datetime, end_dt: datetime.datetime) -> list[str]:
    """対象期間のURLとローカルファイルパスを収集。"""
    sources: list[str] = []

    # ローカルファイル
    fq2 = (
        files.select()
        .with_only_columns(files.c.file_path)
        .where((files.c.user_id == user_id) & (files.c.created_at >= start_dt) & (files.c.created_at <= end_dt))
    )
    fp_rows = await database.fetch_all(fq2)
    for r in fp_rows:
        fp = r[0]
        if fp and os.path.exists(fp):
            sources.append(fp)

    # 検索履歴（URLを優先。無い場合は説明文を一時ファイル化して入力にする）
    hq2 = (
        search_histories.select()
        .with_only_columns(
            search_histories.c.url,
            search_histories.c.description,
        )
        .where(
            (search_histories.c.user_id == user_id)
            & (search_histories.c.created_at >= start_dt)
            & (search_histories.c.created_at <= end_dt)
        )
    )
    hrows = await database.fetch_all(hq2)
    tmpdir = Path(".tmp_daily"); tmpdir.mkdir(exist_ok=True)
    for r in hrows:
        u = r[0]
        desc = r[1]
        if u:
            sources.append(u)
        elif desc:
            name = f"history_{user_id}_{start_dt.date().isoformat()}_{len(sources)+1}.txt"
            path = tmpdir / name
            try:
                path.write_text(desc, encoding="utf-8")
                sources.append(str(path))
            except OSError:
                # この履歴だけスキップ（一時ファイル書き込みに失敗しても全体は継続）
                pass

    return sources


def _parse_category_sections(aggregated_summary: str) -> dict[str, str]:
    """Markdownの「### 見出し」をカテゴリとして分割。無ければ『全体』とする。"""
    sections: dict[str, str] = {}
    current_cat: str | None = None
    current_buf: list[str] = []
    for line in (aggregated_summary or "").splitlines():
        m = re.match(r"^###\s+(.+)$", line.strip())
        if m:
            if current_cat is not None:
                sections[current_cat] = "\n".join(current_buf).strip()
            current_cat = m.group(1).strip()
            current_buf = []
        else:
            current_buf.append(line)
    if current_cat is not None:
        sections[current_cat] = "\n".join(current_buf).strip()
    if not sections:
        sections = {"全体": aggregated_summary}
    return sections


def _refine_sections_with_llm(sections: dict[str, str], api_key: str, model: str) -> str:
    """各カテゴリをLLMで再要約して結合。"""
    parts: list[str] = []
    for cat, text in sections.items():
        if not text:
            continue
        refined = aggregate_category_summary(category=cat, summaries=[text], api_key=api_key, model=model)
        if refined:
            parts.append(refined)
    return "\n\n".join(parts)


async def _upsert_daily_markdown(user_id: int, date: datetime.date, markdown: str) -> None:
    exists_ds = await database.fetch_one(
        daily_summaries.select().with_only_columns(daily_summaries.c.id).where(
            (daily_summaries.c.user_id == user_id) & (daily_summaries.c.date == date)
        )
    )
    if exists_ds:
        await database.execute(
            daily_summaries.update()
            .where((daily_summaries.c.user_id == user_id) & (daily_summaries.c.date == date))
            .values(markdown=markdown)
        )
    else:
        await database.execute(
            daily_summaries.insert().values(
                user_id=user_id,
                date=date,
                markdown=markdown,
                created_at=naive_utc_now(),
            )
        )


async def _extract_and_upsert_tags(user_id: int, date: datetime.date, aggregated_summary: str, api_key: str, model: str) -> int:
    """カテゴリ/キーワードからタグを抽出し、日次要約へ紐付け。件数を返す。"""
    mapping: dict[str, list[str]] = extract_categorical_keywords(aggregated_summary, api_key=api_key, model=model) or {}
    names: set[str] = set()
    for cat, kws in mapping.items():
        cat_name = str(cat).strip()
        if cat_name:
            names.add(cat_name)
        for kw in (kws or []):
            kname = str(kw).strip()
            if kname:
                names.add(kname)

    if not names:
        return 0

    # upsert tags
    tag_ids: list[int] = []
    for tname in sorted(names):
        existing = await database.fetch_one(
            tags.select().with_only_columns(tags.c.id).where((tags.c.user_id == user_id) & (tags.c.name == tname))
        )
        if existing:
            tag_id = existing[0]
        else:
            tag_id = await database.execute(tags.insert().values(user_id=user_id, name=tname))
        tag_ids.append(tag_id)

    # link to daily summary
    ds_row = await database.fetch_one(
        daily_summaries.select()
        .with_only_columns(daily_summaries.c.id)
        .where((daily_summaries.c.user_id == user_id) & (daily_summaries.c.date == date))
    )
    if not ds_row:
        return len(tag_ids)
    ds_id = ds_row[0]
    for tag_id in tag_ids:
        exists_link = await database.fetch_one(
            daily_summary_tags.select()
            .with_only_columns(daily_summary_tags.c.daily_summary_id)
            .where((daily_summary_tags.c.daily_summary_id == ds_id) & (daily_summary_tags.c.tag_id == tag_id))
        )
        if not exists_link:
            await database.execute(
                daily_summary_tags.insert().values(daily_summary_id=ds_id, tag_id=tag_id)
            )
    return len(tag_ids)


async def generate_summary(user_id: int, target_date: datetime.date) -> dict[str, Any]:
    """指定日の日次要約を生成/更新（タグ抽出含む）。"""
    start_dt, end_dt = day_range(target_date)

    # 1) 最低限のファイル要約（プレースホルダー）を作成
    created_file = await _ensure_file_summaries_for_user_date(user_id, start_dt, end_dt)

    # 2) 入力ソースを収集
    sources = await _collect_sources(user_id, start_dt, end_dt)
    if not sources:
    # 入力が無い場合は空の要約だけ作成して終了
        await _upsert_daily_markdown(
            user_id,
            target_date,
            f"# {target_date.isoformat()} のまとめ\n\n(データがありません)",
        )
        return {"file_summaries": created_file, "history_summaries": 0, "daily": True, "tags": 0, "links": 0}

    # 3) カテゴリ別に集約要約を作成
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    # 本番では冗長なため出力しない
    aggregated_summary: str = await summarize_multiple_inputs(
        srcs=sources,
        api_key=api_key,
        model=model,
        max_chars=3500,
        ocr_lang=os.getenv("OCR_LANG", "jpn+eng"),
        aggregate=True,
        google_credentials_path=google_credentials_path,
    )

    # 4) aggregate_category_summary で各カテゴリを簡潔に再要約
    sections = _parse_category_sections(aggregated_summary)
    refined_md = _refine_sections_with_llm(sections, api_key=api_key, model=model)

    # 5) 日次Markdownを保存
    await _upsert_daily_markdown(user_id, target_date, refined_md or aggregated_summary)

    # 6) 集約テキストからタグを抽出し、日次要約に紐付け
    tag_count = await _extract_and_upsert_tags(user_id, target_date, aggregated_summary, api_key=api_key, model=model)

    return {
        "file_summaries": created_file,
        "history_summaries": 0,
        "daily": True,
        "tags": tag_count,
        "links": 0,
    }


# =====================
# 分析ユーティリティ群
# =====================

def _rules_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "python" / "analyzer_function" / "config" / "rules.yml"


def _load_rules_safe() -> dict:
    """rules.yml を読み込む。存在しない/不正な場合は空の辞書を返す。"""
    try:
        return skill_io.load_config(_rules_path())
    except Exception:
        return {}


async def _load_user_logs_df(user_id: int, start_dt: datetime.datetime | None = None, end_dt: datetime.datetime | None = None) -> pd.DataFrame:
    """ユーザの閲覧ログをDFへ整形（必要列: user_id,url,timestamp,domain,dwell_sec,visit_count）。"""
    q = (
        search_histories.select()
        .with_only_columns(
            search_histories.c.user_id,
            search_histories.c.url,
            search_histories.c.title,
            search_histories.c.description,
            search_histories.c.created_at,
        )
        .where(search_histories.c.user_id == user_id)
    )
    if start_dt is not None:
        q = q.where(search_histories.c.created_at >= start_dt)
    if end_dt is not None:
        q = q.where(search_histories.c.created_at <= end_dt)

    rows = await database.fetch_all(q)
    if not rows:
        return pd.DataFrame(columns=["user_id", "url", "timestamp", "domain", "dwell_sec", "visit_count"])  # empty

    data = []
    for r in rows:
        u = r[1]
        created = r[4]
        domain = urlparse(u).netloc if isinstance(u, str) else None
        data.append({
            "user_id": r[0],
            "url": u,
            "title": r[2],
            "description": r[3],
            "timestamp": created,
            "domain": domain,
        # エンゲージメント計算で NaN を避けるためのデフォルト値
            "dwell_sec": 60.0 if u else 0.0,
            "visit_count": 1,
        })
    df = pd.DataFrame(data)
    # 型の整合性をとる
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    return df


def _avg_difficulty_from_evaluations(df: pd.DataFrame) -> float | None:
    if df.empty or "llm_evaluation" not in df.columns:
        return None
    diffs = df["llm_evaluation"].dropna().apply(lambda d: (d or {}).get("difficulty_score"))
    s = pd.to_numeric(diffs, errors="coerce")
    val = s.mean()
    return float(val) if pd.notna(val) else None


def _top_categories_from_evaluations(df: pd.DataFrame, top_n: int = 3) -> list[str]:
    if df.empty or "llm_evaluation" not in df.columns:
        return []
    cats: list[str] = []
    for item in df["llm_evaluation"].dropna():
        try:
            for c in (item or {}).get("skill_categories", []) or []:
                if isinstance(c, str):
                    cats.append(c)
        except Exception:
            continue
    if not cats:
        return []
    vc = pd.Series(cats).value_counts()
    return vc.index.tolist()[:top_n]


async def eval_urls(user_id: int, start_date: datetime.date | None = None, end_date: datetime.date | None = None) -> pd.DataFrame:
    """Evaluate browsing at the domain level and return a DataFrame with `llm_evaluation` per domain.

    Notes:
    - Caching is done per (user, domain), stored in `url_evaluations.url` as the domain string.
    - Within a batch, we deduplicate by domain and evaluate one representative row per domain.
    """
    start_dt = datetime.datetime.combine(start_date, datetime.time.min) if start_date else None
    end_dt = datetime.datetime.combine(end_date, datetime.time.max) if end_date else None
    df_logs = await _load_user_logs_df(user_id, start_dt, end_dt)
    if df_logs.empty:
        return df_logs

    # 1) Try to hydrate from cached url_evaluations (domain-level)
    domains = df_logs["domain"].dropna().unique().tolist()
    cached = await database.fetch_all(
        url_evaluations.select()
        .with_only_columns(
            url_evaluations.c.url,  # stores domain string
            url_evaluations.c.difficulty_score,
            url_evaluations.c.specialization_level,
            url_evaluations.c.skill_categories,
        )
        .where(url_evaluations.c.url.in_(domains))
    )
    cache_map = {
        row[0]: {
            "difficulty_score": row[1],
            "specialization_level": row[2],
            "skill_categories": row[3],
        }
        for row in cached
    }
    df_logs["llm_evaluation"] = df_logs["domain"].map(cache_map)

    # 2) Evaluate only missing domains
    to_eval_domains = [d for d in domains if d not in cache_map]
    if to_eval_domains:
        sub_df = df_logs[df_logs["domain"].isin(to_eval_domains)].copy()
        # choose one representative row per domain
        sub_df = sub_df.drop_duplicates(subset=["domain"]).copy()
        rules = _load_rules_safe()
        evaluated_df = skill_llm.evaluate_pages_with_llm(sub_df, rules)
        # Persist evaluations to cache table
        for _, row in evaluated_df.iterrows():
            url = row.get("url")
            ev = row.get("llm_evaluation") or {}
            if not url or not isinstance(ev, dict):
                continue
            domain_key = row.get("domain") or (urlparse(url).netloc if isinstance(url, str) else None)
            if not domain_key:
                continue
            # upsert
            existing = await database.fetch_one(
                url_evaluations.select()
                .with_only_columns(url_evaluations.c.id)
                .where(url_evaluations.c.url == domain_key)
            )
            values = {
                "url": domain_key,  # store domain in cache
                "difficulty_score": ev.get("difficulty_score"),
                "specialization_level": ev.get("specialization_level"),
                "skill_categories": ev.get("skill_categories"),
                "evaluated_at": naive_utc_now(),
            }
            if existing:
                await database.execute(url_evaluations.update().where(url_evaluations.c.id == existing[0]).values(**values))
            else:
                await database.execute(url_evaluations.insert().values(**values))
            cache_map[domain_key] = {
                "difficulty_score": values["difficulty_score"],
                "specialization_level": values["specialization_level"],
                "skill_categories": values["skill_categories"],
            }

        # merge back by domain
        df_logs.loc[df_logs["domain"].isin(to_eval_domains), "llm_evaluation"] = (
            df_logs.loc[df_logs["domain"].isin(to_eval_domains), "domain"].map(cache_map)
        )
    # Return df_logs with llm_evaluation hydrated/cached
    return df_logs


async def calc_skills(user_id: int, start_date: datetime.date | None = None, end_date: datetime.date | None = None) -> pd.DataFrame:
    """Run evaluation and compute skill features for the user in the window."""
    df_eval = await eval_urls(user_id, start_date, end_date)
    rules = _load_rules_safe()
    # Union existing user tags as additional skill hints
    tag_rows = await database.fetch_all(
        tags.select().with_only_columns(tags.c.name).where(tags.c.user_id == user_id)
    )
    extra_skills = {r[0].lower().replace(" ", "_") for r in tag_rows} if tag_rows else set()
    if not df_eval.empty:
        # augment skill_categories with tags
        def _augment(ev):
            if not isinstance(ev, dict):
                return ev
            skills = set(ev.get("skill_categories") or [])
            skills.update(extra_skills)
            ev["skill_categories"] = sorted(skills)
            return ev
        df_eval["llm_evaluation"] = df_eval["llm_evaluation"].apply(_augment)

    df_features = skill_scoring.extract_features(df_eval, rules, llm_enabled=True)

    # Persist aggregated features for quick retrieval
    for _, row in df_features.iterrows():
        values = {
            "user_id": user_id,
            "skill": row.get("skill"),
            "heuristic_score_sum": float(row.get("heuristic_score_sum", 0) or 0),
            "mean_difficulty": float(row.get("mean_difficulty", 0) or 0),
            "n_pages": int(row.get("n_pages", 0) or 0),
            "mean_engagement": float(row.get("mean_engagement", 0) or 0),
            "updated_at": naive_utc_now(),
        }
        if not values["skill"]:
            continue
        existing = await database.fetch_one(
            user_skill_features.select().with_only_columns(user_skill_features.c.id).where(
                (user_skill_features.c.user_id == user_id) & (user_skill_features.c.skill == values["skill"])
            )
        )
        if existing:
            await database.execute(user_skill_features.update().where(user_skill_features.c.id == existing[0]).values(**values))
        else:
            await database.execute(user_skill_features.insert().values(**values))
    return df_features


async def gen_feedback(user_id: int, start_date: datetime.date | None = None, end_date: datetime.date | None = None) -> str:
    """Generate feedback text for the user's activity using the analyzer's LLM helper."""
    df_features = await calc_skills(user_id, start_date, end_date)
    return skill_analyzer.generate_llm_feedback(df_features)


async def analyze_day(user_id: int, target_date: datetime.date) -> dict[str, Any]:
    """Compute analysis metrics for a date using cached data only (no LLM calls):

    Returns:
    {
      feedback: str,  # composed from stored features (non-LLM)
      url_count: int,
      avg_difficulty: { today: float|None, yesterday: float|None },
      top_categories: [str, ...]
    }
    """
    # Today window
    start_dt, end_dt = day_range(target_date)
    # Yesterday window
    prev_date = target_date - datetime.timedelta(days=1)
    prev_start, prev_end = day_range(prev_date)

    # Load logs only
    df_today_logs = await _load_user_logs_df(user_id, start_dt, end_dt)
    df_y_logs = await _load_user_logs_df(user_id, prev_start, prev_end)

    # Hydrate evaluations from cache only (domain-level); do not evaluate missing

    # Because we are in async def, implement the cache hydration inline
    async def _hydrate(df_logs: pd.DataFrame) -> pd.DataFrame:
        if df_logs.empty:
            return df_logs
        domains = df_logs["domain"].dropna().unique().tolist()
        if not domains:
            df_logs["llm_evaluation"] = None
            return df_logs
        cached = await database.fetch_all(
            url_evaluations.select()
            .with_only_columns(
                url_evaluations.c.url,
                url_evaluations.c.difficulty_score,
                url_evaluations.c.specialization_level,
                url_evaluations.c.skill_categories,
            )
            .where(url_evaluations.c.url.in_(domains))
        )
        cache_map = {r[0]: {"difficulty_score": r[1], "specialization_level": r[2], "skill_categories": r[3]} for r in cached}
        df_logs = df_logs.copy()
        df_logs["llm_evaluation"] = df_logs["domain"].map(cache_map)
        return df_logs

    df_today = await _hydrate(df_today_logs)
    df_yesterday = await _hydrate(df_y_logs)

    # Metrics
    url_count = int(df_today["url"].dropna().nunique()) if not df_today.empty else 0
    avg_today = _avg_difficulty_from_evaluations(df_today)
    avg_yesterday = _avg_difficulty_from_evaluations(df_yesterday)
    top3 = _top_categories_from_evaluations(df_today, top_n=3)

    # Compose feedback from stored features (non-LLM)
    # Use top skills by heuristic_score_sum and n_pages
    feat_rows = await database.fetch_all(
        user_skill_features.select()
        .with_only_columns(
            user_skill_features.c.skill,
            user_skill_features.c.heuristic_score_sum,
            user_skill_features.c.mean_difficulty,
            user_skill_features.c.n_pages,
        )
        .where(user_skill_features.c.user_id == user_id)
    )
    if feat_rows:
        data = [[r[0], r[1], r[2], r[3]] for r in feat_rows]
        feats = pd.DataFrame(data, columns=["skill", "heuristic_score_sum", "mean_difficulty", "n_pages"]).fillna(0)
        # Coerce numeric columns
        feats["heuristic_score_sum"] = pd.to_numeric(feats["heuristic_score_sum"], errors="coerce").fillna(0.0)
        feats["mean_difficulty"] = pd.to_numeric(feats["mean_difficulty"], errors="coerce").fillna(0.0)
        feats["n_pages"] = pd.to_numeric(feats["n_pages"], errors="coerce").fillna(0).astype(int)
        feats = feats.sort_values(["heuristic_score_sum", "n_pages"], ascending=[False, False])
        top_skills = feats["skill"].astype(str).head(3).tolist()
        mdiff = pd.to_numeric(feats["mean_difficulty"], errors="coerce").replace(0, pd.NA).dropna().mean()
        feedback = (
            f"最近は『{', '.join(top_skills)}』の学習・調査が目立ちます。"
            + (f" 平均難易度はおよそ{mdiff:.2f}です。" if pd.notna(mdiff) else "")
        ) if top_skills else "最近のアクティビティデータが十分ではありません。"
    else:
        feedback = "最近のアクティビティデータが十分ではありません。"

    # Upsert feedback for the day
    existing_fb = await database.fetch_one(
        daily_feedback.select().with_only_columns(daily_feedback.c.id).where(
            (daily_feedback.c.user_id == user_id) & (daily_feedback.c.date == target_date)
        )
    )
    if existing_fb:
        await database.execute(
            daily_feedback.update()
            .where((daily_feedback.c.user_id == user_id) & (daily_feedback.c.date == target_date))
            .values(feedback=feedback)
        )
    else:
        await database.execute(
            daily_feedback.insert().values(
                user_id=user_id,
                date=target_date,
                feedback=feedback,
                created_at=naive_utc_now(),
            )
        )

    return {
        "feedback": feedback,
        "url_count": url_count,
        "avg_difficulty": {"today": avg_today, "yesterday": avg_yesterday},
        "top_categories": top3,
    }


# ==============================
# Decoupled daily jobs (requested)
# ==============================

async def evaluate_user_pages(user_id: int, target_date: datetime.date) -> dict[str, Any]:
    """Evaluate and cache missing domain-level evaluations for a user on a given date.

    Returns: { evaluated_new: int, domains: int }
    """
    start_dt, end_dt = day_range(target_date)
    df_logs = await _load_user_logs_df(user_id, start_dt, end_dt)
    if df_logs.empty:
        return {"evaluated_new": 0, "domains": 0}
    todays_domains = df_logs["domain"].dropna().unique().tolist()
    if not todays_domains:
        return {"evaluated_new": 0, "domains": 0}
    before_cached = await database.fetch_all(
        url_evaluations.select().with_only_columns(url_evaluations.c.url).where(url_evaluations.c.url.in_(todays_domains))
    )
    before = {r[0] for r in before_cached}
    # Use existing evaluator which will only evaluate missing domains
    await eval_urls(user_id, target_date, target_date)
    after_cached = await database.fetch_all(
        url_evaluations.select().with_only_columns(url_evaluations.c.url).where(url_evaluations.c.url.in_(todays_domains))
    )
    after = {r[0] for r in after_cached}
    return {"evaluated_new": max(0, len(after) - len(before)), "domains": len(todays_domains)}


async def analyze_user_skills(user_id: int, start_date: datetime.date | None = None, end_date: datetime.date | None = None) -> dict[str, Any]:
    """Compute and persist user skill features using ONLY cached evaluations (no LLM).

    This scans the user's logs in the given window (or all-time if None), hydrates
    domain evaluations from the cache, and extracts features with llm_enabled=False.
    Returns a small stat payload.
    """
    start_dt = datetime.datetime.combine(start_date, datetime.time.min) if start_date else None
    end_dt = datetime.datetime.combine(end_date, datetime.time.max) if end_date else None
    df_logs = await _load_user_logs_df(user_id, start_dt, end_dt)
    if df_logs.empty:
        return {"features_upserted": 0}

    # Hydrate cache only
    domains = df_logs["domain"].dropna().unique().tolist()
    cache_map: dict[str, dict] = {}
    if domains:
        cached = await database.fetch_all(
            url_evaluations.select()
            .with_only_columns(
                url_evaluations.c.url,
                url_evaluations.c.difficulty_score,
                url_evaluations.c.specialization_level,
                url_evaluations.c.skill_categories,
            )
            .where(url_evaluations.c.url.in_(domains))
        )
        cache_map = {r[0]: {"difficulty_score": r[1], "specialization_level": r[2], "skill_categories": r[3]} for r in cached}
    df_eval = df_logs.copy()
    df_eval["llm_evaluation"] = df_eval["domain"].map(cache_map)

    # Ensure required columns for non-LLM scoring path
    # 1) page_difficulty_llm can be hydrated from cached llm_evaluation if present
    def _get_diff(ev):
        try:
            return (ev or {}).get("difficulty_score") if isinstance(ev, dict) else None
        except Exception:
            return None
    df_eval["page_difficulty_llm"] = df_eval["llm_evaluation"].apply(_get_diff)

    # 2) extracted_terms: derive from title + description if available; else empty
    def _simple_terms(title: Any, desc: Any) -> str:
        text = f"{title or ''} {desc or ''}"
        # Keep word-like tokens including Japanese ranges; lowercase ascii
        tokens = re.findall(r"[A-Za-z0-9\u3040-\u30ff\u4e00-\u9fff]+", text)
        tokens = [t.lower() for t in tokens if isinstance(t, str) and len(t) >= 2]
        # Deduplicate preserving order
        seen: set[str] = set(); out: list[str] = []
        for t in tokens:
            if t in seen:
                continue
            seen.add(t); out.append(t)
        return ";".join(out[:50])

    if "extracted_terms" not in df_eval.columns:
        df_eval["extracted_terms"] = [
            _simple_terms(t, d) for t, d in zip(df_eval.get("title", []), df_eval.get("description", []))
        ]

    # Load rules and extract features without LLM
    rules = _load_rules_safe()
    df_features = skill_scoring.extract_features(df_eval, rules, llm_enabled=False)

    # Persist aggregated features
    upserted = 0
    for _, row in df_features.iterrows():
        values = {
            "user_id": user_id,
            "skill": row.get("skill"),
            "heuristic_score_sum": float(row.get("heuristic_score_sum", 0) or 0),
            "mean_difficulty": float(row.get("mean_difficulty", 0) or 0),
            "n_pages": int(row.get("n_pages", 0) or 0),
            "mean_engagement": float(row.get("mean_engagement", 0) or 0),
            "updated_at": naive_utc_now(),
        }
        if not values["skill"]:
            continue
        existing = await database.fetch_one(
            user_skill_features.select().with_only_columns(user_skill_features.c.id).where(
                (user_skill_features.c.user_id == user_id) & (user_skill_features.c.skill == values["skill"])
            )
        )
        if existing:
            await database.execute(user_skill_features.update().where(user_skill_features.c.id == existing[0]).values(**values))
        else:
            await database.execute(user_skill_features.insert().values(**values))
        upserted += 1
    return {"features_upserted": upserted}


async def generate_llm_feedback(user_id: int, target_date: datetime.date) -> str:
    """保存済みの特徴量からLLMフィードバックを生成し、指定日に保存する（再計算なし）。"""
    # テーブルから特徴量を読み込む
    rows = await database.fetch_all(
        user_skill_features.select()
        .with_only_columns(
            user_skill_features.c.skill,
            user_skill_features.c.heuristic_score_sum,
            user_skill_features.c.mean_difficulty,
            user_skill_features.c.n_pages,
            user_skill_features.c.mean_engagement,
        )
        .where(user_skill_features.c.user_id == user_id)
    )
    if rows:
    # pandas が Record を誤って扱わないよう生のリストへ変換
        data = [[r[0], r[1], r[2], r[3], r[4]] for r in rows]
        df_features = pd.DataFrame(
            data,
            columns=[
                "skill",
                "heuristic_score_sum",
                "mean_difficulty",
                "n_pages",
                "mean_engagement",
            ],
        )
    else:
        df_features = pd.DataFrame(columns=["skill", "heuristic_score_sum", "mean_difficulty", "n_pages", "mean_engagement"])  # empty

    # アナライザー向けに型を揃える
    if not df_features.empty:
        # skill ラベルが無い行を除外
        df_features = df_features[df_features["skill"].notna()].copy()
        df_features["skill"] = df_features["skill"].astype(str)
        for col in ["heuristic_score_sum", "mean_difficulty", "mean_engagement"]:
            df_features[col] = pd.to_numeric(df_features[col], errors="coerce").fillna(0.0)
        df_features["n_pages"] = pd.to_numeric(df_features["n_pages"], errors="coerce").fillna(0).astype(int)
    # No debug prints in production path

    feedback_text = skill_analyzer.generate_llm_feedback(df_features)

    # daily_feedback へ upsert
    existing_fb = await database.fetch_one(
        daily_feedback.select().with_only_columns(daily_feedback.c.id).where(
            (daily_feedback.c.user_id == user_id) & (daily_feedback.c.date == target_date)
        )
    )
    if existing_fb:
        await database.execute(
            daily_feedback.update()
            .where((daily_feedback.c.user_id == user_id) & (daily_feedback.c.date == target_date))
            .values(feedback=feedback_text)
        )
    else:
        await database.execute(
            daily_feedback.insert().values(
                user_id=user_id,
                date=target_date,
                feedback=feedback_text,
                created_at=naive_utc_now(),
            )
        )
    return feedback_text


async def run_eval_analysis(user_id: int, target_date: datetime.date) -> dict[str, Any]:
    """管理API/スケジューラ用の一括処理（評価→特徴量→FB）。"""
    # 1) 当日のドメイン評価（新規キャッシュ件数を算出）
    start_dt = datetime.datetime.combine(target_date, datetime.time.min)
    end_dt = datetime.datetime.combine(target_date, datetime.time.max)
    df_today_logs = await _load_user_logs_df(user_id, start_dt, end_dt)
    if df_today_logs.empty:
        evaluated_new = 0
    else:
        todays_domains = df_today_logs["domain"].dropna().unique().tolist()
        before_cached = await database.fetch_all(
            url_evaluations.select().with_only_columns(url_evaluations.c.url).where(url_evaluations.c.url.in_(todays_domains))
        )
        before_set = {r[0] for r in before_cached}
        await eval_urls(user_id, target_date, target_date)
        after_cached = await database.fetch_all(
            url_evaluations.select().with_only_columns(url_evaluations.c.url).where(url_evaluations.c.url.in_(todays_domains))
        )
        after_set = {r[0] for r in after_cached}
        evaluated_new = max(0, len(after_set) - len(before_set))

    # 2) Recompute features (all-time)
    before_rows = await database.fetch_all(
        user_skill_features.select().with_only_columns(user_skill_features.c.id).where(user_skill_features.c.user_id == user_id)
    )
    before_count = len(before_rows)
    df_features = await calc_skills(user_id)
    after_rows = await database.fetch_all(
        user_skill_features.select().with_only_columns(user_skill_features.c.id).where(user_skill_features.c.user_id == user_id)
    )
    after_count = len(after_rows)
    upserted = max(0, after_count - before_count)

    # 3) Feedback
    feedback_text = await gen_feedback(user_id)

    return {
        "url_evaluated_new": evaluated_new,
        "skills_upserted": upserted,
        "feedback": feedback_text,
    }


async def eval_all_urls() -> dict[str, Any]:
    """Evaluate domains across all users for entries in search_histories without a cached evaluation.

    Returns: { evaluated: int, users: int, by_user: {user_id: count} }
    """
    # Get all (user_id, domain) pairs
    rows = await database.fetch_all(
        search_histories.select().with_only_columns(search_histories.c.user_id, search_histories.c.url)
    )
    by_user_domains: dict[int, set[str]] = {}
    for r in rows:
        uid = r["user_id"]
        url = r["url"]
        if not url:
            continue
        dom = urlparse(url).netloc
        if not dom:
            continue
        by_user_domains.setdefault(uid, set()).add(dom)

    total = 0
    stats: dict[int, int] = {}
    rules = _load_rules_safe()

    for uid, domains in by_user_domains.items():
        if not domains:
            continue
        # remove existing cached (domain in url column)
        cached = await database.fetch_all(
            url_evaluations.select().with_only_columns(url_evaluations.c.url).where(
                url_evaluations.c.url.in_(list(domains))
            )
        )
        cached_set = {r[0] for r in cached}
        to_eval = [d for d in domains if d not in cached_set]
        if not to_eval:
            continue

        # Build small DF (one synthetic URL per domain) and evaluate
        df = pd.DataFrame({"user_id": uid, "domain": to_eval})
        df["timestamp"] = pd.Timestamp.utcnow()
        df["url"] = df["domain"].apply(lambda d: f"https://{d}")
        df["dwell_sec"] = 60.0
        df["visit_count"] = 1
        evaluated_df = skill_llm.evaluate_pages_with_llm(df, rules)

        count_user = 0
        for _, row in evaluated_df.iterrows():
            domain_key = row.get("domain") or (urlparse(row.get("url")).netloc if row.get("url") else None)
            ev = row.get("llm_evaluation") or {}
            if not domain_key or not isinstance(ev, dict):
                continue
            await database.execute(
                url_evaluations.insert().values(
                    url=domain_key,  # store domain in cache
                    difficulty_score=ev.get("difficulty_score"),
                    specialization_level=ev.get("specialization_level"),
                    skill_categories=ev.get("skill_categories"),
                    evaluated_at=naive_utc_now(),
                )
            )
            count_user += 1
        if count_user:
            stats[uid] = count_user
            total += count_user

    return {"evaluated": total, "users": len(stats), "by_user": stats}
