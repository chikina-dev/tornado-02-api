"""日次バッチ（要約・分析・フィードバック）。"""

import datetime
import os
from pathlib import Path

from database import database
from models import users
from services.daily_summary import (
	generate_summary,
	run_eval_analysis,
	analyze_day,
	evaluate_user_pages,
	analyze_user_skills,
	generate_llm_feedback,
)
from utils.datetime_utils import day_range, naive_utc_now


async def regen_user_day(user_id: int, target_date: datetime.date) -> dict:
	"""A single user's要約を生成/更新。"""
	return await generate_summary(user_id, target_date)


async def regen_all(target_date: datetime.date) -> list[dict]:
	"""全ユーザー分の要約を生成。"""
	user_rows = await database.fetch_all(users.select().with_only_columns(users.c.id))
	results = []
	for row in user_rows:
		# Access by index to avoid accidentally getting the column name
		uid = row[0]
		res = await regen_user_day(uid, target_date)
		results.append({"user_id": uid, **res})
	return results


async def analyze_all(target_date: datetime.date) -> list[dict]:
	"""全ユーザーの評価/特徴量更新/分析を実行。"""
	user_rows = await database.fetch_all(users.select().with_only_columns(users.c.id))
	results: list[dict] = []
	for row in user_rows:
		uid = row[0]
		# Decoupled pipeline: (1) eval pages (2) analyze skills (3) cache-only analysis & feedback
		eval_stats = await evaluate_user_pages(uid, target_date)
		skill_stats = await analyze_user_skills(uid)
		analysis = await analyze_day(uid, target_date)
		results.append({"user_id": uid, "eval": eval_stats, "skills": skill_stats, "analysis": analysis})
	return results


async def feedback_all(target_date: datetime.date) -> list[dict]:
	"""全ユーザーのフィードバック文を生成（キャッシュ済み特徴量のみ使用）。"""
	user_rows = await database.fetch_all(users.select().with_only_columns(users.c.id))
	results: list[dict] = []
	for row in user_rows:
		uid = row[0]
		text = await generate_llm_feedback(uid, target_date)
		results.append({"user_id": uid, "feedback": text})
	return results

