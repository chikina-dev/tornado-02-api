import datetime

from database import database
from models import (
	users,
	files,
	search_histories,
	file_summaries,
	daily_summaries,
	tags,
	file_tags,
	search_history_tags,
)
from utils.datetime_utils import naive_utc_now
from pathlib import Path
import os

from python.cli import summarize_multiple_inputs, generate_summary_markdown  # type: ignore
import os


def _day_range(d: datetime.date) -> tuple[datetime.datetime, datetime.datetime]:
	start = datetime.datetime.combine(d, datetime.time.min)
	end = datetime.datetime.combine(d, datetime.time.max)
	return start, end


async def regenerate_for_user_date(user_id: int, target_date: datetime.date) -> dict:
	"""
	Create summaries for the given user's files and search histories on the target date
	if they don't already exist. This is a simple placeholder implementation that
	inserts a stub summary text.

	Returns counts of created summaries.
	"""
	start_dt, end_dt = _day_range(target_date)
	created_file = 0
	created_history = 0

	f_query = (
		files.select()
		.with_only_columns(files.c.id)
		.where((files.c.user_id == user_id) & (files.c.created_at >= start_dt) & (files.c.created_at <= end_dt))
	)
	file_rows = await database.fetch_all(f_query)
	for (file_id,) in file_rows:
		exists = await database.fetch_one(
			file_summaries.select().with_only_columns(file_summaries.c.id).where(file_summaries.c.file_id == file_id)
		)
		if exists:
			continue
		await database.execute(
			file_summaries.insert().values(
				file_id=file_id,
				summary=f"Auto summary for file {file_id} on {target_date.isoformat()}",
				created_at=naive_utc_now(),
			)
		)
		created_file += 1

	sources: list[str] = []
	sources_meta: list[dict] = []  # Keeps (kind: "file"|"history", id)
	fq2 = (
		files.select()
		.with_only_columns(files.c.id, files.c.file_path)
		.where((files.c.user_id == user_id) & (files.c.created_at >= start_dt) & (files.c.created_at <= end_dt))
	)
	frows2 = await database.fetch_all(fq2)
	for fr in frows2:
		fid, fp = fr[0], fr[1]
		if fp and os.path.exists(fp):
			sources.append(fp)
			sources_meta.append({"kind": "file", "id": fid})

	hq2 = (
		search_histories.select()
		.with_only_columns(
			search_histories.c.id,
			search_histories.c.url,
			search_histories.c.title,
			search_histories.c.description,
		)
		.where(
			(search_histories.c.user_id == user_id)
			& (search_histories.c.created_at >= start_dt)
			& (search_histories.c.created_at <= end_dt)
		)
	)
	hrows2 = await database.fetch_all(hq2)
	for r in hrows2:
		hid = r[0]
		url = r[1]
		title = r[2]
		desc = r[3]
		if url:
			sources.append(url)
			sources_meta.append({"kind": "history", "id": hid})
		elif desc:
			tmpdir = Path(".tmp_daily"); tmpdir.mkdir(exist_ok=True)
			name = f"history_{user_id}_{target_date.isoformat()}_{len(sources)+1}.txt"
			path = tmpdir / name
			try:
				path.write_text(desc, encoding="utf-8")
				sources.append(str(path))
				sources_meta.append({"kind": "history", "id": hid})
			except Exception:
				pass

	if not sources:
		exists_ds = await database.fetch_one(
			daily_summaries.select().with_only_columns(daily_summaries.c.id).where(
				(daily_summaries.c.user_id == user_id) & (daily_summaries.c.date == target_date)
			)
		)
		if not exists_ds:
			await database.execute(
				daily_summaries.insert().values(
					user_id=user_id,
					date=target_date,
					markdown=f"# {target_date.isoformat()} のまとめ\n\n(データがありません)",
					created_at=naive_utc_now(),
				)
			)
		return {"file_summaries": created_file, "history_summaries": created_history, "daily": True, "tags": 0}

	api_key = os.getenv("OPENAI_API_KEY", "")
	model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
	google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
	terms, category, source_results = await summarize_multiple_inputs(
		srcs=sources,
		api_key=api_key,
		model=model,
		max_chars=3500,
		ocr_lang=os.getenv("OCR_LANG", "jpn+eng"),
		tesseract_cmd=os.getenv("TESSERACT_CMD"),
		google_credentials_path=google_credentials_path,
		highlight=False,
	)
	md = generate_summary_markdown(source_results)


	exists_ds = await database.fetch_one(
		daily_summaries.select().with_only_columns(daily_summaries.c.id).where(
			(daily_summaries.c.user_id == user_id) & (daily_summaries.c.date == target_date)
		)
	)
	if exists_ds:

		await database.execute(
			daily_summaries.update()
			.where((daily_summaries.c.user_id == user_id) & (daily_summaries.c.date == target_date))
			.values(markdown=md)
		)
	else:
		await database.execute(
			daily_summaries.insert().values(
				user_id=user_id,
				date=target_date,
				markdown=md,
				created_at=naive_utc_now(),
			)
		)

	# Materialize tags for the day (create if not exists), and link to the daily summary
	tag_ids: list[int] = []
	for t in terms:
		tname = str(t).strip()
		if not tname:
			continue
		# get or create tag
		existing = await database.fetch_one(
			tags.select().with_only_columns(tags.c.id).where((tags.c.user_id == user_id) & (tags.c.name == tname))
		)
		if existing:
			tag_id = existing[0]
		else:
			tag_id = await database.execute(tags.insert().values(user_id=user_id, name=tname))
		tag_ids.append(tag_id)

	# Link tags to the daily_summary
	if tag_ids:
		ds_row = await database.fetch_one(
			daily_summaries.select()
			.with_only_columns(daily_summaries.c.id)
			.where((daily_summaries.c.user_id == user_id) & (daily_summaries.c.date == target_date))
		)
		if ds_row:
			ds_id = ds_row[0]
			from models import daily_summary_tags  # local import to avoid circulars
			for tag_id in tag_ids:
				exists_link = await database.fetch_one(
					daily_summary_tags.select()
					.with_only_columns(daily_summary_tags.c.daily_summary_id)
					.where(
						(daily_summary_tags.c.daily_summary_id == ds_id)
						& (daily_summary_tags.c.tag_id == tag_id)
					)
				)
				if not exists_link:
					await database.execute(
						daily_summary_tags.insert().values(daily_summary_id=ds_id, tag_id=tag_id)
					)

	# Additionally, link tags back to each source (file/search_history)
	linked_count = 0
	if source_results:
		for res in source_results:
			try:
				idx_str = str(res.get("id", "")).split("-")[-1]
				orig_idx = int(idx_str) - 1
			except Exception:
				continue
			if orig_idx < 0 or orig_idx >= len(sources_meta):
				continue
			meta = sources_meta[orig_idx]
			res_terms = [str(t).strip() for t in (res.get("terms") or []) if str(t).strip()]
			if not res_terms:
				continue
			# Resolve tag ids for this result
			result_tag_ids: list[int] = []
			for tname in res_terms:
				existing = await database.fetch_one(
					tags.select().with_only_columns(tags.c.id).where((tags.c.user_id == user_id) & (tags.c.name == tname))
				)
				if existing:
					result_tag_ids.append(existing[0])
				else:
					tid = await database.execute(tags.insert().values(user_id=user_id, name=tname))
					result_tag_ids.append(tid)
			if meta["kind"] == "file":
				fid = meta["id"]
				for tid in result_tag_ids:
					exists_link = await database.fetch_one(
						file_tags.select().with_only_columns(file_tags.c.id).where(
							(file_tags.c.file_id == fid) & (file_tags.c.tag_id == tid)
						)
					)
					if not exists_link:
						await database.execute(file_tags.insert().values(file_id=fid, tag_id=tid))
						linked_count += 1
			elif meta["kind"] == "history":
				hid = meta["id"]
				for tid in result_tag_ids:
					exists_link = await database.fetch_one(
						search_history_tags.select().with_only_columns(search_history_tags.c.id).where(
							(search_history_tags.c.search_history_id == hid) & (search_history_tags.c.tag_id == tid)
						)
					)
					if not exists_link:
						await database.execute(
							search_history_tags.insert().values(search_history_id=hid, tag_id=tid)
						)
						linked_count += 1

	return {
		"file_summaries": created_file,
		"history_summaries": created_history,
		"daily": True,
		"tags": len(tag_ids),
		"links": linked_count,
	}


async def regenerate_for_all_users(target_date: datetime.date) -> list[dict]:
	"""Run regeneration for all users for the given date."""
	user_rows = await database.fetch_all(users.select().with_only_columns(users.c.id))
	results = []
	for (uid,) in user_rows:
		res = await regenerate_for_user_date(uid, target_date)
		results.append({"user_id": uid, **res})
	return results

