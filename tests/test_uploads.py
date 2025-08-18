import io
import datetime


async def test_upload_file_and_list(logged_in_client):
    file_content = b"hello world"
    resp = await logged_in_client.post(
        "/upload/file",
        files={"file": ("sample.txt", io.BytesIO(file_content), "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    file_id = resp.json()["file_id"]

    list_resp = await logged_in_client.get("/files")
    assert list_resp.status_code == 200
    assert file_id in list_resp.json()["file_ids"]

    single = await logged_in_client.get(f"/file/{file_id}")
    assert single.status_code == 200
    assert single.json()["file_id"] == file_id


async def test_upload_history_and_retrieve(logged_in_client):
    resp = await logged_in_client.post("/upload/history", json={"query": "python fastapi"})
    assert resp.status_code == 200
    history_id = resp.json()["history_id"]

    today = datetime.date.today().isoformat()
    hist_resp = await logged_in_client.get(f"/history/{today}")
    assert hist_resp.status_code == 200
    ids = [h["id"] for h in hist_resp.json()["histories"]]
    assert history_id in ids
