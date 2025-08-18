import uuid


async def test_create_and_login(client):
    unique_email = f"a_{uuid.uuid4().hex[:8]}@example.com"
    resp = await client.post("/create", json={"email": unique_email, "password": "pw123456"})
    assert resp.status_code == 200, resp.text
    resp2 = await client.post("/login", json={"email": unique_email, "password": "pw123456"})
    assert resp2.status_code == 200, resp2.text
    token = resp2.cookies.get("token")
    if token:
        client.cookies.set("token", token)
    profile = await client.get("/profile")
    assert profile.status_code == 200
    assert profile.json()["email"] == unique_email


async def test_duplicate_create(client):
    await client.post("/create", json={"email": "dup@example.com", "password": "pw"})
    resp = await client.post("/create", json={"email": "dup@example.com", "password": "pw"})
    assert resp.status_code == 400
