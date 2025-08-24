import uuid


async def test_create_and_login(client):
    unique_email = f"a_{uuid.uuid4().hex[:8]}@example.com"
    resp = await client.post("/create", json={"email": unique_email, "password": "pw123456"})
    assert resp.status_code == 200, resp.text
    resp2 = await client.post("/login", json={"email": unique_email, "password": "pw123456"})
    assert resp2.status_code == 200, resp2.text
    token = resp2.json().get("access_token")
    assert token
    client.headers["Authorization"] = f"Bearer {token}"
    profile = await client.get("/profile")
    assert profile.status_code == 200
    assert profile.json()["email"] == unique_email


async def test_refresh_and_logout_flow(client):
    email = f"b_{uuid.uuid4().hex[:8]}@example.com"
    r = await client.post("/create", json={"email": email, "password": "pw123456"})
    assert r.status_code == 200
    r2 = await client.post("/login", json={"email": email, "password": "pw123456"})
    assert r2.status_code == 200
    body = r2.json()
    assert "refresh_token" in body
    refresh_token = body["refresh_token"]

    # refresh -> new access + new refresh (rotation)
    r3 = await client.post("/refresh", json={"refresh_token": refresh_token})
    assert r3.status_code == 200
    new_access = r3.json()["access_token"]
    new_refresh = r3.json()["refresh_token"]
    assert new_access and new_refresh

    # old refresh should be revoked
    r4 = await client.post("/refresh", json={"refresh_token": refresh_token})
    assert r4.status_code == 401

    # logout the new refresh
    r5 = await client.post("/logout", json={"refresh_token": new_refresh})
    assert r5.status_code == 200
    r6 = await client.post("/refresh", json={"refresh_token": new_refresh})
    assert r6.status_code == 401


async def test_duplicate_create(client):
    await client.post("/create", json={"email": "dup@example.com", "password": "pw"})
    resp = await client.post("/create", json={"email": "dup@example.com", "password": "pw"})
    assert resp.status_code == 400
