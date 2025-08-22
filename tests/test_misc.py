async def test_public_test_endpoint(client):
    r = await client.get("/test", params={"tags": ["a", "b"]})
    assert r.status_code == 200
    data = r.json()
    assert data["test"] == "success"
    assert data["tags"] == ["a", "b"]
