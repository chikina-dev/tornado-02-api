# EPSON Connect API

本番環境のurlは<https://api.epsonconnect.com/api/2>だが
テストとして<https://dummy-api.epsonconnect.com/api/2>を使う
また今回は動作テストのために既存のEPSON_REFRESH_TOKENを使った検証をする

## 環境変数

以下を .env に設定する（認証は refresh_token で簡易動作確認する想定）

- EPSON_API_KEY: 管理ポータルで発行された API キー（必須）
- EPSON_REFRESH_TOKEN: 既存のリフレッシュトークン（テスト用）
- EPSON_CLIENT_ID: 任意（必要な環境では Basic 認証用）
- EPSON_CLIENT_SECRET: 任意（必要な環境では Basic 認証用）
- EPSON_BASE_URL / EPSON_AUTH_URL / USE_DUMMY は環境変数ではなく、`EpsonConfig` の引数で指定する（下記参照）

## Python クライアント

`services/epson_client.py` に Epson Connect API の最小クライアントを追加。主要機能:

- アクセストークンのリフレッシュ（refresh_token ベース）
- 印刷能力取得 (printing/capability/document)
- スキャン送信先の作成 (scanning/destinations)
- 印刷ジョブ作成 (printing/jobs)
- ファイルアップロード（uploadUri に直接 POST）
- 印刷開始 (printing/jobs/{jobId}/print)

### 使い方（async）

```python
from services.epson_client import EpsonClient, EpsonConfig

# ダミーAPIを使う場合は use_dummy=True を指定
config = EpsonConfig(use_dummy=True)
client = EpsonClient(config)

cap = await client.get_print_capability_document()

dest = await client.create_scan_destination(
  alias_name="Viofolio",
  destination_url="https://tornado2025.chigayuki.com/upload/file/1",
)

job = await client.create_print_job(
  job_name="daily",
  print_mode="document",
  print_settings={
    "paperSize": "ps_a4",
    "paperType": "pt_plainpaper",
    "borderless": False,
    "printQuality": "normal",
    "paperSource": "front2",
    "colorMode": "mono",
    "doubleSided": "none",
    "reverseOrder": False,
    "copies": 1,
    "collate": True,
  },
)

upload_uri = job["uploadUri"]

# PDF バイナリをアップロード
with open("summary.pdf", "rb") as f:
  content = f.read()
await client.upload_job_file(upload_uri, filename="summary.pdf", content=content, content_type="application/pdf")

# 印刷開始
await client.start_print(job_id=job["jobId"])
```


## 認証

```bash
curl -X POST https://auth.epsonconnect.com/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -H 'Authorization: Bearer {access_token}' \
  -d "grant_type=refresh_token&refresh_token={REFRESH_TOKEN}"
```

レスポンス

```json
{
  "access_token": "Sowof7oz47eWiRfYObIDMjZ0DiWatmIxJA3ckE3qTNQ",
  "refresh_token": "{refresh_token}",
  "scope": "device",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

## 印刷能力取得

```bash
curl -x GET "https://api.epsonconnect.com/api/2/printing/capability/document"
  --header 'Authorization: Bearer {access_token}' \
  --header 'x-api-key: {EPSON_API_KEY}' \
```

レスポンス

```json
{
  "colorModes": [
    "color",
    "mono"
  ],
  "resolutions": [
    360,
    720
  ],
  "paperSizes": [
    {
      "paperSize": "ps_a4",
      "paperTypes": [
        {
          "paperType": "pt_plainpaper",
          "borderless": false,
          "paperSources": [
            "rear"
          ],
          "printQualities": [
            "draft",
            "normal"
          ],
          "doubleSided": false
        },
        ...
      ]
    },
    ...
  ]
}
```

## scan

今回はuser_idは1

```bash
curl --request POST \
  --url 'https://api.epsonconnect.com/api/2/scanning/destinations' \
  --header 'Authorization: Bearer {access_token}' \
  --header 'x-api-key: {EPSON_API_KEY}' \
  --header 'content-type: application/json' \
  --data '{"aliasName":"Viofolio","destinationService":"url","destination":"https://tornado2025.chigayuki.com/upload/file/{user_id}"}'
```

レスポンス

```json
{
  "destinationId": "{destinationId}"
}
```

## print job

印刷に関しては以下を参照
<https://docs.epsonconnect.com/jp/api_appendix.html>

```bash
curl --request POST \
  --url 'https://api.epsonconnect.com/api/2/printing/jobs' \
  --header 'Authorization: Bearer {access_token}' \
  --header 'x-api-key: {EPSON_API_KEY}' \
  --header 'content-type: application/json' \
  --data '{"jobName":"daily","printMode":"document","printSettings":{"paperSize":"ps_a4","paperType":"pt_plainpaper","borderless":false,"printQuality":"normal","paperSource":"front2","colorMode":"mono","doubleSided":"none","reverseOrder":false,"copies":1,"collate":true}}'
```

レスポンス

```json
{
  "jobId": "{jobid}",
  "uploadUri": "{upload_url}"
}
```

## ファイルアップロード

```bash
curl --request POST \
  --url '{uploadUri}&File=summary.pdf'
  --header 'content-type: application/pdf' \
  # ... バイナリなので省略
```

レスポンスボディなし

## 印刷実行

```bash
curl --request POST \
  --url 'https://api.epsonconnect.com/api/2/printing/jobs/{jobId}/print' \
  --header 'Authorization: Bearer {access_token}' \
  --header 'x-api-key: {EPSON_API_KEY}' \
```

レスポンスボディ

```json
{}
```
