# Tornado-02-api

## システム構成

- api
- ai

## ライブラリ

- fastAPI

## 実行

[引用](https://fastapi.tiangolo.com/ja/virtual-environments/)

### インストール

```sh
uv venv
```

### 仮想環境の有効化

mac

```sh
source .venv/bin/activate
```

windows

```sh
.venv\Scripts\Activate.ps1
```

### コードの実行

DB の追加が必要なので Docker でも postgresql を作ってください
その後以下のかんじの url を作って.env に保存

```Properties
DATABASE_URL="postgresql://postgres:mysecretpassword@localhost:5434/postgres"
SECRET_KEY=09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7
OPENAI_API_KEY="openai-key"
GOOGLE_APPLICATION_CREDENTIALS="PATH/json-key"
```

その後以下のコマンドを実行

```sh
uv run init_db.py
```

```sh
uvicorn main:app --reload
```

### コードの検証

```sh
uv run pytest -q
```

## APIサーバーリファレンス

userはBearer認証で判断
urlは<https://tornado2025.chigayuki.com>

- [POST] /create
  - アカウント作成
- [POST] /login
  - アカウントログイン
- [GET] /profile?{month}
  - アカウントデータ+指定(未指定時は当月)のアクティブ日一覧取得
- [POST] /upload/file
  - ファイルアップロード(ユーザーによるアップロード)
- [POST] /upload/file/{id}
  - ファイルアップロード(スキャナー用の webhook)
  - id はアカウント作成時に作成にしとく
- [POST] /upload/history/{id}
  - 検索履歴アップロード(自動送信システム用の webhook)
  - id はアカウント作成時に作成にしとく
- [get] /files?{date}
  - 指定の日付のファイルデータ(id のリスト)が出てくる(パラメーターで日付指定、ない場合はその日付)
- [GET] /file/{id}
  - ファイル閲覧
- [GET] /history/{date}
  - 検索履歴閲覧
- [GET] /log/{date}?{type}
  - 生データでのアップロードなどの履歴
- [GET] /archive/{date}?{type}
  - AI 加工を行なったデータの履歴
- [GET] /test?{date}&{tags}
  - 確認テストを出力(パラメーターで日付指定、ない場合はその日付)
  - タグで指定可能
- [GET] /tags
  - タグ一覧取得 (AI により自動生成されたタグ)
- [GET] /tags/search_history/{search_history_id}
  - 指定検索履歴のタグ一覧
  - (付与/生成は定期実行 AI が実施)

## 定期実行
