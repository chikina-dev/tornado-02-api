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
uv pip sync
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

DBの追加が必要なのでDockerでもpostgresqlを作ってください
その後以下のかんじのurlを作って.envに保存

```Properties
DATABASE_URL="postgresql://postgres:mysecretpassword@localhost:5434/postgres"
SECRET_KEY=09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7
```

その後以下のコマンドを実行

```sh
uv run init_db.py
```

```sh
uvicorn main:app --reload
```

## APIサーバーリファレンス

userはセッションで管理

- [POST] /create
  - アカウント作成
- [POST] /login
  - アカウントログイン
- [GET] /profile
  - アカウントデータ取得
- [POST] /upload/file
  - ファイルアップロード(スキャナー)
- [POST] /upload/history
  - 検索履歴アップロード
- [GET] /file/[id]
  - ファイル閲覧
- [GET] /history/[date]
  - 検索履歴閲覧
- [GET] /log/[date]?[type]
  - 生データでのアップロードなどの履歴
- [GET] /archive/[date]?[type]
  - AI加工を行なったデータの履歴
- [GET] /test
  - 確認テストを出力
