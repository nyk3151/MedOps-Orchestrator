# Med-Ops Orchestrator

医師が自然言語で指示すると、エージェントが PACS（Orthanc）・解析エンジン（3D Slicer +
SlicerAorticPipeline）・データベースをまたいで処理を進め、**画像の取得 → 大動脈の計測 → DB 登録**
までを一通り実行します。終わるとダッシュボードが自動で更新されます。

PoC の題材は **CT での大動脈計測**です。椎体レベル（T8 / T12）ごとに断面積・径・true lumen 比を求めます。

```
Streamlit（左: 統計ダッシュボード / 右: 指令box）
   └ エージェント（Gemini function calling / オフライン MOCK）  …… 司令塔。計算もパス解決もしない
        ├─ 解析ジョブサーバー（FastAPI, GPU 同時実行 1）
        │     ├─ Orthanc (PACS)：検索・取得 → DICOM→NIfTI 変換（幾何チェック付き）
        │     └─ 3D Slicer headless + AorticPipeline：セグメンテーション・計測・解析画面の保存
        └─ PostgreSQL / SQLite：解析結果 + 実行メタデータ（トレーサビリティ）
```

## できること

指令box（右側のチャット）に日本語で指示します。

| 指示の例 | 動作 |
|---|---|
| 症例一覧を見せて | PACS の症例を、検査日・モダリティ・解析済みかどうかとともに表示 |
| CT 画像はどのくらいありますか？ / 10月の CT データはどのくらいある？ | モダリティ・検査日で絞り込み、検査数・症例数・枚数を回答 |
| 002 を T8 と T12 で解析して | 件数と所要時間を示して確認 →「はい」でジョブ投入。進捗はダッシュボードに表示 |
| 症例002、2026/10/13 の解析画面をだして | 各レベルの**計測断面**（大動脈・真腔・偽腔を色分け）と **CPR 像**（大動脈を伸ばした長軸像）を表示。3D Slicer のシーン（.mrb）もダウンロードできる |
| 002 を PACS で開いて | PACS の検査ページと画像ビューア（Stone Web Viewer）を開くボタンを表示 |
| T8 の統計を出して / CSV で出力して | レベル別の N・平均・SD / CSV ダウンロード |

- 解析は GPU を十数分占有するため、エージェントは**必ず件数と所要時間を伝えて同意を得てから**実行します。
- 1 Study に CT シリーズが複数ある症例は、推測で選ばずに**解析を断ります**（シリーズ一覧を表示）。
- ダッシュボードは解析の完了に合わせて自動で更新されます（症例ごとに最新の成功結果を集計）。

## クラウドのデモ環境

Google Cloud 上で、上の操作をすべて実データ（公開データセット AortaSeg24 の匿名化済み CT。[出典](#データの出典)）で試せます。
URL と審査用アカウント（指令box 用と PACS 用）は、提出フォームの「動作確認の方法」に記載しています。

- 解析はクラウドの GPU（NVIDIA L4）で実行されます。1 症例あたり約 12〜16 分です。
- 費用の歯止めとして、実 CT の解析は 1 日 10 件までです。
- PACS の審査用アカウントは閲覧専用です（画像の削除・追加はできません）。

## 開発範囲（ハッカソン期間中に開発した部分）

審査対象の **AI エージェントとそれを動かすシステム一式は、本ハッカソンの期間中に新規に開発しました**。
大動脈の計測そのもの（解析エンジン）は、以前から公開している既存の部品を**変更せずに**呼び出しています。

| 区分 | 内容 | 場所 |
|---|---|---|
| **ハッカソン期間中に開発** | AI エージェント（Gemini function calling のツール群、確認・拒否の判断、MOCK エージェント） | `src/medops/agent/` |
| 〃 | PACS 連携（Orthanc の検索・取得、シリーズ判定、DICOM→NIfTI 変換と幾何チェック） | `src/medops/pacs/` |
| 〃 | 解析ジョブサーバー、解析エンジンの呼び出し（ローカル / Docker / Cloud Run Jobs）、解析画面 | `src/medops/analysis/` |
| 〃 | データベース（解析結果と実行メタデータ、統計） | `src/medops/db/` |
| 〃 | 画面（統計ダッシュボード + 指令box） | `src/medops/ui/` |
| 〃 | Google Cloud 構成（Cloud Run・Cloud Run Jobs GPU・Cloud SQL・PACS）とデプロイ | `deploy/` |
| 既存の検証済みコードを移植・拡張 | 3D Slicer をヘッドレスで動かすスクリプト（結果の書き出し・解析画面の保存を追加） | `scripts/run_pipeline.py` |
| 既存の部品（変更なし・リポジトリ外） | 解析エンジン [SlicerAorticPipeline](https://github.com/nyk3151/SlicerAorticPipeline)（3D Slicer 拡張）と学習済み nnU-Net モデル | — |

## Quick start（Slicer・GPU・クラウド鍵なし）

各層に **MOCK モード**があり、既定は MOCK です。Slicer・GPU・Orthanc・DB・Vertex AI の鍵が
ひとつも無くても、UI・指令box・ダッシュボード・解析画面は一通り動きます。Python 3.10 以上が必要です。

```bash
git clone https://github.com/nyk3151/MedOps-Orchestrator.git && cd MedOps-Orchestrator
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows (PowerShell)。Git Bash: source .venv/Scripts/activate
                                    # macOS / Linux: source .venv/bin/activate
pip install -e .
python -m medops.launch                              # ジョブサーバー + UI を起動
```

<http://localhost:8501> を開き、指令box で「症例一覧を見せて」「demo_001 を解析して」
「demo_001 の解析画面を出して」「CT はどのくらいある？」などと指示してみてください。
「demo_multi を解析して」は、1 Study に CT シリーズが複数あるため**拒否**されます。

| 層 | MOCK 時の動作 |
|---|---|
| PACS | 内蔵のデモ症例カタログ（`demo_001`〜`demo_008`、`demo_multi`。検査日つき） |
| 解析 | 症例 ID から決まる決定論的なダミー計測値（数秒で完了）と、それに合わせた模式的な解析画面 |
| DB | `data/medops.db`（SQLite） |
| LLM | ルールベースのエージェント（Gemini と同じツールを呼ぶ） |

ダッシュボードには、解析エンジンが動いているモードのデータだけを表示します
（LIVE で動かせば実測値だけ。MOCK のダミー値と混ざることはありません）。

### Docker で Orthanc と PostgreSQL も動かす

```bash
cp .env.example .env
docker compose up -d --wait                       # Orthanc :8042（認証 medops/medops）+ PostgreSQL :5432
cp settings.example.yaml settings.yaml            # 下の 4 行のコメントを外す
```

```yaml
pacs_mode: live
orthanc_user: medops
orthanc_password: medops
database_url: postgresql+psycopg://medops:medops@127.0.0.1:5432/medops
```

```bash
python -m medops.pacs.phantom --upload            # 合成ファントム CT（DICOM）を Orthanc に投入
python -m medops.launch                           # 指令box で「症例一覧を見せて」「phantom_001 を解析して」
```

合成ファントム（`phantom_001`〜`003`、`phantom_multi`）は、リポジトリ内のコードで生成した
架空のデータです。実患者データは含みません。PACS 取得・変換・複数シリーズ拒否の確認用で、
本物の nnU-Net で解析できる画像ではありません。

## フル構成（実CT解析）

| 要件 | 内容 |
|---|---|
| 3D Slicer 5.8 | 拡張 **SlicerNNUNet**（登録名 `slicernnunet`）・TotalSegmentator・SlicerVMTK を含む |
| AorticPipeline | [SlicerAorticPipeline](https://github.com/nyk3151/SlicerAorticPipeline) のチェックアウト |
| 学習済み重み | 大動脈・true lumen の nnU-Net モデル。データセットのライセンス上、**同梱していません**。再現手順は SlicerAorticPipeline の README を参照してください |
| GPU | CUDA 対応 GPU（実測: Quadro RTX 6000 で 1 症例 8〜13 分） |
| Vertex AI | サービスアカウント鍵（`GOOGLE_APPLICATION_CREDENTIALS`）、`GOOGLE_CLOUD_PROJECT`、`GOOGLE_CLOUD_LOCATION`。ADC は使いません |

`settings.yaml` の設定例:

```yaml
slicer_exe: C:/Users/<you>/AppData/Local/slicer.org/Slicer 5.8.1/Slicer.exe
module_dir: C:/path/to/SlicerAorticPipeline/AorticPipeline
aorta_model: C:/path/to/Nets/Aorta_net_2
tl_model: C:/path/to/Nets/Dissection_net
llm_mode: gemini
```

```bash
python -m medops.launch --live    # 起動時に設定を検証し、不足があれば一覧を出して停止
```

LIVE は `--live` フラグ（または `MEDOPS_ANALYSIS_LIVE=1`）を付けたときだけ有効になります。
`settings.yaml` からは有効化できません。設定ファイルの消し忘れで、GPU を十数分占有する解析が
走らないようにするためです。

ローカルで解析した症例は、解析画面の「3D Slicer で開く」ボタンで、保存したシーンをそのまま
3D Slicer で開けます。

## GCP へのデプロイ（推論時だけクラウド GPU）

```
Cloud Run  medops-ui   (Streamlit + Gemini, ID/パスワードでログイン)
   └→ Cloud Run  medops-api   (ジョブサーバー, IAM 保護, 常時 1 インスタンス)
         ├→ Cloud Run  medops-pacs  (Orthanc + Orthanc Explorer 2 + Stone Web Viewer, Orthanc のログイン)
         ├→ Cloud Run Job  medops-gpu  (Slicer + AorticPipeline, NVIDIA L4 ×1, 解析 1 件 = 1 実行)
         ├→ Cloud SQL  (PostgreSQL 16: 解析結果 / Orthanc)
         └→ GCS  (models/ = 推論に必要な重みのみ, data/ = NIfTI・ジョブ入出力・解析画面・シーン)
```

- GPU は解析の実行中だけ起動します（scale to zero）。Cloud Run GPU は asia-northeast1 では
  使えないため、リージョンは **asia-southeast1** です（Gemini は asia-northeast1）。
- GPU ワーカーのイメージ（`deploy/gpu/Dockerfile`）は、検証済みの Windows 環境と同じ
  Slicer 5.8.1（r33241）・拡張機能・Python パッケージで構成します。
- 鍵ファイルはイメージに入れません。Cloud Run 上では、各サービスに割り当てた実行用
  サービスアカウントで認証します。
- UI は ID/パスワードでログインします（ソーシャルログインやメール受信は不要）。
  パスワードは scrypt のハッシュだけを Secret Manager に置きます。API は IAM で保護され、UI からしか呼べません。
- PACS は Orthanc 自身のログインで保護します。書き込めるのはジョブサーバーのアカウントだけで、
  それ以外のアカウントは閲覧専用です（`deploy/pacs/readonly.lua`）。
- 3D Slicer のシーン（約 200 MB）は、Cloud Run の応答サイズ上限を避けるため、ジョブサーバーが
  発行する有効期限 15 分の署名付きリンクで GCS から直接ダウンロードします。
- 実 CT の解析は 1 日あたりの上限（既定 10 件）を超えると受け付けません（GPU 課金の歯止め）。

手順:

1. GCP プロジェクトのオーナーが、Cloud Shell で `PROJECT_ID=<id> bash deploy/gcp/bootstrap.sh` を実行します
   （API の有効化、最小権限のサービスアカウント 4 つ、デプロイ用の鍵の作成）。
2. イメージをビルドします（`deploy/app/Dockerfile`、`deploy/pacs/Dockerfile`、`deploy/gpu/Dockerfile`。
   手順は各ファイルの先頭に記載）。
3. `python deploy/gcp/deploy.py --key <deployer key> --aorta-model <dir> --tl-model <dir> --seed-case <case_id>`
   - 審査用アカウント（ID `judge`）を 2 つ生成します。指令box 用はハッシュ、PACS 用は
     パスワードを Secret Manager に置き、平文は `--account-file`（既定 `~/.gcp/medops-judge-account.txt`）
     にだけ書き出します。作り直すときは `--rotate-account` を付けます。
   - `--seed-case` で、ローカルの Orthanc にある症例をクラウドの PACS にコピーします。

学習済み重みは、推論に必要なファイル（`checkpoint_final.pth`、`plans.json` など）だけを
非公開バケットにアップロードします。学習症例のセグメンテーション（`validation/`）は送りません。

## 設計のポイント

- **設定の解決順**は「環境変数 > `settings.yaml` > リポジトリ相対の既定値」で固定です。
  コードに絶対パスは書いていません。LIVE で使う設定は起動時にまとめて検証します。
- **LLM に渡すのは識別子・モダリティ・検査日・数値だけ**です。ツールの引数は `case_id` / `levels` /
  `job_id` / モダリティ / 日付の範囲に限られ、ファイルパス・画像・氏名などの患者情報は渡しません
  （テストで確認しています）。解析画面の画像も UI にだけ表示し、LLM には渡しません。
  `case_id` は `^[A-Za-z0-9_-]+$` のみ受け付けます。
- **シリーズは推測で選びません。** 候補は CT かつ LOCALIZER / SECONDARY / DERIVED でないシリーズで、
  これがちょうど 1 件のときだけ解析します。2 件以上あれば、シリーズの一覧（枚数・SliceThickness・
  ImageType）を添えてエラーにします。
- **DICOM→NIfTI 変換**では、まず pydicom でヘッダを読み、向きの一貫性・スライス間隔の均一性・
  欠損・ガントリ傾斜を確認します。そのうえで SimpleITK で読み込み、2 つの結果を突き合わせます。
  ファントムで 5 通りの格納向きについてボクセル単位の一致をテストしています。
  実データでも、検証済みの NIfTI とボクセル単位で完全に一致することを確認済みです。
- **解析画面**: 計測に使ったまさにその断面（大動脈の中心線に垂直な 100 mm 四方の断面）と、
  中心線に沿って伸ばした CPR 像を、解析ごとに保存します。3D Slicer のシーン全体も残すので、
  後から数値の根拠を画像で確かめられます。
- **ジョブ方式**: 1 回の解析は「PACS 取得 → 解析 → DB 登録」の段階に分かれ、段階ごとに
  進捗をダッシュボードに表示します。GPU を使う処理は同時に 1 件だけ実行します。
  再現用に `config.json` / `run.log` / `ingest.json` を `data/jobs/{job_id}/` に残します。
- **トレーサビリティ**: `analysis_runs` には、使ったシリーズの UID、検査日・モダリティ、
  AorticPipeline のバージョン（`git describe`）、重みの名前（パスではなく名前）、Slicer のバージョン、
  実行場所（Windows / Docker / Cloud Run）、所要時間、失敗した段階を記録します。
- **統計**: 症例ごとに「最新の成功 run」だけを使うので、再解析しても N は二重に数えられません。
  SD は標本標準偏差です。CSV は UTF-8 BOM 付きで、Excel でそのまま開けます（`TL%` 列）。

## 開発

```bash
pip install -e ".[dev]"
pytest                                                # MOCK / SQLite。外部サービス不要
docker compose exec postgres createdb -U medops medops_test   # once; name must end in _test
MEDOPS_TEST_DATABASE_URL=postgresql+psycopg://medops:medops@127.0.0.1:5432/medops_test pytest tests/test_db.py
python -m medops.pacs.verify <case_id> --reference <verified.nii.gz>   # 変換結果を検証済み NIfTI と比較
```

| ディレクトリ | 内容 |
|---|---|
| `src/medops/pacs/` | Orthanc クライアント、検査カタログ、シリーズ選択、DICOM→NIfTI 変換、ファントム生成 |
| `src/medops/analysis/` | ジョブサーバー、Slicer 呼び出し、解析画面の描画 |
| `src/medops/db/` | スキーマ、登録・集計 |
| `src/medops/agent/` | ツール定義、Gemini / MOCK エージェント |
| `src/medops/ui/` | Streamlit |
| `scripts/run_pipeline.py` | **Slicer 内蔵 Python** で実行するスクリプト。`medops` パッケージは import しない |
| `deploy/` | Docker イメージ（アプリ / PACS / GPU ワーカー）と GCP デプロイスクリプト |

## ライセンス

本リポジトリのコードは [Apache License 2.0](LICENSE) です。

利用しているオープンソースソフトウェアと、そのライセンスは次のとおりです。いずれも改変せずに使っています。
Orthanc とそのプラグインは公式の Docker イメージを別のサーバー（コンテナ）として動かし、
REST API で通信するだけで、本リポジトリのコードには組み込んでいません。

| ソフトウェア | 用途 | ライセンス |
|---|---|---|
| [Orthanc](https://www.orthanc-server.com/)（`orthancteam/orthanc` イメージ） | PACS | GPL-3.0 |
| Orthanc Stone Web Viewer / DICOMweb プラグイン | PACS の画像ビューア | AGPL-3.0 |
| [3D Slicer](https://www.slicer.org/) | 解析の実行環境 | 3D Slicer License（BSD 系） |
| [SlicerAorticPipeline](https://github.com/nyk3151/SlicerAorticPipeline) | 解析エンジン | MIT |
| [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) / SlicerNNUNet | セグメンテーション | Apache-2.0 |
| [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) / SlicerTotalSegmentator | 椎体レベルの特定（公開モデル `total`） | Apache-2.0 |
| PostgreSQL | データベース | PostgreSQL License |
| google-genai, google-cloud-storage, Streamlit, SimpleITK, requests | SDK・画面・画像処理 | Apache-2.0 |
| FastAPI, PyYAML, pydicom, SQLAlchemy | ジョブサーバー・設定・DICOM・DB | MIT |
| uvicorn, pandas, NumPy | ジョブサーバー・集計 | BSD-3-Clause |
| psycopg | PostgreSQL 接続 | LGPL-3.0 |

各ソフトウェアのライセンス全文は、それぞれの配布元を参照してください。
学習済み重みと CT データはリポジトリに含めていません。

## データの出典

クラウドのデモ環境の症例（`001`〜`010`）は、公開データセット **AortaSeg24** の匿名化済み CT 血管造影（CTA）です。
画像は利用条件に従って入手したものを PACS に置いており、本リポジトリでは再配布していません。

> Imran M, et al. Multi-class segmentation of aortic branches and zones in computed tomography angiography:
> The AortaSeg24 challenge. arXiv preprint arXiv:2502.05330. 2025.
> <https://arxiv.org/abs/2502.05330>

Quick start の合成ファントム（`phantom_*`）と MOCK のデモ症例（`demo_*`）は、本リポジトリのコードで生成した架空のデータです。
