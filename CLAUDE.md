# Med-Ops Orchestrator

医用画像の取得 → 解析 → DB登録までをEnd-to-Endで完遂する自律型エージェント。
PoCタスクは**CT画像での大動脈計測**（椎体レベルごとの断面積・径・true lumen比）。
前プロジェクトで実CT検証済みの SlicerAorticPipeline をそのまま解析エンジンとして使う。

ユーザー（医師）が自然言語で指示 → Geminiがfunction callingで
PACS / 解析エンジン / DB を横断操作 → ダッシュボードが自動更新される、
という体験が成果物の中心。**新規の解析開発は行わない。**

**公開リポジトリ（ハッカソン提出）。** 審査員がcloneする前提で作る。
この開発機に固有の情報（ローカルのパス、GCP の識別子、判断の記録）は `CLAUDE.local.md`
（gitignore 済み）に置き、このファイルには書かない。

## アーキテクチャ

```
Streamlit (分割UI: 左=統計ダッシュボード / 右=指令box)
   └ Cloud Run
        └ Gemini (Vertex AI, function calling)   ... オーケストレーター
             ├─ Orthanc (PACS)                   : DICOM検索・取得
             ├─ 解析エンジン (3D Slicer headless + GPU) : 大動脈セグメンテーション・計測
             └─ Cloud SQL (PostgreSQL)           : 解析結果 + 実行メタデータ
```

Geminiは「司令塔」。計算もパス解決もしない。
ツール呼び出しの引数は症例ID・レベル指定など**意味のある識別子のみ**。

## ディレクトリ構成

```
MedOps-Orchestrator/            ← gitリポジトリ（Claude Codeはここで起動）
├── CLAUDE.md
├── README.md
├── settings.example.yaml       （settings.yaml はgitignore）
├── .env.example                （.env はgitignore）
├── docker-compose.yml          （Orthanc + PostgreSQL）
├── pyproject.toml
├── src/medops/
│   ├── settings.py             設定解決（環境変数 > settings.yaml > 相対デフォルト）
│   ├── pacs/                   Orthancクライアント、DICOM→NIfTI変換
│   ├── analysis/               ジョブサーバー、Slicer呼び出し
│   ├── db/                     スキーマ、登録・集計
│   ├── agent/                  Geminiツール定義、チャット
│   └── ui/                     Streamlit
├── scripts/
│   └── run_pipeline.py         ★Slicer内蔵Pythonで実行。src/とは別環境
├── tests/
└── data/                       （gitignore。中身は .gitkeep のみ追跡）
    ├── dicom/  nifti/  jobs/  results/
```

- `scripts/` は **Slicer内蔵Python** で動く。`src/medops/` のパッケージを
  import できない（実行環境が別）。両者を混ぜないこと。
- 学習済みモデル（nnU-Net）とCT原本はリポジトリ外。`settings.yaml` で指す。
  既存の配置は他プロジェクトも参照しているため**移動しない**（場所は CLAUDE.local.md）。
- 移植元（前プロジェクト）は読み取り専用。変更しないこと。

## 設定とパスの扱い（公開前提）

**絶対パスをコードに書かない。** 解決順は固定:

```
環境変数 > settings.yaml > リポジトリ相対のデフォルト
```

- デフォルトをリポジトリ相対（`./data/...`）にし、
  **設定ゼロ・clone直後にMOCKモードが動く**状態を保つ。審査員はSlicerもGPUも持っていない。
- YAML内のパスはスラッシュ区切りで書く（`C:/Program Files/...`）。
  Windowsでも動き、バックスラッシュのエスケープ事故を防げる。コード側は `pathlib.Path` で統一。
- LIVEモード起動時は**必ずバリデーション**。必要な設定が未解決なら、
  何が足りないかを明示して起動時点で落とす。10分走ってから
  `FileNotFoundError` が出るのが最悪のパターン。
- 追跡するのは `settings.example.yaml` と `.env.example`。
  実体（`settings.yaml` / `.env`）はgitignore。
- 学習済み重みはリポジトリに含めない（データセットのライセンス制約）。
  READMEに再現手順を書く方針で SlicerAorticPipeline と揃える。

### .gitignore（最低限）

```
settings.yaml
.env
data/
*.nii  *.nii.gz  *.dcm
*-sa.json
```

### README の構成

- **Quick start**: Slicer・GPU不要のデモモード。`docker compose up` →
  `streamlit run` でUI・チャット・ダッシュボードが一通り動く。
- **フル構成**: 必要な追加要件（Slicer拡張、GPU、モデル、Vertex AI鍵）を表で明示。

## 検証済みの制約（厳守・デバッグで踏み直さないこと）

### Slicerヘッドレス
- 引数は「JSON設定ファイル1個の**位置引数**」で渡す。
  `--input` のような `--` 付き引数はSlicer本体のパーサーに食われる。使用禁止。
- print/stdoutは呼び出し元に届かない。**ログは必ずファイルへ書く**。
  デバッグにprintを使わない。
- cwdはSlicerインストールディレクトリになる。相対パス禁止、Slicerへ渡す時点で絶対パス。
- `slicer.app.layoutManager()` はNoneを返す。GUI依存処理は
  `AorticPipelineLogic._isHeadless()` で早期リターン済み。
- `slicer.util.infoDisplay()` はヘッドレスでブロックする恐れあり。
- 結果抽出コードは `runPipeline` の直後・`slicer.util.exit()` より**前**に置く。
- 拡張モジュールの登録名は小文字。nnU-Netは `nnunet` ではなく `slicernnunet`。

### AorticPipeline（解析エンジン。仕様変更しない）
- `runPipeline(...)` は戻り値なし（None）。
- 結果はシーン内テーブルノード `"FinalResults"` に入る。
  列順: `Level, Aorta, TL, FL, TL%`（TL系は列数>=5のときのみ）
- 径は面積から算出: `D = 2*sqrt(A/pi)`
- `vertebraCodes` は `["T8", "T12"]` 形式
- ただし `FinalResults` の Level 列は**表示ラベル**で、胸椎は `Th8` / `Th12` になる（実CTで確認）。
  取り込み時に `engine.canonical_level()` で `T8` / `T12` に戻す。`results.json` は原本のまま残す。
- **CPR 断面サイズは 100 mm（モジュール GUI の既定値）で渡す。** `runPipeline()` の引数既定値は
  `cprSliceSize=40.0` で GUI（ROI 100 / CPR 100）と食い違う。ROI が断面より大きいと
  `_measureArea` の窓の開始位置が負になり、断面の一部しか計測されない。
  `tests/test_engine.py` が ROI ≤ CPR 断面サイズを保証する。
- `runBatch` は術前後ペア前提かつinfoDisplayを含むため使わない
- 1症例あたり実測 8〜13 分（Quadro RTX 6000）、12〜16 分（Cloud Run の NVIDIA L4）

### Gemini / google-genai SDK
- `client.chats.create()` + `send_message` を使う。
  `models.generate_content` でのfunction callingは非推奨警告が出る。
- `build_chat()` はGC絡みの落とし穴があり、Chatオブジェクトにclientの参照を
  保持させる必要がある（移植元のコード内コメント参照）。
- 認証はサービスアカウント鍵 + `GOOGLE_APPLICATION_CREDENTIALS`。
  **ADCは使わない。gcloud authを実行しないこと。**
- 環境変数を変更したらClaude Codeを再起動する（プロセスに引き継がれない）。
- **ツール定義モジュール（`agent/tools.py`）で `from __future__ import annotations` を使わない。**
  SDKの自動関数呼び出しが実際の型注釈で `isinstance` するため、文字列注釈だと
  引数付きツールが全て呼び出し前に失敗する（モデルには「内部エラー」としか見えない）。
- `llm_mode: mock` はルールベースのオフラインエージェント（同じツールを使う）。
  Vertex鍵なしでもチャットのデモが動く。

## 前プロジェクトからの資産

前プロジェクトに実CTで動作確認済みの実装がある（場所は CLAUDE.local.md）。再発明しない。

- `scripts/run_pipeline.py` — Slicerヘッドレス実行スクリプト
- `scripts/job_server.py` — FastAPIジョブサーバー（MOCK/LIVE切替付き）
- `scripts/chat_agent.py` — Gemini function calling層
- `scripts/settings.py` — 設定の優先順解決

移植時は既存の構造（ログ、例外処理、結果抽出の順序）を壊さないこと。
**解析ロジック自体には手を入れない。** 本プロジェクトの新規開発は
PACS層・DB層・UI層と、それらを束ねるオーケストレーションに限る。

## 開発フェーズ

ローカルで全層を組んでからGCPへ段階移行する。
最初からクラウドに載せるとGPU・Slicer・ネットワークの問題が同時に出て
切り分け不能になる。

- [x] **Phase 0**: プロジェクト初期化、資産移植、設定基盤
- [x] **Phase 1**: Orthanc層（Docker、テストDICOM投入、取得・NIfTI変換）
- [x] **Phase 2**: DB層（PostgreSQL、スキーマ、登録/集計API）
- [x] **Phase 3**: オーケストレーション（Geminiツール定義の拡張）
- [x] **Phase 4**: Streamlit分割UI（ダッシュボード自動更新）
- [x] **Phase 5**: GCP移行（Cloud Run / Cloud SQL / Cloud Run Jobs GPU）

各Phase完了時に動作確認してコミットする。まとめて進めない。

## 新規レイヤの方針

### Orthanc
- ローカルはDockerで起動。REST APIで操作。
- エージェントに渡すのは PatientID / StudyInstanceUID などの識別子のみ。
- **`case_id` = Orthanc 上の PatientID（匿名化ID）そのもの。** 変換・接頭辞付与はしない。
- 取得したDICOMは `data/dicom/` へ展開 → NIfTI変換 → `data/nifti/{case_id}_0000.nii.gz`。
  この命名は解析層（`{data_root}/{case_id}_0000.nii.gz`）と揃える。
- **DICOM→NIfTI変換が本プロジェクト最大の新規リスク。** 向き（LPS/RAS）、
  spacing の誤りは解析結果を静かに狂わせる。
  既存の検証済みNIfTIと変換結果を突き合わせて確認すること（`python -m medops.pacs.verify`）。
- **実患者データは投入しない。** テストデータか匿名化済みデータのみ。

#### シリーズ選択（当面の方針）
- **1 Study = 1 Series のデータのみを扱う。** シリーズ選択ロジックは実装しない。
- ingest時に候補シリーズ数を数え、2件以上なら**エラーで停止**する。
  推測で1つ選ばないこと。静かに間違ったシリーズを解析するのが最悪のパターン。
  エラーメッセージには見つかったシリーズの一覧（枚数・SliceThickness・ImageType）を出す。
- 候補の数え方: `Modality == "CT"` かつ `ImageType` に
  `LOCALIZER` / `SECONDARY` / `DERIVED` を含まないもの。
  （スカウト像やdose reportは候補から除くだけで、選択はしない）
- 採用した `SeriesInstanceUID` は `analysis_runs` に記録する（トレーサビリティ）。
- 複数シリーズ対応は将来の拡張。今回のスコープ外。

### PostgreSQL
解析出力が確定しているので、スキーマもそのまま決まる。

`analysis_results` — 1行 = 1症例 × 1椎体レベル

| 列 | 型 | 備考 |
|---|---|---|
| case_id | text | |
| level | text | "T8" / "T12" |
| aorta_area_mm2 | numeric | |
| aorta_dia_mm | numeric | D = 2√(A/π) |
| tl_area_mm2 | numeric | nullable（解離なし症例） |
| fl_area_mm2 | numeric | nullable |
| tl_ratio_pct | numeric | nullable |
| run_id | FK → analysis_runs | |

`analysis_runs` — 実行メタデータ

| 列 | 備考 |
|---|---|
| run_id / job_id | |
| case_id | |
| module_name, module_version | AorticPipeline のバージョン |
| series_instance_uid | 解析に使ったシリーズ |
| aorta_model, tl_model | 使用した重みの識別名（パスは入れない） |
| slicer_version | |
| started_at, finished_at, elapsed_s | |
| status | success / error |

- 列名の `%` はSQLで扱いにくいので `tl_ratio_pct` にする（CSV出力時のみ `TL%` 表記に戻す）。
- **メタデータはトレーサビリティ担保のため必須**。審査基準の可観測性に直結する。
- 実装上の追加列: `analysis_runs.analysis_mode`（mock/live。統計で混ぜない）、`stage`、
  `study_instance_uid`、`error_message`。run はジョブ投入時に `queued` で作成し状態遷移で更新。
- 統計は「症例ごとの最新の成功 run」のみを使う（再解析で N を二重計上しない）。
- ローカルはDocker PostgreSQL、接続文字列の差し替えだけでCloud SQLに移れる形にする。

### ダッシュボード
- N数、平均、標準偏差などの標準統計量。レベル別（T8 / T12）に出す。CSVエクスポート必須。
- 解析完了後に自動更新される（デモのハイライト）。
- Streamlitの再実行周期とジョブ完了タイミングの設計が肝。

## 設計方針

- 解析は1症例10〜20分。**同期APIにしない**。ジョブ方式。
  GPU競合を避けるため同時実行は1件に制限。
- **モックモードを全層に用意する。既定はMOCK、実行には明示フラグ。**
  Slicer・GPU・Orthanc・DBのいずれが無くても上位層を開発できる状態を保つ。
  これが無いとデバッグ速度が落ちる（前プロジェクトで実証済み）。
- config.jsonはジョブごとに実行時生成し `data/jobs/{job_id}/` に残す（再現用）。
- パス解決は必ずAPI側で行う。リクエストやLLMにファイルパスを渡さない。
- 長時間処理は進捗を言語化して報告する。黙って待たせない。

## セキュリティ / ガバナンス

- **Geminiに渡すのは匿名化ID・モダリティ・検査日（StudyDate）・枚数と数値のみ。** DICOM・画像・氏名などの患者情報を渡さない。
  （検査日とモダリティは「10月のCTは何件？」に答えるためチーム判断で追加。解析画面の画像も UI にだけ出す）
- NCD本番環境には接続しない。モックアップDBのみ。
- サービスアカウント鍵・審査用アカウントの平文はリポジトリ外（場所は CLAUDE.local.md）。
  パスをコードに書かない。ログにも出さない。
- `case_id` は `^[A-Za-z0-9_-]+$` のみ許可（パス組み立てに使うためtraversal対策）

## 用語

- 症例識別子は `case_id` で統一（`patient_id` は使わない）
- 椎体レベルは `levels`、値は `["T8", "T12"]` 形式
- ジョブ状態: `queued` → `running` → `success` / `error`

## 実行時の注意

- 実CTでの解析は10〜20分かかりGPUを占有する。**勝手に実行せず必ず確認を取る。**
- 通常の動作確認はMOCKモードで行う。
- Slicer実行時のログはファイルに出る。stdoutには出ない。
- クラウドの GPU ワーカーは Linux 版 Slicer + 仮想ディスプレイ（xvfb）で動く（`deploy/gpu/`）。

## 決定事項

- 審査員向けデモデータ: 合成ファントム（`python -m medops.pacs.phantom --upload`）。
  実データはリポジトリに含めない。クラウドのデモは公開データの匿名化済み CT を使う。
- GCP 構成: **推論時だけクラウドGPUを使う**。
  - TPU は不可（nnU-Net は PyTorch+CUDA。解析ロジックに手を入れない方針）。
  - 解析は Cloud Run Jobs（NVIDIA L4、1解析=1実行、scale to zero）。
    **asia-northeast1 は Cloud Run GPU 非対応** → クラウド側は asia-southeast1 に集約
    （Gemini は asia-northeast1 のまま）。ゾーン冗長は無効、最低 4 CPU / 16 GiB。
  - GPUワーカー: `deploy/gpu/Dockerfile`（Linux版 Slicer 5.8.1 r33241 + xvfb、
    拡張・Python パッケージは Windows 検証環境と同じ版）。AorticPipeline は `git archive` で
    named build context として渡す（リポジトリに入れない）。
  - 学習済み重みは非公開 GCS バケットに置き、実行時にマウント（イメージに焼かない）。
  - デプロイは `deploy/gcp/bootstrap.sh`（Cloud Shell でオーナーが実行）で作る deployer SA の鍵で行う。
  - Cloud Run 上の認証は**アタッチした実行SA**（鍵をイメージに入れない）。
    「ADCを使わない／gcloud auth しない」はローカル開発機の方針として継続。
  - 解析バックエンド `engine_backend`: local（Windows Slicer）/ docker（GPUワーカーをローカルで）/ cloudrun。
    `analysis_runs.engine_backend` に実行場所を記録する（slicer_version だけでは区別できない）。
  - UI は ID/パスワードでログイン（ハッカソン規定: テストアカウントを提出フォームに記載、
    ソーシャル/メールログイン不可）。PACS は Orthanc のログインで、書き込みはジョブサーバーのみ。
- Linux / クラウドで踏んだ制約（デバッグで踏み直さないこと）:
  - ロケール未設定だと Slicer が run_pipeline.py を ASCII で読み失敗し、exit() に届かずハングする
    （UTF-8 ロケール＋timeout で対処）。
  - TotalSegmentator は Slicer 拡張の `setupPythonRequirements()` で入れる
    （素の pip だと requests==2.27.1 制約が残り CLI が落ちる）。Slicer の Python は manylinux2014 まで。
  - GCS FUSE に DICOM を置いて読み戻すと変換が止まる。Cloud Run 上ではインスタンス内の一時領域で
    取得・変換し、NIfTI だけを GCS に置く。
  - Linux Slicer は results.json を書いた**後**、終了処理で double free を起こし exit 1 になることがある。
    成否は全バックエンドで results.json で決める。
  - Cloud Run の通常の応答は 32 MiB まで。シーン（約 200 MB）は署名付き URL で GCS から直接渡す。
  - API サービスは 1 vCPU / 2 GiB が下限（0.5 vCPU は 1 GiB まで。ファイルシステムがメモリ扱い）。
  - イメージはコミットごとのタグ（`:latest` だと新リビジョンに反映されないことがある）。
