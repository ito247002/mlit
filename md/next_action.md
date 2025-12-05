# [Action Plan] 国土数値情報（KSJ）特徴量追加プロセス

不動産売買価格予測モデルの精度向上に向けた、外部データ結合タスクの優先順位と実行計画。

## 1. 直近の実行タスク (Current Focus)
**【最優先】地価公示データ（L01）の最近傍結合**
* **理由:** 不動産価格に最も直接的な相関があるため、最初に追加して効果を検証すべき。
* **ゴール:** `train.csv` の各物件に対し、最も近い地価ポイントの「価格」と「距離」を付与する。

## 2. 優先度別タスクリスト

### Phase 1: 環境構築とCore特徴量の実装 (Priority: High)
- [ ] **ライブラリ準備**
    - `geopandas`, `shapely`, `rtree` (または `pygeos`) のインストール確認。
- [ ] **データ取得**
    - [国土数値情報](https://nlftp.mlit.go.jp/ksj/) から「地価公示 (L01)」のShapefileをダウンロード。
- [ ] **地価データ結合の実装 (Nearest Neighbor)**
    - 座標変換: WGS84 (`EPSG:4326`) → 平面直角座標系 (例: `EPSG:6668`) ※メートル計算用
    - 高速化: `scipy.spatial.cKDTree` を使用して全探索を回避。
    - 出力: `nearest_land_price` (円/m²), `dist_to_land_price` (m) を結合。

### Phase 2: 補完的特徴量の追加 (Priority: Medium)
※ Phase 1のモデルが動作確認でき次第着手
- [ ] **駅乗降客数 (S12) の結合**
    - 駅名マッチングではなく、位置情報ベースの最近傍探索で結合（表記ゆれ回避）。
- [ ] **将来推計人口メッシュの結合**
    - `geopandas.sjoin` (Spatial Join) を使用。
    - 物件が含まれるメッシュの「2030年推計人口」などを付与。

### Phase 3: クレンジングと最適化 (Priority: Low)
- [ ] **欠損値戦略の決定**
    - 近傍点が見つからない、メッシュ外のデータの扱い（NaN維持 or 埋め合わせ）。
- [ ] **カラム名の英語化**
    - `L01_006` → `land_price` 等へのリネーム処理の実装。

---

## 3. CLI生成AI用プロンプト (Context for CLI AI)
※ 以下のテキストをCLIツール（Copilot CLI, Aider, ChatGPT等）に入力して、Phase 1のコードを生成させてください。

```text
# 指示
PythonとGeoPandasを使用して、不動産取引データに「最寄りの地価公示価格」を特徴量として結合するスクリプト `add_land_price.py` を作成してください。

# 入力データ
1. ./Data/train.csv
   - カラム: `lon` (経度), `lat` (緯度), その他物件情報
   - 座標系: WGS84
2. ./Data/KSJ/L01-xx.shp (地価公示データ)
   - 国土数値情報のShapefile
   - カラム: `L01_006` (公示価格), ジオメトリ情報

# 処理要件
1. **データ読み込み**: 
   - pandasでcsvを、geopandasでshpを読み込む。
2. **座標変換 (重要)**:
   - 距離計算を正確に行うため、両方のデータを適切な投影座標系（例: EPSG:6668 または JGD2011）に変換する。
3. **最近傍探索 (高速化)**:
   - データ量が多いため、全探索ではなく `scipy.spatial.cKDTree` を使用して、各物件から最も近い地価ポイントを探索する。
4. **データ結合**:
   - 最寄りポイントの `L01_006` を `nearest_land_price` として取得。
   - そのポイントまでの距離を計算し `dist_to_land_price` (メートル単位) として取得。
   - これらを元のDataFrameに結合する。
5. **出力**:
   - 結果を `./Data/train_with_landprice.csv` として保存する。

# 技術スタック
- pandas, geopandas, scipy, shapely