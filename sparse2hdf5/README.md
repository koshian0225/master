# sparse2hdf5

HDF5形式で疎な距離画像（sparse depth map）を格納するリポジトリです。

## 概要

このツールは、疎な距離画像データをHDF5形式のファイルに変換・格納するためのPythonスクリプトです。CARLA、KITTI、実環境データなど、複数のデータセット形式に対応しています。

## 使用方法

基本的な使用方法は以下のコマンドで確認できます：

```bash
python sparse2hdf5.py --help
```

### 基本的な実行例

```bash
python sparse2hdf5.py \
  -dc **.json \
  -d **.hdf5 \
  -v kitti
```

### 引数の詳細

- `-dc, --eval-dl-config`: データローダー設定のJSONファイルパス（必須）
- `-d, --eval-data`: HDF5データセットのパス（必須）
- `-v, --version`: データセットのバージョン（carla, kitti, real）（必須）
- `-bs, --block-size`: データセットのブロックサイズ（デフォルト: 0）
- `-b, --batch-size`: バッチサイズ（デフォルト: 1）
- `-ter, --tr-error-range`: 並進誤差範囲 [m]（デフォルト: [0.6, 1.3, 0.7]）
- `-rer, --rot-error-range`: 回転誤差範囲 [deg]（デフォルト: 3.0）
- `--thresholds`: 深度閾値のリスト

## ⚠️ 注意点

### 1. H5Datasetのモード設定

**実行前に必ずH5Datasetのモード設定を確認してください。**

- `mode='w'`: 新規書き込みモード（既存ファイルを上書き）
- `mode='a'`: 追記モード（既存ファイルに追加）
- `mode='r'`: 読み込み専用モード

基本的には `mode='a'` を使用してください。

### 2. 位置データ拡張の使用

位置データ拡張を使用する場合は、`process_dataset`関数内の以下のコメントアウトされたコードを有効にしてください（その後誤差なしをコメントアウト）：

```python
        # aug_map: torch.Tensor = batch[DATASET_MAP]
        # aug_map_np = process_depth_data(aug_map, device)
        # set_depth(data_group, config.get('aug_map_sensor', config['in_map_sensor']), aug_map_np, 
        #          config['frame_id'], int(stamp_sec_list[i]), int(stamp_nsec_list[i]))
```

### 3. Ground Truthの正規化

Ground Truthの正規化設定（`process_depth_data`関数内の処理）は、使用するモデルに合わせて調整してください。現在の設定：

- 値が2の画素を0に変換
- 255倍のスケーリング

## 出力

処理完了後、入力ファイル名に `_wm` （wm : with map）が追加されたファイルが生成されます。

例: `input_data.hdf5` → `input_data_wm.hdf5`

## 注意事項

⚠️ **このコードはとんでもなく雑に作成されたものです。** 本格的な運用前には自分の使用したいモデルやデータセットに合わせて十分なテストと検証を行ってください。