# 三次元地図における静的領域と単眼深度推定に基づく動的領域補完による距離推定

## 目次

- [概要](#概要)
- [環境構築](#環境構築)
- [データセット準備](#データセット準備)
- [学習](#学習)
- [評価](#評価)
- [備考](#備考)
- [ベースライン: BP-Net (原著)](#bp-net)

---

## 概要

本リポジトリは修士論文の実装コードです。
事前構築済みの3次元地図から得られる疎な距離画像に対し、動的物体（車・歩行者など）が存在する領域を
単眼深度推定モデル（NDDepth）で補完したうえで、深度補完ネットワーク（BP-Net）によって
高精度な密な深度マップを推定します。

### 処理の流れ

1. **動的クラス検出** — RGB画像から車・歩行者などの動的クラスを検出し、補正マスクを作成する（セグメンテーションモデルのほかYOLO・SAM等の基盤モデルも利用可能）。
2. **動的領域の補完** — 補正マスクに該当するピクセルに対し、単眼深度推定モデル（NDDepth）の出力で地図深度を差し替える。
3. **Depth Completion** — 補完後の疎な深度をBP-Net（Bilateral Propagation Network, CVPR 2024）で密な深度マップに変換する。

### 対応データセット

| データセット | 用途 | 最大深度 | `data=` | `version=` |
|:---|:---|:---:|:---:|:---:|
| CARLA | CG環境での評価 | 80 m | `CARLA` | `carla` |
| KITTI-360 | 実環境での評価・田崎研究室データセット用の事前学習 | 80 m | `KITTI360` | `kitti360` |
| 田崎研究室データセット | 実機環境での評価 | 20 m | `REAL` | `real` |

---

## 環境構築

> [!IMPORTANT]
> 学習には **VRAM 12 GB 以上** の GPU が必要です。
> 複数 GPU 環境では DistributedDataParallel（DDP）による分散学習も利用できます。下記のサーバーで学習・評価の動作を確認しています。

| サーバー | 備考 |
|:---|:---|
| **razer**, **teddybear** | 主に使用 |
| **blackbox**, **tuduku**, **gin系** | 動作確認済み |
| **madorun** | 動作可能（後述の注意事項を参照） |

### 1. リポジトリのクローン

```bash
git clone http://tfsv.tasakilab:5051/git/fukuta/BP-Net_AugMap.git
cd BP-Net_AugMap
git switch conf
```

> [!WARNING]
> `git switch conf` を忘れると `main` ブランチのコードで環境構築が行われます。
> 必ずクローン直後にブランチを切り替えてください。

### 2. Docker イメージのビルドとコンテナの起動

```bash
cd docker
bash build.sh
bash run.sh
```

> [!NOTE]
> **madorun** でも動作しますが、GPU アーキテクチャの違いにより浮動小数点演算の丸め誤差が
> razer / teddybear などと比べてわずかに大きくなる場合があります。
> また CUDA バージョンの違いにより Docker ビルドでエラーが発生することがあります。
> その場合は `docker/Dockerfile_madorun` を適時修正してください。


---

## データセット準備

各データセットは `datas/` 以下に配置します。

```
datas/
├── carla/
│   ├── train/
│   ├── selval/
│   └── test/
├── kitti360/
│   ├── train/
│   ├── selval/
│   └── test/
└── real/
    ├── train/
    ├── selval/
    └── test/
```

各サンプルは `.hdf5` 形式で格納されています（詳細は `datasets/self_made.py` 参照）。

> [!NOTE]
> **田崎研究室データセット（real）のサンプル数内訳**
>
> | シーケンス | train | selval |
> |:---|:---:|:---:|
> | cross1 | 57 | 4 |
> | cross2 | 74 | 4 |
> | follow1 | 62 | 4 |
> | follow2 | 73 | 4 |
> | **合計** | **266** | **16** |
>
> 後述する`configs/data/REAL.yaml` の `train_sample_size` / `test_sample_size` がこれらの値と一致しているか確認してください。

### データセットの前処理

通常のデータセット（`t_data3/2023/fukuta/dataset/`）に対して、事前に[sparse2hdf5](http://tfsv.tasakilab:5051/git/fukuta/sparse2hdf5.git) リポジトリを用いて疎な距離画像を `.hdf5` に格納したものを使用しています。

> [!TIP]
> **Docker 環境には [PMOD-Net](http://tfsv.tasakilab:5051/git/fukuta/PMOD-Net_master.git) の DataLoader が動作する環境が構築済みです。**
> そのため、疎な距離画像を事前に `.hdf5` へ埋め込む前処理は必須ではありません。
> `datasets/self_made.py` を PMOD の DataLoader を用いた実装に書き換えれば、通常のデータセットのまま動作させることができます。（データセット作成後に対応させたため、自分は実装をサボりましたがすぐできるはずです）

### ⚠️ KITTI-360 データセットの既知の問題

> [!WARNING]
> KITTI-360 の **train / selval** データセットのみ、疑似誤差あり・なしのデータが**入れ替わった状態**で作成されてしまっています。
> データセットを作り直さずに `datasets/self_made.py` 側の分岐を逆にすることで対処しているため、
> コードの見た目が直感に反します（`ppa=True` のとき `image_00_in_map` を読む等）。
>
> 気になる場合は sparse2hdf5 で正しい設定のデータセットを作り直し、
> `self_made.py` の以下の分岐を本来の論理に修正してください。
>
> ```python
> # datasets/self_made.py（現状：train/selval のみ逆転している）
> if self.mode == 'train' or self.mode == 'selval':
>     if self.ppa == True:
>         in_map = sample['image_00_in_map'][()]       # ← 本来は in_aug_map
>     else:
>         in_map = sample['image_00_in_aug_map'][()]   # ← 本来は in_map
> ```

### `datasets/self_made.py` 実装メモ

**画像正規化の数値**

`mean=(90.9950, 96.2278, 94.3213)`, `std=(79.2382, 80.5267, 82.1483)` は BP-Net 原著実装の値をそのまま流用しています。
KITTI-360・田崎研究室データセットについてはさほど問題ありませんが、CARLA は CG データのため統計分布が異なる可能性があります。
精度向上を狙う場合は CARLA データセット上で mean/std を再計算することも検討してください。

**クロップの 2 段階構成**

NDDepth の入力サイズは **1216×352** で固定のため、BP-Net への入力とズレが生じないよう、クロップを 2 段階に分けています。

| ステップ | 入力サイズ | 出力サイズ | 処理 |
|:---:|:---|:---|:---|
| 1 | 1408×376（hdf5 格納サイズ） | 1216×352 | ランダムクロップ（学習時）または中央クロップ |
| 2 | 1216×352 | 1216×256 | 固定クロップ（上から 96 px を除去） |

田崎研究室データセット（real）は元画像が **960×540** でアスペクト比が異なるため、ランダムクロップは行わず直接 1216×352 にリサイズした後、ステップ 2 の固定クロップのみ適用しています。

**セグメンテーションのクラス定義**

すべてのデータセットで KITTI-360 の 5 クラス定義に統一しています（詳細は論文参照）。
各データセット固有のラベル ID から 5 クラスへの変換は `LabelConverter` クラスが担っています。

---

## 学習

学習は `train.py` を `torchrun` 経由で実行します。
設定は [Hydra](https://hydra.cc/) で管理されており、すべての項目をコマンドライン引数で上書き可能です。

> [!IMPORTANT]
> 学習前に、使用するデータセットの `configs/data/<データセット名>.yaml` を開き、
> 以下の 2 項目が実際のファイル数と一致しているか確認してください。
>
> | キー | 対応するスプリット |
> |:---|:---|
> | `train_sample_size` | `datas/<データセット>/train/` 内のサンプル数 |
> | `test_sample_size` | `datas/<データセット>/selval/` 内のサンプル数（`test` ではなく `selval`）|
>
> 値がずれていると DataLoader のサンプル数計算が狂い、学習が意図どおりに進まない場合があります。

### KITTI-360 での学習例

```bash
export OMP_NUM_THREADS=4
HYDRA_FULL_ERROR=1 torchrun \
  --nproc_per_node=1 \
  --master_port=4321 \
  train.py \
  gpus=[0] \
  num_workers=4 \
  name=<実験名> \
  ++nddepth_chpt=kitti360eigen \
  net=PMP \
  data=KITTI360 \
  num_classes=5 \
  lr=1e-3 \
  train_batch_size=2 \
  test_batch_size=2 \
  sched/lr=NoiseOneCycleCosMo \
  sched.lr.policy.max_momentum=0.90 \
  nepoch=50 \
  test_epoch=25 \
  ++net.sbn=true \
  ++ppa=true \
  ++training_stage=pretrain
```

> [!NOTE]
> `++nddepth_chpt=` や `++chpt=` の `++` は、Hydra の設定ファイルに定義されていないキーを
> コマンドラインから追加するための記法です。

### コマンドライン引数

チェックポイントは `checkpoints/<name>/` に保存されます。

#### 実験ごとに変更する引数

| 引数 | 説明 | 選択肢 / 例 |
|:---|:---|:---|
| `name=` | 実験名（チェックポイントの保存先フォルダ名になる） | 任意の文字列 |
| `data=` | データセット設定 | `CARLA` / `KITTI360` / `REAL` |
| `training_stage=` | 学習段階。`pretrain` は動的マスク作成に GT を使用、`finetune` は物体検出結果を使用 | `pretrain` / `finetune` |
| `gpus=` | 使用する GPU インデックスのリスト | `[0]`、`[0,1]` など |
| `++nddepth_chpt=` | NDDepth の事前学習済みチェックポイント名（`models/nd_weights` 以下のファイル名） | `kitti360eigen` など |
| `++ppa=` | 疑似誤差あり（PPA）データを使用するか | `true` / `false` |
| `++chpt=` | 学習を途中から再開するチェックポイント名 | チェックポイントのフォルダ名 |
| `version=` | データセットバージョン（`data=` から自動設定されるため通常は不要） | `carla` / `kitti360` / `real` |

#### 通常は固定の引数（従来値を流用）

| 引数 | 説明 | デフォルト |
|:---|:---|:---|
| `net=` | ネットワーク設定 | `PMP` |
| `num_classes=` | セグメンテーションのクラス数 | `5` |
| `loss=` | 損失関数 | `MSMSE` |
| `lr=` | 学習率 | `1e-3`（finetune 時は `1e-4`） |
| `nepoch=` | 総エポック数 | `50` |
| `test_epoch=` | 評価を開始するエポック数 | `25` |
| `train_batch_size=` | 学習時バッチサイズ | `2` |
| `test_batch_size=` | 評価時バッチサイズ | `2` |
| `num_workers=` | DataLoader のワーカー数 | `4` |
| `dynamic_mask_thresh=` | 動的マスク生成のスコア閾値 | `0.2` |

---

## 評価

評価は `test.py` を直接実行します（`torchrun` 不要）。

```bash
HYDRA_FULL_ERROR=1 python test.py \
  gpus=[0] \
  name=<実験名> \
  ++chpt=<チェックポイント名> \
  num_classes=5 \
  net=PMP \
  num_workers=4 \
  data=KITTI360 \
  data.testset.mode=test \
  test_batch_size=1 \
  metric=RMSE \
  ++net.compile=false \
  ++save=false \
  ++ppa=true \
  ++training_stage=inference
```

### 評価固有の引数

評価結果は `checkpoints/<name>/` に保存されます。

| 引数 | 説明 | 備考 |
|:---|:---|:---|
| `++training_stage=inference` | **必須**。評価時は常に `inference` を指定する | 誤るとマスク作成にGTを使用します |
| `data.testset.mode=` | 評価するスプリット | `test`（評価用） |
| `++save=` | 推論結果を `.hdf5` に保存するか | KITTI-360 の `test` セットは容量が非常に大きくなるため、定性評価が必要な場合のみ `true` にする |
| `++ppa=` | PPA データで評価するか | 学習時の設定に合わせる |
| `++chpt=` | 評価するチェックポイント名 | `checkpoints/` 以下のフォルダ名 |

> [!NOTE]
> `++nddepth_chpt=` は **指定不要**です。
> NDDepth の重みは学習時に `checkpoints/<name>/` 内へコピーされるため、`++chpt=` で指定したチェックポイントから自動的に読み込まれます。

---

## 備考

- **学習が動かない場合** — まず `num_workers=` を `4` 以下（`1` や `2`）に下げてみてください。
  DDP 関係のエラーが出る場合は `--nproc_per_node` の数と `gpus=` のリスト長が一致しているか確認してください。

- **引数の確認** — 実験前に引数が目的に沿っているかをしっかり確認してください。

- **DDP 使用時の `--nproc_per_node`** — `gpus=[0,1]` のように複数 GPU を指定する場合は、
  `--nproc_per_node` の数を GPU 数に合わせてください。

- **PMOD との公平な比較（エポック数）** — 比較手法 PMOD は 100 epoch で学習しています。
  本研究では **事前学習 50 epoch + finetune 50 epoch = 計 100 epoch** となるように設定し、公平性を保っています。
  `nepoch=50` を変更する場合はこのバランスに注意してください。

- **損失関数と PMOD との比較** — `loss=MSL1` を指定すると L1 損失ベースの学習になり、
  PMOD と同じ損失関数同士で定量比較できます。また定量評価の指標や計算方法はPMODの移植になります。

- **カメラ内部パラメータ** — 内部パラメータはデータセット・カメラごとに異なります。
  新しい環境・単眼深度推定モデルで実験する際は `datasets/self_made.py` の該当箇所をモデルに合わせて修正してください。


---

> [!NOTE]
> 以下は本研究のベースラインである **BP-Net**（Bilateral Propagation Network, CVPR 2024）の原著READMEです。
> 原著の学習・評価手順（KITTI / NYUv2）や学習済みモデルの情報は下記を参照してください。

---

# BP-Net

## Introduction

This is the pytorch implementation of our paper [Bilateral Propagation Network for Depth Completion](https://arxiv.org/abs/2403.11270).

## Environment


You can directly build the environment by running the following command if you use *conda* as the environment management
tool.
```
conda env create -f environment.yml
```
Then you should have an env named *bp*.

## Setup
Compile the C++ and CUDA code:
```
cd exts
python setup.py install
```

## Dataset
We train and evaluate on KITTI and NYUV2 dataset.

### KITTI
Please download KITTI [depth completion](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion)
dataset.
The structure of data directory:
```
└── datas
    └── kitti
        ├── data_depth_annotated
        │   ├── train
        │   └── val
        ├── data_depth_velodyne
        │   ├── train
        │   └── val
        ├── raw
        │   ├── 2011_09_26
        │   ├── 2011_09_28
        │   ├── 2011_09_29
        │   ├── 2011_09_30
        │   └── 2011_10_03
        ├── test_depth_completion_anonymous
        │   ├── image
        │   ├── intrinsics
        │   └── velodyne_raw
        └── val_selection_cropped
            ├── groundtruth_depth
            ├── image
            ├── intrinsics
            └── velodyne_raw
```

### NYUV2
We used preprocessed NYUv2 [HDF5 dataset](http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz) provided by [Fangchang Ma](https://github.com/fangchangma/sparse-to-dense).
Note, the original full NYUv2 dataset is available at the [official website](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).

The structure of data directory:

```
└── datas
    └── nyudepthv2
        ├── train
        │    ├── basement_0001a
        │    │    ├── 00001.h5
        │    │    └── ...
        │    ├── basement_0001b
        │    │    ├── 00001.h5
        │    │    └── ...
        │    └── ...
        └── val
            └── official
                ├── 00001.h5
                └── ...
```



## Trained Models
You can directly download the trained models and extract them in *checkpoints* directory.
Our models are trained on a GPU workstation with 4 Nvidia GTX 3090.
- [BP_KITTI](https://drive.google.com/file/d/1sNq68jEEyLU2_XmH1UZRs4wOrt_MidcO/view)
- [BP_NYU](https://drive.google.com/file/d/1XfKoJVCkj_J5euTkONI_iBx9ZxLUov9w/view)

## Train
You can also train by yourself.

### KITTI

train on KITTI
```
torchrun --nproc_per_node=4 --master_port 4321 train.py \
gpus=[0,1,2,3] num_workers=4 name=BP_KITTI \
net=PMP data=KITTI \
lr=1e-3 train_batch_size=2 test_batch_size=2 \
sched/lr=NoiseOneCycleCosMo sched.lr.policy.max_momentum=0.90 \
nepoch=30 test_epoch=25 ++net.sbn=true
```


### NYUV2

train on NYUV2
```
torchrun --nproc_per_node=2 --master_port 1100 train.py \
gpus=[0,1] num_workers=4 name=BP_NYU \
net=PMP data=NYU data.num_mask=1 \
lr=2e-3 train_batch_size=8 test_batch_size=1 \
nepoch=100 test_epoch=80 ++net.sbn=true
```

## Test

With the trained model, 
you can test and save results.

### KITTI

test on KITTI selval set
```
python test.py gpus=[0] name=BP_KITTI ++chpt=BP_KITTI \
net=PMP num_workers=4 \
data=KITTI data.testset.mode=selval \
test_batch_size=1 metric=RMSE ++net.compile=true
```


test on KITTI test set and save for submission
```
python test.py gpus=[0] name=BP_KITTI ++chpt=BP_KITTI \
net=PMP num_workers=4 \
data=KITTI data.testset.mode=test data.testset.height=352 \
test_batch_size=1 metric=RMSE ++save=true
```

### NYUV2

test on NYUV2 test set

```
python test.py gpus=[0] name=BP_NYU ++chpt=BP_NYU \
net=PMP num_workers=4 \
data=NYU test_batch_size=1 metric=MetricALL
```

## Citation
If you find this work useful in your research, please consider citing:
```
@article{BP-Net,
title={Bilateral Propagation Network for Depth Completion},
author={Tang, Jie and Tian, Fei-Peng and An, Boshi and Li, Jian and Tan, Ping},
journal={CVPR},
year={2024}
}
```