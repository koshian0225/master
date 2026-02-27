## ROSBAGからデータをdumpして同期を取り、CMRNetやLHMap-locで使用できる形式にデータをまとめるツール

### 使い方
1. 環境構築（buildするだけ）
```bash
cd ~/rosbag_dump/docker
./build.sh
```

2. Autowareなどで作成したROSBAGをディレクトリに保存する
~/rosbag_dump/shared_dir/rosbag_dump/rosbagの中にデータを取得したいROSBAGを保存してください。

3. mainコードの実行
```bash
python3 /workspace/script/rosbag_dump.py 
```

### 注意
このコードはROSの達人の徳田君と偉大な先輩が作ってくださったコードです。
共同研究先との兼ね合いもありますので研究室内での使用にとどめておいてください。

### その他（トラブルシューティング）
#### numpy.typing（2024.11.22追記）
numpy.typing関連のエラーが出た場合はnumpyのバージョンが古い可能性があるので新しいバージョンのものに変えてください。
（1.20.3以上なら使えるみたいです。）