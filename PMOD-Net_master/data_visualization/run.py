import os
import cv2
import numpy as np
import open3d as o3d

# 入力ディレクトリと出力ディレクトリ
input_dir = "img"
output_dir = "result"

# 出力ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

# カメラの内部パラメータ (例: 焦点距離 fx, fy と光学中心 cx, cy)
fx, fy = 704.0, 704.0  # 焦点距離
cx, cy = 608.0, 68.0

def depth_to_point_cloud(depth_image, fx, fy, cx, cy):
    """深度画像を点群に変換"""
    h, w = depth_image.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_image / 1000.0  # 深度値をメートル単位に変換
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points

# ディレクトリ内のすべての画像を処理
for filename in os.listdir(input_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # 深度画像を読み込み
        depth_path = os.path.join(input_dir, filename)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if depth_image is None:
            print(f"画像を読み込めませんでした: {filename}")
            continue

        # 点群を生成
        points = depth_to_point_cloud(depth_image, fx, fy, cx, cy)

        # 有効な点 (深度が0でない点) をフィルタリング
        points = points[points[:, 2] > 0]

        # Open3DのPointCloudオブジェクトを作成
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 出力ファイルパス
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".ply")

        # PLYファイルとして保存
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"PLYファイルを作成しました: {output_path}")