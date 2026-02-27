import rosbag
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import glob
import os
import csv
import shutil
import open3d as o3d
from rich.progress import track


class MyException(Exception):
    pass


# tf_staticをJSONファイルに保存する関数
def get_tf_static_to_json(bag_path, json_path):
    # bagファイルを開く
    bag = rosbag.Bag(bag_path)

    # /tf_staticトピックのメッセージを格納するリスト
    tf_static_msgs = []

    # bagファイルから/tf_staticトピックのメッセージを取得
    for _, msg, _ in bag.read_messages(topics='/tf_static'):
        tf_static_msgs.append(msg)

    # bagファイルを閉じる
    bag.close()

    # tf_staticを取得
    transformations_dict = {}
    for msg_index, msg in enumerate(tf_static_msgs):
        for transform_index, transform in enumerate(msg.transforms):
            transformations_dict[f"msg_{msg_index}_transform_{transform_index}"] = {
                "header": {
                    "seq": transform.header.seq,
                    "stamp": {
                        "secs": transform.header.stamp.secs,
                        "nsecs": transform.header.stamp.nsecs
                    },
                    "frame_id": transform.header.frame_id
                },
                "child_frame_id": transform.child_frame_id,
                "transform": {
                    "translation": {
                        "x": transform.transform.translation.x,
                        "y": transform.transform.translation.y,
                        "z": transform.transform.translation.z
                    },
                    "rotation": {
                        "x": transform.transform.rotation.x,
                        "y": transform.transform.rotation.y,
                        "z": transform.transform.rotation.z,
                        "w": transform.transform.rotation.w
                    }
                }
            }

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # JSON ファイルに保存
    with open(json_path, 'w') as f:
        json.dump(transformations_dict, f, indent=4)


# カメラの内部パラメータをJSONファイルに保存する関数
def get_camerainfo_to_json(bag_path, json_path):
    # bagファイルを開く
    bag = rosbag.Bag(bag_path)

    camerainfo_left_flag = False
    camerainfo_right_flag = False

    # bagファイルから/zed/camerainfoトピックのメッセージを取得
    for _, msg, _ in bag.read_messages(topics='/zed/camerainfo'):
        if (msg.header.frame_id == 'camera_left') and (camerainfo_left_flag == False):
            camerainfo_msg_left = msg
        elif (msg.header.frame_id == 'camera_right') and (camerainfo_right_flag == False):
            camerainfo_msg_right = msg

        if camerainfo_left_flag and camerainfo_right_flag:
            break

    # bagファイルを閉じる
    bag.close()

    # 内部パラメータを取得
    camerainfo_dict = {}
    camerainfo_dict[f"camerainfo_left"] = {
        "header": {
            "seq": camerainfo_msg_left.header.seq,
            "stamp": {
                "secs": camerainfo_msg_left.header.stamp.secs,
                "nsecs": camerainfo_msg_left.header.stamp.nsecs
            },
            "frame_id": camerainfo_msg_left.header.frame_id
        },
        "height": camerainfo_msg_left.height,
        "width": camerainfo_msg_left.width,
        "K": camerainfo_msg_left.K,
    }
    camerainfo_dict[f"camerainfo_right"] = {
        "header": {
            "seq": camerainfo_msg_right.header.seq,
            "stamp": {
                "secs": camerainfo_msg_right.header.stamp.secs,
                "nsecs": camerainfo_msg_right.header.stamp.nsecs
            },
            "frame_id": camerainfo_msg_right.header.frame_id
        },
        "height": camerainfo_msg_right.height,
        "width": camerainfo_msg_right.width,
        "K": camerainfo_msg_right.K,
    }

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # JSON ファイルに保存
    with open(json_path, 'w') as f:
        json.dump(camerainfo_dict, f, indent=4)


# JSONファイルから各センサ間のtfを取得する関数
def get_tf_link_a2link_b_from_json(json_path, link_a, link_b):
    # JSON ファイルを読み込む
    with open(json_path, 'r') as f:
        transformations_dict = json.load(f)

    # link_aからlink_bへのtfを取得
    link_a2link_b_transform = None

    for _, transform_data in transformations_dict.items():
        if transform_data["header"]["frame_id"] == link_a and transform_data["child_frame_id"] == link_b:
            link_a2link_b_transform = transform_data

    # 取得した変換を出力
    if link_a2link_b_transform:
        tf_link_a2link_b = [
            link_a2link_b_transform['transform']['translation']['x'],
            link_a2link_b_transform['transform']['translation']['y'],
            link_a2link_b_transform['transform']['translation']['z'],
            link_a2link_b_transform['transform']['rotation']['x'],
            link_a2link_b_transform['transform']['rotation']['y'],
            link_a2link_b_transform['transform']['rotation']['z'],
            link_a2link_b_transform['transform']['rotation']['w']
        ]

    else:
        raise MyException("データがNoneです。処理を続行できません。")

    return tf_link_a2link_b


# 同次変換行列を作成する関数
def homogeneous_transform(translation, rotation):
    # 回転行列を取得
    r = R.from_quat(rotation)

    # 同次変換行列を構築
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = r.as_matrix()  # 回転行列を代入
    homogeneous_matrix[:3, 3] = translation[:3]  # 位置ベクトルを代入

    return homogeneous_matrix


# 同次変換行列の逆行列を作成する関数
def inverse_homogeneous_transform(homogeneous_matrix):
    # 3x3の回転行列を取得
    rotation_matrix = homogeneous_matrix[:3, :3]

    # 回転行列からクォータニオンに変換
    r = R.from_matrix(rotation_matrix)
    rotation_quaternion = r.as_quat()

    # 位置ベクトルを取得
    translation_vector = homogeneous_matrix[:3, 3]

    return translation_vector, rotation_quaternion


# link_bを経由してlink_aからlink_cまでのtfを計算する関数 ex)baselink→lidar, lidar→cameraを使用してbaselink→cameraを計算
def calc_tf_link_a2link_c(link_a2link_b, link_b2link_c):
    # 同次変換行列を作成
    if len(link_a2link_b) == 7:
        homogeneous_matrix_a2b = homogeneous_transform(link_a2link_b[:3], link_a2link_b[3:])
    else:
        homogeneous_matrix_a2b = link_a2link_b
    if len(link_b2link_c) == 7:
        homogeneous_matrix_b2c = homogeneous_transform(link_b2link_c[:3], link_b2link_c[3:])
    else:
        homogeneous_matrix_b2c = link_b2link_c

    # base_linkからcamera_leftへの変換行列を計算
    homogeneous_matrix_a2c = np.dot(homogeneous_matrix_a2b, homogeneous_matrix_b2c)

    return homogeneous_matrix_a2c


# 同期が取れているデータを保存する関数
def check_sync(cource, tf_base_link2cam_left, cam_json_path):
    left_rgb_dir_path = f'/workspace/output/{cource}/zed-left-image_rect-compressed'
    right_rgb_dir_path = f'/workspace/output/{cource}/zed-right-image_rect-compressed'
    points_dir_path = f'/workspace/output/{cource}/points_raw'

    # ndtマッチングで自己位置を取り直した場合はndtマッチングの自己位置を使用
    if os.path.exists(f'/workspace/output/{cource}/ndt_pose'):
        pose_dir_path = f'/workspace/output/{cource}/ndt_pose'
    else:
        pose_dir_path = f'/workspace/output/{cource}/vehicle-odom'

    output_dir = f'/workspace/output/synced/{cource}'
    left_rgb_path_list = sorted(glob.glob(f'{left_rgb_dir_path}/*.png'))
    right_rgb_path_list = sorted(glob.glob(f'{right_rgb_dir_path}/*.png'))
    points_path_list = sorted(glob.glob(f'{points_dir_path}/*.pcd'))
    pose_path_list = sorted(glob.glob(f'{pose_dir_path}/*.csv'))
    points_index_list = list()
    pose_index_list = list()

    for idx in range(len(left_rgb_path_list)):
        left_rgb_path = left_rgb_path_list[idx]
        left_rgb_timestamp = float(left_rgb_path.split('/')[-1].replace('.png', ''))

        for i in range(len(points_path_list)):
            after_points_path = points_path_list[i]
            after_points_timestamp = float(after_points_path.split('/')[-1].replace('.pcd', ''))
            if after_points_timestamp >= left_rgb_timestamp:
                before_points_path = points_path_list[i - 1]
                before_points_timestamp = float(before_points_path.split('/')[-1].replace('.pcd', ''))
                break

        diff_before = abs(left_rgb_timestamp - before_points_timestamp)
        diff_after = abs(left_rgb_timestamp - after_points_timestamp)

        if diff_before <= diff_after:
            if not int(i - 1) in points_index_list:
                points_index_list.append(int(i - 1))
            elif not int(i - 2) in points_index_list:
                points_index_list.append(int(i - 2))
            else:
                points_index_list.append(int(i))

        else:
            points_index_list.append(int(i))

        for i in range(len(pose_path_list)):
            after_pose_path = pose_path_list[i]
            after_pose_timestamp = float(after_pose_path.split('/')[-1].replace('.csv', ''))
            if after_pose_timestamp >= left_rgb_timestamp:
                before_pose_path = pose_path_list[i - 1]
                before_pose_timestamp = float(before_pose_path.split('/')[-1].replace('.csv', ''))
                break

        diff_before = abs(left_rgb_timestamp - before_pose_timestamp)
        diff_after = abs(left_rgb_timestamp - after_pose_timestamp)

        if diff_before <= diff_after:
            if not int(i - 1) in pose_index_list:
                pose_index_list.append(int(i - 1))
            elif not int(i - 2) in pose_index_list:
                pose_index_list.append(int(i - 2))
            else:
                pose_index_list.append(int(i))

        else:
            pose_index_list.append(int(i))

    os.makedirs(f'{output_dir}/image_left', exist_ok=True)
    os.makedirs(f'{output_dir}/image_right', exist_ok=True)
    os.makedirs(f'{output_dir}/points', exist_ok=True)
    os.makedirs(f'{output_dir}/pose', exist_ok=True)
    for index, (points_index, pose_index) in enumerate(zip(sorted(points_index_list), pose_index_list)):
        shutil.copy(left_rgb_path_list[index], f'{output_dir}/image_left/{index:06d}.png')
        shutil.copy(right_rgb_path_list[index], f'{output_dir}/image_right/{index:06d}.png')
        shutil.copy(points_path_list[points_index], f'{output_dir}/points/{index:06d}.pcd')
        shutil.copy(pose_path_list[pose_index], f'{output_dir}/pose/{index:06d}.csv')

    with open(f'{output_dir}/poses.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        for i in range(len(pose_index_list)):
            with open(f'{output_dir}/pose/{i:06d}.csv', 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    x = float(row[1])
                    y = float(row[2])
                    z = float(row[3])
                    qx = float(row[4])
                    qy = float(row[5])
                    qz = float(row[6])
                    qw = float(row[7])
                    tf_world2base_link = homogeneous_transform([x, y, z], [qx, qy, qz, qw])
                    homogeneous_matrix = calc_tf_link_a2link_c(tf_world2base_link, tf_base_link2cam_left)
                    translation_vector, rotation_quaternion = inverse_homogeneous_transform(homogeneous_matrix)
                writer.writerow([f'{i:06d}', f'{translation_vector[0]}', f'{translation_vector[1]}', f'{translation_vector[2]}', f'{rotation_quaternion[0]}', f'{rotation_quaternion[1]}', f'{rotation_quaternion[2]}', f'{rotation_quaternion[3]}'])

    with open(cam_json_path, 'r') as f:
        camerainfo_dict = json.load(f)

        for _, camerainfo in camerainfo_dict.items():
            if camerainfo["header"]["frame_id"] == "camera_left":
                height_left = camerainfo['height']
                width_left = camerainfo['width']
                K_left = camerainfo['K']
            if camerainfo["header"]["frame_id"] == "camera_right":
                height_right = camerainfo['height']
                width_right = camerainfo['width']
                K_right = camerainfo['K']

    with open(f'{output_dir}/image_left/intrinsic.txt', mode = 'w') as f:
        f.write(str(K_left[0]) + str(','))
        f.write(str(K_left[4]) + str(','))
        f.write(str(K_left[2]) + str(','))
        f.write(str(K_left[5]) + str(','))
        f.write(str(width_left) + str(','))
        f.write(str(height_left) + str(','))

    with open(f'{output_dir}/image_right/intrinsic.txt', mode = 'w') as f:
        f.write(str(K_right[0]) + str(','))
        f.write(str(K_right[4]) + str(','))
        f.write(str(K_right[2]) + str(','))
        f.write(str(K_right[5]) + str(','))
        f.write(str(width_right) + str(','))
        f.write(str(height_right) + str(','))