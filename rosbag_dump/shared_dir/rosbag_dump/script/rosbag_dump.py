from __future__ import annotations

import time
from datetime import datetime
import os
import csv
from concurrent.futures import Future, ThreadPoolExecutor
from hashlib import sha1
from itertools import chain
from multiprocessing import Event, Process, Queue
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml
from cv_bridge import CvBridge
from pydantic import BaseModel, PositiveInt
from pypcd4 import PointCloud
from collections import defaultdict

# from rich import print
from rosbags.highlevel import AnyReader
from rosbags.typesys.stores.ros1_noetic import (
    sensor_msgs__msg__CameraInfo as CameraInfo,
)
from rosbags.typesys.stores.ros1_noetic import (
    sensor_msgs__msg__CompressedImage as CompressedImage,
)
from rosbags.typesys.stores.ros1_noetic import sensor_msgs__msg__Image as Image
from rosbags.typesys.stores.ros1_noetic import (
    sensor_msgs__msg__PointCloud2 as PointCloud2,
)
from rosbags.typesys.stores.ros1_noetic import std_msgs__msg__Header as Header

from utils import *

PathLike = Union[str, Path]

cv_bridge = CvBridge()


class TopicInfo(BaseModel):
    rosbag_path: Path
    topic_name: str
    msg_type: str
    stamps: List[float]
    synced_stamps: List[float]
    elapsed: List[float]

    def get_average_frequency(self) -> float:
        differences = [
            self.stamps[i + 1] - self.stamps[i] for i in range(len(self.stamps) - 1)
        ]

        # average_difference = sum(differences) / len(differences)
        if differences:
            average_difference = sum(differences) / len(differences)
        else:
            average_difference = 0.0000000000000000000000001
        average_frequency = 1 / average_difference

        return average_frequency


class RosbagInfo(BaseModel):
    rosbag_path: Path
    topics: List[TopicInfo]

    @property
    def num_topics(self) -> int:
        return len(self.topics)

    @property
    def start_time(self) -> float:
        return min([topic.stamps[0] for topic in self.topics])

    @property
    def end_time(self) -> float:
        return max([topic.stamps[-1] for topic in self.topics])

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def average_frequency(self) -> float:
        return (
            sum([topic.get_average_frequency() for topic in self.topics])
            / self.num_topics
        )

    @property
    def max_frequency(self) -> float:
        return max([topic.get_average_frequency() for topic in self.topics])

    @property
    def min_frequency(self) -> float:
        return min([topic.get_average_frequency() for topic in self.topics])

    @property
    def topic_names(self) -> List[str]:
        return [topic.topic_name for topic in self.topics]

    @property
    def topic_types(self) -> List[str]:
        return [topic.msg_type for topic in self.topics]

    @property
    def topic_stamps(self) -> List[List[float]]:
        return [topic.stamps for topic in self.topics]

    @property
    def topic_synced_stamps(self) -> List[List[float]]:
        return [topic.synced_stamps for topic in self.topics]

    @property
    def topic_elapsed(self) -> List[List[float]]:
        return [topic.elapsed for topic in self.topics]

    @property
    def topic_average_frequencies(self) -> List[float]:
        return [topic.get_average_frequency() for topic in self.topics]

    @property
    def topic_max_frequencies(self) -> List[float]:
        return [
            max(
                topic.stamps[i + 1] - topic.stamps[i]
                for i in range(len(topic.stamps) - 1)
            )
            for topic in self.topics
        ]

    @property
    def topic_min_frequencies(self) -> List[float]:
        return [
            min(
                topic.stamps[i + 1] - topic.stamps[i]
                for i in range(len(topic.stamps) - 1)
            )
            for topic in self.topics
        ]

    @property
    def image_topics(self) -> List[str]:
        return [topic.topic_name for topic in self.topics if "Image" in topic.msg_type]

    @property
    def compressed_image_topics(self) -> List[str]:
        return [
            topic.topic_name
            for topic in self.topics
            if "CompressedImage" in topic.msg_type
        ]

    @property
    def point_cloud_topics(self) -> List[str]:
        return [
            topic.topic_name for topic in self.topics if "PointCloud2" in topic.msg_type
        ]

    @property
    def camera_info_topics(self) -> List[str]:
        return [
            topic.topic_name for topic in self.topics if "CameraInfo" in topic.msg_type
        ]

    @property
    def datatime(self) -> str:
        return datetime.fromtimestamp(os.path.getmtime(self.rosbag_path)).strftime(
            r"%d %b %Y at %H:%M:%S"
        )


class BaseData(BaseModel):
    rosbag_path: Path
    topic_name: str
    paths: List[Path] = []
    stamps: List[float]

    @property
    def num_paths(self) -> int:
        return len(self.paths)

    @property
    def start_time(self) -> float:
        return self.stamps[0]

    @property
    def end_time(self) -> float:
        return self.stamps[-1]


class LidarData(BaseData):
    pass


class ImageData(BaseData):
    pass


class CameraInfoData(BaseData):
    intrinsics: Optional[List[List[float]]] = None
    distortion: Optional[List[float]] = None
    imgae_size: Optional[Tuple[int, int]] = None


class RosbagLoader:
    def __init__(
        self,
        output_dir: PathLike,
        num_workers: PositiveInt = 1,
        sync_threshold: float = 0.1,
        course: str = 'None'
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.dump_times: List[float] = []
        self.sync_threshold = sync_threshold
        self.num_workers = num_workers
        self.course = course

    def load_file(self, input_paths: PathLike) -> None:
        if not isinstance(input_paths, Iterable):
            raise TypeError(
                f"Expected Iterable[str or Path] but got {type(input_paths).__name__}"
            )
        input_paths = [Path(path) for path in input_paths]

        ros1_paths = [
            path for path in input_paths if path.is_file() and path.suffix == ".bag"
        ]
        ros2_paths = [
            path
            for path in input_paths
            if path.is_dir()
            and len(list(path.iterdir())) > 0
            and any(child.suffix == ".db3" for child in path.iterdir())
        ]
        self.input_paths = ros1_paths + ros2_paths
        if not self.input_paths:
            return
        self.pre_fetching_queue: Queue[list[TopicInfo]] = Queue()
        self.pre_fetching_cache: Union[list[TopicInfo], None] = None
        self.pre_fetching_event = Event()
        self.pre_fetching_proc = Process(
            target=self._pre_fetching,
            name="RosbagLoaderPreFetching",
            args=(
                self.input_paths,
                self.num_workers,
                self.sync_threshold,
                self.pre_fetching_queue,
                self.pre_fetching_event,
            ),
            daemon=True,
        )
        self.pre_fetching_proc.start()

        self.dump_data_executor = ThreadPoolExecutor(self.num_workers)

    def set_dump_times(self, timestamps: list[float]) -> None:
        self.timestamps = timestamps

    def _pre_fetching(
        self,
        input_paths: list[Path],
        num_workers: int,
        sync_threshold: float,
        queue: Queue[list[TopicInfo]],
        event: EventClass,
    ):
        topic_infos: list[TopicInfo] = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            topic_infos = list(
                chain.from_iterable(executor.map(RosbagLoader._read_stamp, input_paths))
            )
            executor.shutdown(wait=True)

        topic_infos = RosbagLoader._synchronize_timestamps(topic_infos, sync_threshold)

        queue.put_nowait(topic_infos)
        event.set()

    @staticmethod
    def _read_stamp(path: Path):
        print(f"Reading {path}...")

        with AnyReader([path]) as reader:
            connections = [
                connection
                for connection in reader.connections
                if connection.msgtype
                in (
                    "sensor_msgs/msg/Image",
                    "sensor_msgs/msg/CompressedImage",
                    "sensor_msgs/msg/PointCloud2",
                    # "sensor_msgs/msg/CameraInfo",
                    "geometry_msgs/msg/PoseStamped",
                    "nav_msgs/msg/Odometry",
                )
            ]

            topic_infos: dict[str, TopicInfo] = {}
            for connection in connections:
                topic_infos[connection.topic] = TopicInfo(
                    rosbag_path=path,
                    topic_name=connection.topic,
                    msg_type=connection.msgtype,
                    stamps=[],
                    synced_stamps=[],
                    elapsed=[],
                )

            for connection, _, raw_data in reader.messages(connections=connections):
                header: Header = reader.deserialize(raw_data, connection.msgtype).header

                topic_infos[connection.topic].stamps.append(
                    header.stamp.sec + header.stamp.nanosec * 1e-9
                )

        return list(topic_infos.values())

    @staticmethod
    def _synchronize_timestamps(
        topic_infos: list[TopicInfo], threshold: float
    ) -> list[TopicInfo]:
        ref_stamps = np.array(
            min(topic_infos, key=TopicInfo.get_average_frequency).stamps
        )

        valids = []
        argmins = []
        for topic_info in topic_infos:
            differences = np.abs(np.array(topic_info.stamps) - ref_stamps[:, None])
            valids.append(np.min(differences, axis=1) <= threshold)
            argmins.append(np.argmin(differences, axis=1))

        valid = np.all(valids, axis=0)

        if np.count_nonzero(valid) == 0:
            raise ValueError(f"No valid timestamps found. {topic_infos}")

        new_topic_infos = []
        for topic_info, argmin in zip(topic_infos, argmins):
            topic_info.synced_stamps = np.array(topic_info.stamps)[
                argmin[valid]
            ].tolist()
            topic_info.elapsed = [
                stamp - topic_info.synced_stamps[0]
                for stamp in topic_info.synced_stamps
            ]
            new_topic_infos.append(topic_info)

        return new_topic_infos

    @staticmethod
    def find_nearest_index(array: Iterable[float], value: float) -> int:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()

        return idx

    @staticmethod
    def read_exact_from_rosbag(
        topic_info: TopicInfo, stamp: float
    ) -> Optional[np.ndarray]:
        result: Optional[np.ndarray] = None

        with AnyReader([topic_info.rosbag_path]) as reader:
            connections = [
                connection
                for connection in reader.connections
                if topic_info.topic_name == connection.topic
            ]

            for connection, _, raw_data in reader.messages(
                connections=connections,
                start=int((stamp - 0.5) * 1e9),
                stop=int((stamp + 0.5) * 1e9),
            ):
                msg: Union[Image, CompressedImage, PointCloud2, CameraInfo] = (
                    reader.deserialize(raw_data, connection.msgtype)
                )

                if msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 != stamp:
                    continue

                if topic_info.msg_type == "sensor_msgs/msg/Image":
                    result = cv_bridge.imgmsg_to_cv2(
                        reader.deserialize(raw_data, topic_info.msg_type), "bgr8"
                    )
                elif topic_info.msg_type == "sensor_msgs/msg/CompressedImage":
                    result = cv_bridge.compressed_imgmsg_to_cv2(
                        reader.deserialize(raw_data, topic_info.msg_type), "bgr8"
                    )
                elif topic_info.msg_type == "sensor_msgs/msg/PointCloud2":
                    result = PointCloud.from_msg(msg).numpy()

                if result is not None:
                    return result

    def is_pre_fetching_done(self) -> bool:
        return self.pre_fetching_event.is_set()

    def get_topic_infos(self) -> Optional[list[TopicInfo]]:
        if not self.is_pre_fetching_done():
            return None

        if self.pre_fetching_cache is None:
            self.pre_fetching_cache = self.pre_fetching_queue.get_nowait()

        return self.pre_fetching_cache

    def get_rosbag_infos(self) -> List[RosbagInfo]:
        if not self.is_pre_fetching_done():
            return None
        rosbag_infos = defaultdict(list)
        for topic_info in self.get_topic_infos():
            rosbag_infos[topic_info.rosbag_path].append(topic_info)

        return [
            RosbagInfo(
                rosbag_path=rosbag_path,
                topics=topics,
            )
            for rosbag_path, topics in rosbag_infos.items()
        ]

    def get_data_of(
        self, topic_info: TopicInfo, seconds: float
    ) -> Optional[np.ndarray]:
        if seconds < topic_info.elapsed[0] or topic_info.elapsed[-1] < seconds:
            return None
        stamp = topic_info.synced_stamps[
            RosbagLoader.find_nearest_index(topic_info.elapsed, seconds)
        ]
        return RosbagLoader.read_exact_from_rosbag(topic_info, stamp)

    def _dump_data(
        self, topic_info: TopicInfo, stamps: list[float]
    ) -> Union[LidarData, ImageData, CameraInfoData]:
        # output_base_path = (
        #     self.output_dir
        #     / f'{sha1((str(topic_info.rosbag_path) + topic_info.topic_name).encode()).hexdigest()[:8]}-'
        #     f'{topic_info.topic_name.replace("/", "-")[1:]}'
        # )
        output_base_path = (self.output_dir / self.course / f'{topic_info.topic_name.replace("/", "-")[1:]}')
        output_base_path.mkdir(exist_ok=True, parents=True)

        stamps_set = set(stamps)

        if topic_info.msg_type in (
            "sensor_msgs/msg/Image",
            "sensor_msgs/msg/CompressedImage",
            "sensor_msgs/msg/PointCloud2",
            "geometry_msgs/msg/PoseStamped",
            "nav_msgs/msg/Odometry",
        ):
            cached_stamps = [float(p.stem) for p in output_base_path.iterdir()]
            uncached_stamps_set = stamps_set - set(cached_stamps)
            uncached_stamps = sorted(list(uncached_stamps_set))
        else:
            uncached_stamps_set = stamps_set
            uncached_stamps = stamps

        out_data: Union[LidarData, ImageData, CameraInfoData]
        if topic_info.msg_type in (
            "sensor_msgs/msg/Image",
            "sensor_msgs/msg/CompressedImage",
        ):
            cached_paths = [
                output_base_path / f"{stamp}.png"
                for stamp in (stamps_set & set(cached_stamps))
            ]

            out_data = ImageData(
                rosbag_path=topic_info.rosbag_path,
                topic_name=topic_info.topic_name,
                stamps=topic_info.synced_stamps,
                paths=cached_paths,
            )
            output_base_path.mkdir(exist_ok=True, parents=True)
        elif topic_info.msg_type == "sensor_msgs/msg/PointCloud2":
            cached_paths = [
                output_base_path / f"{stamp}.pcd"
                for stamp in (stamps_set & set(cached_stamps))
            ]

            out_data = LidarData(
                rosbag_path=topic_info.rosbag_path,
                topic_name=topic_info.topic_name,
                stamps=topic_info.synced_stamps,
                paths=cached_paths,
            )
            output_base_path.mkdir(exist_ok=True, parents=True)
        # elif topic_info.msg_type == "sensor_msgs/msg/CameraInfo":
        #     cached_paths = (
        #         [ost_path]
        #         if (ost_path := output_base_path / "ost.yaml").exists()
        #         else []
        #     )

        #     out_data = CameraInfoData(
        #         rosbag_path=topic_info.rosbag_path,
        #         topic_name=topic_info.topic_name,
        #         stamps=topic_info.synced_stamps,
        #         paths=cached_paths,
        #     )
        #     output_base_path.mkdir(exist_ok=True, parents=True)
        elif topic_info.msg_type == "geometry_msgs/msg/PoseStamped":
            cached_paths = [
                output_base_path / f"{stamp}.txt"
                for stamp in (stamps_set & set(cached_stamps))
            ]

            out_data = CameraInfoData(
                rosbag_path=topic_info.rosbag_path,
                topic_name=topic_info.topic_name,
                stamps=topic_info.synced_stamps,
                paths=cached_paths,
            )
            output_base_path.mkdir(exist_ok=True, parents=True)
        elif topic_info.msg_type == "nav_msgs/msg/Odometry":
            cached_paths = [
                output_base_path / f"{stamp}.txt"
                for stamp in (stamps_set & set(cached_stamps))
            ]

            out_data = CameraInfoData(
                rosbag_path=topic_info.rosbag_path,
                topic_name=topic_info.topic_name,
                stamps=topic_info.synced_stamps,
                paths=cached_paths,
            )
            output_base_path.mkdir(exist_ok=True, parents=True)
        else:
            raise ValueError(f"Unsupported message type: {topic_info.msg_type}")

        if not uncached_stamps:
            return out_data

        with AnyReader([topic_info.rosbag_path]) as reader:
            connections = [
                connection
                for connection in reader.connections
                if topic_info.topic_name == connection.topic
            ]

            counter = 0
            for _, _, raw_data in reader.messages(
                connections=connections,
                start=int((uncached_stamps[0] - 0.5) * 1e9),
                stop=int((uncached_stamps[-1] + 0.5) * 1e9),
            ):
                msg: Union[Image, CompressedImage, PointCloud2, CameraInfo] = (
                    reader.deserialize(raw_data, topic_info.msg_type)
                )

                if (
                    stamp := msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                ) not in uncached_stamps_set:
                    continue

                if topic_info.msg_type in (
                    "sensor_msgs/msg/Image",
                    "sensor_msgs/msg/CompressedImage",
                ):
                    output_path = output_base_path / f"{stamp}.png"

                    image: np.ndarray
                    if topic_info.msg_type == "sensor_msgs/msg/Image":
                        image = cv_bridge.imgmsg_to_cv2(
                            reader.deserialize(raw_data, topic_info.msg_type), "bgr8"
                        )
                    else:
                        image = cv_bridge.compressed_imgmsg_to_cv2(
                            reader.deserialize(raw_data, topic_info.msg_type), "bgr8"
                        )

                    cv2.imwrite(str(output_path), image)
                elif topic_info.msg_type == "sensor_msgs/msg/PointCloud2":
                    output_path = output_base_path / f"{stamp}.pcd"

                    pc = PointCloud.from_msg(msg)
                    pc.save(output_path)

                # elif topic_info.msg_type == "sensor_msgs/msg/CameraInfo":
                #     if out_data.num_paths > 0:
                #         continue

                #     output_path = output_base_path / "ost.yaml"

                #     with open(output_path, "w") as fp:
                #         yaml.dump(
                #             {
                #                 "image_width": msg.width,
                #                 "image_height": msg.height,
                #                 "camera_name": "",
                #                 "camera_matrix": {
                #                     "rows": 3,
                #                     "cols": 3,
                #                     "data": msg.K.flatten().tolist(),
                #                 },
                #                 "distortion_model": msg.distortion_model,
                #                 "distortion_coefficients": {
                #                     "rows": 1,
                #                     "cols": len(msg.D),
                #                     "data": msg.D.flatten().tolist(),
                #                 },
                #                 "rectification_matrix": {
                #                     "rows": 3,
                #                     "cols": 3,
                #                     "data": msg.R.flatten().tolist(),
                #                 },
                #                 "projection_matrix": {
                #                     "rows": 3,
                #                     "cols": 4,
                #                     "data": msg.P.flatten().tolist(),
                #                 },
                #             },
                #             fp,
                #             default_flow_style=None,
                #             indent=2,
                #             sort_keys=False,
                #         )

                #     out_data.imgae_size = (msg.width, msg.height)
                #     out_data.intrinsics = msg.K.tolist()
                #     out_data.distortion = msg.D.tolist()

                elif topic_info.msg_type == "geometry_msgs/msg/PoseStamped":
                    # Extract position
                    x = msg.pose.position.x
                    y = msg.pose.position.y
                    z = msg.pose.position.z

                    # Extract orientation (quaternion)
                    qx = msg.pose.orientation.x
                    qy = msg.pose.orientation.y
                    qz = msg.pose.orientation.z
                    qw = msg.pose.orientation.w
                    output_path = output_base_path / f"{stamp}.csv"
                    with open(output_path, 'w') as f:
                        writer = csv.writer(f)
                        writer.writerow(['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
                        writer.writerow([f'{stamp}', f'{x}', f'{y}', f'{z}', f'{qx}', f'{qy}', f'{qz}', f'{qw}'])
                elif topic_info.msg_type == "nav_msgs/msg/Odometry":
                    # Extract position
                    x = msg.pose.pose.position.x
                    y = msg.pose.pose.position.y
                    z = msg.pose.pose.position.z

                    # Extract orientation (quaternion)
                    qx = msg.pose.pose.orientation.x
                    qy = msg.pose.pose.orientation.y
                    qz = msg.pose.pose.orientation.z
                    qw = msg.pose.pose.orientation.w
                    output_path = output_base_path / f"{stamp}.csv"
                    with open(output_path, 'w') as f:
                        writer = csv.writer(f)
                        writer.writerow(['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
                        writer.writerow([f'{stamp}', f'{x}', f'{y}', f'{z}', f'{qx}', f'{qy}', f'{qz}', f'{qw}'])

                out_data.paths.append(output_path)
                counter += 1

        out_data.paths = sorted(list(set(out_data.paths)), key=str)

        return out_data

    def _dump_data_in_range_of(
        self, topic_info: TopicInfo, start: float, end: float
    ) -> Union[LidarData, ImageData, CameraInfoData]:
        stamps = topic_info.synced_stamps[
            RosbagLoader.find_nearest_index(
                topic_info.elapsed, start
            ) : RosbagLoader.find_nearest_index(topic_info.elapsed, end)
        ]

        return self._dump_data(topic_info, stamps)

    def dump_data_in_range_of(
        self, topic_infos: list[TopicInfo], start: float, end: float
    ) -> list[Future[Union[LidarData, ImageData, CameraInfoData]]]:
        return [
            self.dump_data_executor.submit(
                self._dump_data_in_range_of, topic_info, start, end
            )
            for topic_info in topic_infos
        ]

    def _dump_data_at(self, topic_info: TopicInfo, elapsed: float) -> None:
        stamps = [
            topic_info.synced_stamps[
                RosbagLoader.find_nearest_index(topic_info.elapsed, elapsed_)
            ]
            for elapsed_ in elapsed
        ]

        return self._dump_data(topic_info, stamps)

    def dump_data_at(
        self, topic_infos: list[TopicInfo], elapsed: list[float]
    ) -> list[Future[Union[LidarData, ImageData, CameraInfoData]]]:
        return [
            self.dump_data_executor.submit(self._dump_data_at, topic_info, elapsed)
            for topic_info in topic_infos
        ]


def time_display(time):
    if time < 60:
        text = f'{time:.2f}s'
    elif time < 3600:
        text = f'{(time // 60):.0f}m{(time % 60):.0f}s'
    else:
        text = f'{(time // 3600):.0f}h{((time % 3600) // 60):.0f}m{(time % 60):.0f}s'
    return text



if __name__ == "__main__":
    input_path_list = ['/workspace/rosbag/001.bag']

    output_dir='/workspace/output'

    for input_path in input_path_list:
        input_paths=[input_path]
        course = input_path.split('/')[-1].split('.')[0]
        tf_json_path = f'{output_dir}/{course}/transformations_{course}.json'
        cam_json_path = f'{output_dir}/{course}/camerainfo_{course}.json'

        get_tf_static_to_json(input_path, tf_json_path)
        get_camerainfo_to_json(input_path, cam_json_path)
        tf_base_link_to_lidar = get_tf_link_a2link_b_from_json(tf_json_path, 'base_link', 'lidar')
        tf_lidar_to_camera_left = get_tf_link_a2link_b_from_json(tf_json_path, 'lidar', 'camera_left')
        tf_base_link2cam_left = calc_tf_link_a2link_c(tf_base_link_to_lidar, tf_lidar_to_camera_left)

        # インスタンス化
        ## input_pathsはROS1の場合は.bagへのパス、ROS2の場合は.db3を含むディレクトリへのパスを指定する
        rosbag_loader = RosbagLoader(
            output_dir=output_dir,
            num_workers=4,
            sync_threshold=0.1,
            course = course
        )

        os.makedirs(output_dir, exist_ok=True)

        rosbag_loader.load_file(input_paths=input_paths)

        ## インスタンス化と同時にROSBAGの事前解析が別のプロセスで開始される
        ## 以降の操作はこの事前解析が終わるまで出来ないのでget_topic_infos()メソッドの返り値がNoneでなくなるまで待つ
        ## 事前解析が終了すると読み込んだ全てのトピックの情報がリストで返ってくる
        start = time.perf_counter()
        while (topic_infos := rosbag_loader.get_topic_infos()) is None:
            print(f"Waiting for rosbag to be pre fetched... / Elapsed time : {time_display(time.perf_counter() - start)}")
            time.sleep(5)
        print("Rosbag pre fetching done!")

        # 範囲指定によるデータの出力
        ## 出力したいトピックの情報を選択し（複数可）、最初と最後の時間とともにdump_data_in_range_of()メソッドに渡す
        ## 非同期処理なのでデータが直接帰ってくるのではなくFutureオブジェクトのリストが返ってくる
        ## キャッシュ機能があるので二回目以降の実行は高速になる

        futures = rosbag_loader.dump_data_in_range_of(topic_infos, start=0.0, end=200.0)

        ## 以下のようにすべてのFutureオブジェクトのfuture.done()がTrueになれば出力が完了したということになる

        start = time.perf_counter()
        while True:
            print(f"Waiting for dumping to be done... / Elapsed time : {time_display(time.perf_counter() - start)}")

            if all([future.done() for future in futures]):
                break

            time.sleep(5)

        print(f"Dumping done in {time.perf_counter() - start:.03f}s")

        ## 以下のようにFutureオブジェクトのfuture.result()で出力されたデータの情報が取得できる

        for future in futures:
            result = future.result()
            print(f"\ntopic_name:   {result.topic_name}")
            print(f"num_paths:    {result.num_paths}")
            print(f"rosbag_path:  {result.rosbag_path}")
            # 他にも色々使える情報があります

        check_sync(course, tf_base_link2cam_left, cam_json_path)