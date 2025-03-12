from typing import List, TypedDict

import numpy as np

from actverse.entity import Animal, Mouse
from actverse.utils.math import correct_angle, vector_to_degree


class Metadata(TypedDict):
    origin_width: int
    origin_height: int


class PoseResult(TypedDict):
    ids: List[str]
    boxes: List[np.ndarray]
    boxes_score: List[float]
    smoothed_keypoints: List[np.ndarray]
    keypoints_score: List[np.ndarray]
    timestamp: float


class Prediction(TypedDict):
    metadata: Metadata
    results: List[PoseResult]


class Metric(TypedDict):
    position: list[np.ndarray]
    distance_change: list[float]
    cumulative_distance_change: list[float]
    speed: list[float]
    average_speed: list[float]
    angle: list[float]
    angle_change: list[float]
    cumulative_angle_change: list[float]
    angular_speed: list[float]
    timestamp: list[float]


def create_metric(animal: Animal, body_part: str) -> Metric:
    metric: Metric = {}
    metric["timestamp"] = [animal.timestamp]
    metric["position"] = [animal.get_keypoint(body_part)]
    metric["distance_change"] = [0]
    metric["cumulative_distance_change"] = [0]
    metric["speed"] = [0]
    metric["average_speed"] = [0]
    if body_part == "body center":
        metric["angle"] = [calc_body_angle(animal)]
        metric["angle_change"] = [0]
        metric["cumulative_angle_change"] = [0]
        metric["angular_speed"] = [0]
    return metric


def update_metric(metric: Metric, animal: Animal, previous: Animal, body_part: str):
    current_position = animal.get_keypoint(body_part)
    previous_position = previous.get_keypoint(body_part)
    distance_change = np.linalg.norm(current_position - previous_position)
    cumulative_distance_change = (
        metric["cumulative_distance_change"][-1] + distance_change
    )
    speed = distance_change / (animal.timestamp - previous.timestamp)
    average_speed = cumulative_distance_change / animal.timestamp

    metric["timestamp"].append(animal.timestamp)
    metric["position"].append(current_position)
    metric["distance_change"].append(distance_change)
    metric["cumulative_distance_change"].append(cumulative_distance_change)
    metric["speed"].append(speed)
    metric["average_speed"].append(average_speed)

    if body_part == "body center":
        angle = calc_body_angle(animal)
        angle = correct_angle(angle, metric["angle"][-1])
        angular_speed = calc_angular_speed(
            angle, metric["angle"][-1], animal.timestamp - previous.timestamp
        )
        angle_change = angle - metric["angle"][-1]
        cumulative_angle_change = metric["cumulative_angle_change"][-1] + angle_change

        metric["angle"].append(angle)
        metric["angle_change"].append(angle_change)
        metric["cumulative_angle_change"].append(cumulative_angle_change)
        metric["angular_speed"].append(angular_speed)


def measure_physical_metrics(
    prediction: Prediction, body_parts: list[str]
) -> list[dict[str, dict[str, Metric]]]:
    video_mice_map: list[dict[str, Mouse]] = []
    animal_ids = set()
    metadata = prediction["metadata"]
    image_width = metadata["origin_width"]
    image_height = metadata["origin_height"]
    pose_results = prediction["results"]
    for pose_result in pose_results:
        mice_map = {}
        for index, animal_id in enumerate(pose_result["ids"]):
            mice_map[animal_id] = Mouse(
                id=int(animal_id),
                bbox=pose_result["boxes"][index],
                bbox_score=pose_result["boxes_score"][index],
                keypoints=pose_result["smoothed_keypoints"][index],
                keypoints_score=pose_result["keypoints_score"][index],
                timestamp=pose_result["timestamp"],
                origin_shape=(image_height, image_width),
                normalized=True,
            )
            animal_ids.add(animal_id)
        video_mice_map.append(mice_map)

    metrics: dict[str, dict[str, Metric]] = {}
    for i in range(len(video_mice_map)):  # for each frame
        for animal_id in video_mice_map[i]:  # for each animal
            if animal_id not in metrics:
                metrics[animal_id] = {}
            for body_part in body_parts:  # for each body part
                if i == 0 or animal_id not in video_mice_map[i - 1]:
                    current_animal = video_mice_map[i][animal_id]
                    metric = create_metric(current_animal, body_part)
                    metrics[animal_id][body_part] = metric
                else:
                    metric: Metric = metrics[animal_id][body_part]
                    current_animal = video_mice_map[i][animal_id]
                    previous_animal = video_mice_map[i - 1][animal_id]
                    update_metric(metric, current_animal, previous_animal, body_part)
    return metrics, sorted(list(animal_ids))


def calc_angular_speed(current_angle, previous_angle, time_interval):
    if current_angle - previous_angle > 180:  # np.pi
        previous_angle += 360  # 2 * np.pi
    elif current_angle - previous_angle < -180:  # -np.pi:
        previous_angle -= 360  # 2 * np.pi
    return (current_angle - previous_angle) / time_interval


def calc_body_angle(animal: Animal):
    """동물의 몸체의 각도, 동물이 정우측을 바라보는 경우 0도이며 시계방향으로 회전할수록 각도가 증가"""
    body_vector = animal.neck - animal.centre
    angle = vector_to_degree(body_vector)
    return angle
