from typing import List, TypedDict

import numpy as np

from actverse.entity import Animal, BodyPart, Mouse
from actverse.utils.math import vector_to_degree, wrap_angle_180


class Metadata(TypedDict):
    origin_width: int
    origin_height: int


class PoseResult(TypedDict):
    ids: List[str]
    boxes: List[np.ndarray]
    boxes_score: List[float]
    keypoints: List[np.ndarray]
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
    """Create initial metric entry for a body part at the first detection timestamp."""
    metric: Metric = {}
    metric["timestamp"] = [animal.timestamp]
    metric["position"] = [animal.get_keypoint(body_part)]
    metric["distance_change"] = [0]
    metric["cumulative_distance_change"] = [0]
    metric["speed"] = [0]
    metric["average_speed"] = [0]
    if body_part == "body center":
        metric["angle"] = [wrap_angle_180(calc_body_angle(animal))]
        metric["angle_change"] = [0]
        metric["cumulative_angle_change"] = [0]
        metric["angular_speed"] = [0]
    return metric


def update_metric(metric: Metric, animal: Animal, previous: Animal, body_part: str):
    """Append metric values using delta between current and previous observations."""
    current_position = animal.get_keypoint(body_part)
    previous_position = previous.get_keypoint(body_part)
    distance_change = np.linalg.norm(current_position - previous_position)
    cumulative_distance_change = (
        metric["cumulative_distance_change"][-1] + distance_change
    )
    speed = distance_change / (animal.timestamp - previous.timestamp)
    first_ts = metric["timestamp"][0]
    elapsed_since_first_detection = max(1e-9, animal.timestamp - first_ts)
    average_speed = cumulative_distance_change / elapsed_since_first_detection

    metric["timestamp"].append(animal.timestamp)
    metric["position"].append(current_position)
    metric["distance_change"].append(distance_change)
    metric["cumulative_distance_change"].append(cumulative_distance_change)
    metric["speed"].append(speed)
    metric["average_speed"].append(average_speed)

    if body_part == "body center":
        raw_angle = calc_body_angle(animal)
        angle = wrap_angle_180(raw_angle)
        prev_angle = metric["angle"][-1]
        dt = animal.timestamp - previous.timestamp
        angular_speed = calc_angular_speed(angle, prev_angle, dt)
        angle_change = ((angle - prev_angle + 180) % 360) - 180
        cumulative_angle_change = metric["cumulative_angle_change"][-1] + angle_change

        metric["angle"].append(angle)
        metric["angle_change"].append(angle_change)
        metric["cumulative_angle_change"].append(cumulative_angle_change)
        metric["angular_speed"].append(angular_speed)


def _build_video_mice_map(
    prediction: Prediction, image_height: int, image_width: int
) -> list[dict[str, Mouse]]:
    """Transform raw pose results into a list of frame-wise maps: id -> Mouse."""
    video_mice_map: list[dict[str, Mouse]] = []
    for pose_result in prediction["results"]:
        mice_map: dict[str, Mouse] = {}
        for index, animal_id in enumerate(pose_result["ids"]):
            boxes = pose_result.get("boxes", []) or []
            boxes_score = pose_result.get("boxes_score", []) or []
            keypoints_score = pose_result.get("keypoints_score", []) or []

            smoothed = pose_result.get("smoothed_keypoints", None)
            raw = pose_result.get("keypoints", None)

            keypoints = None
            if isinstance(smoothed, list) and index < len(smoothed):
                keypoints = smoothed[index]
            elif isinstance(raw, list) and index < len(raw):
                keypoints = raw[index]

            if (
                index >= len(boxes)
                or index >= len(boxes_score)
                or index >= len(keypoints_score)
                or keypoints is None
            ):
                continue

            mice_map[animal_id] = Mouse(
                id=int(animal_id),
                bbox=boxes[index],
                bbox_score=boxes_score[index],
                keypoints=keypoints,
                keypoints_score=keypoints_score[index],
                timestamp=pose_result["timestamp"],
                origin_shape=(image_height, image_width),
                normalized=True,
            )
        video_mice_map.append(mice_map)
    return video_mice_map


def _ensure_metrics_initialized_for_id(
    metrics: dict[str, dict[str, Metric]],
    animal_id: str,
    current_animal: Mouse,
    body_parts: list[BodyPart],
) -> None:
    """Ensure metric containers exist for an id, initializing on first detection."""
    if animal_id not in metrics:
        metrics[animal_id] = {}
    for body_part in body_parts:
        if body_part not in metrics[animal_id]:
            metrics[animal_id][body_part] = create_metric(current_animal, body_part)


def _accumulate_metrics_from_previous(
    metrics: dict[str, dict[str, Metric]],
    animal_id: str,
    current_animal: Mouse,
    previous_animal: Mouse,
    body_parts: list[BodyPart],
) -> None:
    """Accumulate metric values using the last seen observation for the same id."""
    for body_part in body_parts:
        metric: Metric = metrics[animal_id][body_part]
        update_metric(metric, current_animal, previous_animal, body_part)


def measure_physical_metrics(
    prediction: Prediction, body_parts: list[BodyPart]
) -> list[dict[str, dict[str, Metric]]]:
    """Compute per-id physical metrics, bridging re-appearances without interpolation.

    - Positions/angles are recorded only at observed timestamps.
    - Speed/angular speed use delta and dt across gaps when an id re-appears.
    """
    animal_ids = set()
    metadata = prediction["metadata"]
    image_width = metadata["origin_width"]
    image_height = metadata["origin_height"]
    video_mice_map = _build_video_mice_map(prediction, image_height, image_width)
    # collect unique ids
    for frame_map in video_mice_map:
        for animal_id in frame_map:
            animal_ids.add(animal_id)

    metrics: dict[str, dict[str, Metric]] = {}
    last_seen: dict[str, Mouse] = {}
    for mice_map in video_mice_map:
        for animal_id, current_animal in mice_map.items():
            if animal_id not in last_seen:
                _ensure_metrics_initialized_for_id(
                    metrics, animal_id, current_animal, body_parts
                )
            else:
                previous_animal = last_seen[animal_id]
                _accumulate_metrics_from_previous(
                    metrics, animal_id, current_animal, previous_animal, body_parts
                )
            last_seen[animal_id] = current_animal
    return metrics, sorted(list(animal_ids))


def calc_angular_speed(current_angle, previous_angle, time_interval):
    """Compute angular velocity with wrap-around handling at +/-180 degrees."""
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
