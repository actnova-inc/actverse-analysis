from typing import TypedDict, List

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

def measure_physical_metrics(prediction: Prediction):
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


    metrics: dict[str, dict[str, list]] = {}
    for i in range(len(video_mice_map)):
        for animal_id in video_mice_map[i]:
            if i == 0 or animal_id not in video_mice_map[i - 1]:
                current_animal = video_mice_map[i][animal_id]
                # current_position = current_animal.centre.copy()
                # angle = calc_body_angle(current_animal)
                # distance_change = 0
                # cumulative_distance_change = 0
                # speed = 0
                # average_speed = 0
                # angle_change = 0
                # cumulative_angle_change = 0
                # angular_speed = 0
                metrics[animal_id] = {
                    "position": [current_animal.centre.copy()],
                    "distance_change": [0],
                    "cumulative_distance_change": [0],
                    "speed": [0],
                    "average_speed": [0],
                    "angle": [calc_body_angle(current_animal)],
                    "angle_change": [0],
                    "cumulative_angle_change": [0],
                    "angular_speed": [0],
                    "timestamp": [current_animal.timestamp],
                }
            else:
                current_animal = video_mice_map[i][animal_id]
                previous_animal = video_mice_map[i - 1][animal_id]

                current_position = current_animal.centre
                previous_position = previous_animal.centre

                distance_change = np.linalg.norm(current_position - previous_position)
                cumulative_distance_change = (
                    # analysis_results[-1][animal_id]["cumulative_distance_change"]
                    metrics[animal_id]["cumulative_distance_change"][-1]
                    + distance_change
                )

                time_interval = current_animal.timestamp - previous_animal.timestamp

                speed = distance_change / time_interval
                average_speed = cumulative_distance_change / current_animal.timestamp

                angle = calc_body_angle(current_animal)
                angle = correct_angle(angle, metrics[animal_id]["angle"][-1])
                angular_speed = calc_angular_speed(
                    angle, metrics[animal_id]["angle"][-1], time_interval
                )
                angle_change = angle - metrics[animal_id]["angle"][-1]
                cumulative_angle_change = (
                    metrics[animal_id]["cumulative_angle_change"][-1] + angle_change
                )

                metrics[animal_id]["position"].append(current_position)
                metrics[animal_id]["distance_change"].append(distance_change)
                metrics[animal_id]["cumulative_distance_change"].append(
                    cumulative_distance_change
                )
                metrics[animal_id]["speed"].append(speed)
                metrics[animal_id]["average_speed"].append(average_speed)
                metrics[animal_id]["angle"].append(angle)
                metrics[animal_id]["angle_change"].append(angle_change)
                metrics[animal_id]["cumulative_angle_change"].append(
                    cumulative_angle_change
                )
                metrics[animal_id]["angular_speed"].append(angular_speed)
                metrics[animal_id]["timestamp"].append(current_animal.timestamp)

            # analysis_result[animal_id] = {
            #     "position": current_position,
            #     "distance_change": distance_change,
            #     "cumulative_distance_change": cumulative_distance_change,
            #     "speed": speed,
            #     "average_speed": average_speed,
            #     "angle": angle,
            #     "angle_change": angle_change,
            #     "cumulative_angle_change": cumulative_angle_change,
            #     "angular_speed": angular_speed,
            #     "timestamp": current_animal.timestamp,
            # }
        # analysis_results.append(analysis_result)
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
