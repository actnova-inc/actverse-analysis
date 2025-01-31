from typing import Union

import numpy as np


class Animal:
    def __init__(
        self,
        id: int,
        bbox: Union[np.ndarray, list],
        bbox_score: float,
        keypoints: Union[np.ndarray, list],
        keypoints_score: Union[np.ndarray, list],
        timestamp: float,
    ):
        self.id: int = id
        self.timestamp: float = timestamp
        self.bbox: np.ndarray = np.array(bbox)
        self.bbox_score: float = bbox_score
        self.keypoints: np.ndarray = np.array(keypoints)
        self.keypoints_score: np.ndarray = np.array(keypoints_score)
        self.neck: np.ndarray = None
        self.centre: np.ndarray = None
        self.centre_of_head: np.ndarray = None

    def index_of(self, bodypoint: str):
        raise NotImplementedError
