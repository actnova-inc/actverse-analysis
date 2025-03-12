import numpy as np

from .animal import Animal

BODY_PART_INDEX: dict[str, int] = {
    "nose": 0,
    "left ear": 1,
    "right ear": 2,
    "left forepaw": 3,
    "right forepaw": 4,
    "left hindpaw": 5,
    "right hindpaw": 6,
    "tail root": 7,
    "tail center": 8,
    "tail tip": 9,
    "body center": 10,
}

SKELETON_MOUSE: list[list[int]] = [
    [0, 1],  # nose - left eye
    [1, 2],  # left eye - right eye
    [2, 0],  # right eye - nose
    [3, 10],  # left forepaw - body center
    [4, 10],  # right forepaw - body center
    [10, 7],  # body center - tail root
    [5, 7],  # left hindpaw - tail root
    [6, 7],  # right hindpaw - tail root
    [7, 8],  # tail root - tail center
    [8, 9],  # tail center - tail tip
]


class Mouse(Animal):
    def __init__(
        self,
        id: int,
        bbox: np.ndarray,
        bbox_score: float,
        keypoints: np.ndarray,
        keypoints_score: np.ndarray,
        timestamp: float,
        origin_shape: tuple[int, int] = None,
        normalized: bool = False,
    ):
        super().__init__(id, bbox, bbox_score, keypoints, keypoints_score, timestamp)
        self._neck: np.ndarray = None
        self._centre: np.ndarray = None
        self._centre_of_head: np.ndarray = None
        self._orientation: np.ndarray = None  # 머리 - 몸 중심 방향
        self._moving_direction: np.ndarray = None  # 움직이는 방향
        self.origin_shape = origin_shape  # (height, width)
        self.normalized = normalized
        assert (
            not self.normalized or self.origin_shape is not None
        ), "origin_shape is required when normalized is False"

    def __repr__(self):
        return f"Mouse(id={self.id})"

    def get_keypoint(self, body_part: str):
        return self.denormalize(self.keypoints[self.index_of(body_part)])

    def index_of(self, bodypoint: str):
        return BODY_PART_INDEX[bodypoint.lower()]

    def denormalize(self, point: tuple[float, float]):
        return (
            np.array(
                [
                    point[0] * self.origin_shape[1],
                    point[1] * self.origin_shape[0],
                ]
            )
            if self.normalized
            else point
        )  # (x, y)

    @property
    def neck(self):
        if self._neck is None:
            self._neck = (self.left_ear + self.right_ear) / 2
        return self.denormalize(self._neck)

    @property
    def centre(self, method=None):
        if self._centre is None:
            if method == "mean":
                self._centre = np.mean(self.keypoints, axis=0)
            else:
                self._centre = self.keypoints[self.index_of("body center")]
        return self.denormalize(self._centre)

    @property
    def centre_of_head(self):
        if self._centre_of_head is None:
            self._centre_of_head = self.keypoints[self.index_of("nose")]
        return self.denormalize(self._centre_of_head)

    @neck.setter
    def neck(self, value):
        self._neck = value

    @centre.setter
    def centre(self, value):
        self._centre = value

    @centre_of_head.setter
    def centre_of_head(self, value):
        self._centre_of_head = value

    @property
    def nose(self):
        return self.denormalize(self.keypoints[self.index_of("nose")])

    @property
    def left_ear(self):
        return self.denormalize(self.keypoints[self.index_of("left ear")])

    @property
    def right_ear(self):
        return self.denormalize(self.keypoints[self.index_of("right ear")])

    @property
    def left_forepaw(self):
        return self.denormalize(self.keypoints[self.index_of("left forepaw")])

    @property
    def right_forepaw(self):
        return self.denormalize(self.keypoints[self.index_of("right forepaw")])

    @property
    def left_hindpaw(self):
        return self.denormalize(self.keypoints[self.index_of("left hindpaw")])

    @property
    def right_hindpaw(self):
        return self.denormalize(self.keypoints[self.index_of("right hindpaw")])

    @property
    def tail_root(self):
        return self.denormalize(self.keypoints[self.index_of("tail root")])

    @property
    def tail_center(self):
        return self.denormalize(self.keypoints[self.index_of("tail center")])

    @property
    def tail_tip(self):
        return self.denormalize(self.keypoints[self.index_of("tail tip")])

    @property
    def body_center(self):
        return self.denormalize(self.keypoints[self.index_of("body center")])
