from typing import Literal, Union

import numpy as np


class BodyPart:
    def __init__(self, en, ko, lang: Literal["en", "ko"] = "en"):
        self._en = en
        self._ko = ko
        self._lang = lang

    def set_lang(self, lang):
        self._lang = lang

    def __str__(self):
        return self._ko if self._lang == "ko" else self._en

    def __repr__(self):
        return f"BodyPart('{self._en}')"

    def __eq__(self, other):
        if isinstance(other, BodyPart):
            return self._en == other._en
        elif isinstance(other, str):
            return self._en == other
        return False

    def __hash__(self):
        return hash(self._en)

    def __getattr__(self, attr):
        return getattr(self._en, attr)


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
