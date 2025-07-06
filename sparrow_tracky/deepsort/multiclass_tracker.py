from typing import Callable

import numpy as np
import numpy.typing as npt
from sparrow_datums import (
    AugmentedBoxTracking,
    BoxTracking,
    FrameAugmentedBoxes,
    FrameBoxes,
    PType,
)

from .distance import iou_distance
from .tracker import ByteTracker


class MultiClassTracker:
    """Maintain and update tracklets with separate classes using ByteTrack."""

    def __init__(
        self,
        n_classes: int,
        distance_threshold: float = 0.5,
        distance_function: Callable[
            [FrameBoxes, FrameBoxes], npt.NDArray[np.float64]
        ] = iou_distance,
        high_score_threshold: float = 0.6,
        low_score_threshold: float = 0.1,
        match_threshold: float = 0.8,
        missing_threshold: int = 30,
    ) -> None:
        """
        Maintain and update tracklets with ByteTrack.

        Parameters
        ----------
        n_classes
            Number of classes to track
        distance_threshold
            IoU threshold for association
        distance_function
            Function for computing pairwise distances
        high_score_threshold
            High confidence detection threshold
        low_score_threshold
            Low confidence detection threshold
        match_threshold
            Matching threshold for second association
        missing_threshold
            Number of frames to wait before finalizing a tracklet
        """
        if n_classes < 1:
            raise ValueError(f"Invalid number of classes: {n_classes}")
        self.n_classes = n_classes
        self.trackers: dict[int, ByteTracker] = {}
        for class_idx in range(n_classes):
            self.trackers[class_idx] = ByteTracker(
                distance_threshold=distance_threshold,
                distance_function=distance_function,
                missing_threshold=missing_threshold,
                high_score_threshold=high_score_threshold,
                low_score_threshold=low_score_threshold,
                match_threshold=match_threshold,
            )

    @property
    def _first_tracker(self) -> ByteTracker:
        return self.trackers[0]

    @property
    def start_frame(self) -> int:
        return self._first_tracker.start_frame

    @property
    def frame_index(self) -> int:
        return self._first_tracker.frame_index

    def track(self, boxes: FrameAugmentedBoxes) -> None:
        """
        Update tracklets with boxes from a new frame.

        Parameters
        ----------
        boxes : FrameAugmentedBoxes
            A ``(n_boxes, 6)`` array of bounding boxes
        """
        for class_idx in range(self.n_classes):
            _boxes = boxes[boxes.labels == class_idx].to_frame_boxes()
            self.trackers[class_idx].track(_boxes)

    def make_chunk(
        self, fps: float, min_tracklet_length: int = 1
    ) -> AugmentedBoxTracking:
        """Consolidate tracklets to AugmentedBoxTracking chunk."""
        n_frames = self.frame_index - self.start_frame
        start_time = self.start_frame / fps
        n_objects = 0
        metadata = {"fps": fps, "start_time": start_time}
        chunks: dict[int, BoxTracking] = {}
        for class_idx in range(self.n_classes):
            chunk = self.trackers[class_idx].make_chunk(fps, min_tracklet_length)
            n_objects += chunk.shape[1]
            chunks[class_idx] = chunk
        if n_objects == 0:
            return AugmentedBoxTracking(
                np.ones((n_frames, 0, 6)), ptype=PType.unknown, **metadata
            )
        data = np.zeros((n_frames, n_objects, 6)) * np.nan
        object_idx = 0
        object_ids = []
        ptype = PType.unknown

        for class_idx in range(self.n_classes):
            chunk = chunks[class_idx]
            if chunk.ptype != PType.unknown:
                ptype = chunk.ptype
                metadata.update(chunk.metadata_kwargs)
            _n_objects = chunk.shape[1]
            object_ids.extend(chunk.object_ids)
            data[:, object_idx : object_idx + _n_objects, :4] = chunk.array
            data[:, object_idx : object_idx + _n_objects, -2] = 1.0
            data[:, object_idx : object_idx + _n_objects, -1] = class_idx
            object_idx += _n_objects
        metadata["object_ids"] = object_ids
        return AugmentedBoxTracking(data, ptype=ptype, **metadata)