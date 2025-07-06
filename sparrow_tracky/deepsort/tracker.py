from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from sparrow_datums import BoxTracking, FrameBoxes, PType

from .distance import iou_distance
from .tracklet import Tracklet


class ByteTracker:
    """ByteTrack implementation maintaining original Tracker interface."""

    def __init__(
        self,
        distance_threshold: float = 0.5,
        distance_function: Callable[
            [FrameBoxes, FrameBoxes], npt.NDArray[np.float64]
        ] = iou_distance,
        missing_threshold: int = 30,
        high_score_threshold: float = 0.6,
        low_score_threshold: float = 0.1,
        match_threshold: float = 0.8,
    ) -> None:
        """
        ByteTrack implementation.

        Parameters
        ----------
        distance_threshold
            IoU threshold for association
        distance_function
            Function for computing pairwise distances
        missing_threshold
            Number of frames to wait before finalizing a tracklet
        high_score_threshold
            High confidence detection threshold
        low_score_threshold
            Low confidence detection threshold
        match_threshold
            Matching threshold for second association
        """
        self.tracked_stracks = []  # type: list[Tracklet]
        self.lost_stracks = []     # type: list[Tracklet]
        self.removed_stracks = []  # type: list[Tracklet]
        
        self.distance_threshold = distance_threshold
        self.distance_function = distance_function
        self.missing_threshold = missing_threshold
        self.high_score_threshold = high_score_threshold
        self.low_score_threshold = low_score_threshold
        self.match_threshold = match_threshold
        
        self.frame_index = 0
        self.start_frame = 0
        self.previous_boxes: Optional[FrameBoxes] = None

    @property
    def active_tracklets(self) -> list[Tracklet]:
        """Return active tracklets for compatibility."""
        return self.tracked_stracks

    @property
    def missing_tracklets(self) -> list[Tracklet]:
        """Return missing tracklets for compatibility."""
        return self.lost_stracks

    @property
    def finished_tracklets(self) -> list[Tracklet]:
        """Return finished tracklets for compatibility."""
        return self.removed_stracks

    @property
    def possible_tracklets(self) -> list[Tracklet]:
        """Return the list of possible tracklets."""
        return self.tracked_stracks + self.lost_stracks

    def track(self, boxes: FrameBoxes) -> None:
        """
        Update tracklets with boxes from a new frame using ByteTrack algorithm.

        Parameters
        ----------
        boxes : FrameBoxes
            A ``(n_boxes, 4)`` array of bounding boxes
        """
        boxes = boxes[np.isfinite(boxes.x)]
        
        if len(boxes) == 0:
            # No detections, predict all tracklets
            for track in self.tracked_stracks:
                track.predict()
                track.mark_lost()
            
            self.lost_stracks.extend(self.tracked_stracks)
            self.tracked_stracks = []
            
            # Remove old lost tracks
            self._remove_old_tracks()
            self.frame_index += 1
            return

        # Separate high and low confidence detections
        if hasattr(boxes, 'scores'):
            scores = boxes.scores
        else:
            scores = np.ones(len(boxes))
            
        high_det_indices = scores >= self.high_score_threshold
        low_det_indices = (scores >= self.low_score_threshold) & (scores < self.high_score_threshold)
        
        high_dets = boxes[high_det_indices] if np.any(high_det_indices) else boxes[:0]
        low_dets = boxes[low_det_indices] if np.any(low_det_indices) else boxes[:0]

        # Predict current tracklets
        for track in self.tracked_stracks:
            track.predict()

        # First association with high confidence detections
        matched_tracks, unmatched_dets, unmatched_tracks = self._associate(
            self.tracked_stracks, high_dets, self.distance_threshold
        )

        # Update matched tracklets
        for track_idx, det_idx in matched_tracks:
            self.tracked_stracks[track_idx].update(high_dets.get_single_box(det_idx))

        # Second association with low confidence detections
        if len(low_dets) > 0 and len(unmatched_tracks) > 0:
            unmatched_tracked_stracks = [self.tracked_stracks[i] for i in unmatched_tracks]
            matched_tracks_2, unmatched_dets_2, unmatched_tracks_2 = self._associate(
                unmatched_tracked_stracks, low_dets, 0.5
            )
            
            # Update matched tracklets from second association
            for track_idx, det_idx in matched_tracks_2:
                original_track_idx = unmatched_tracks[track_idx]
                self.tracked_stracks[original_track_idx].update(low_dets.get_single_box(det_idx))
            
            # Update unmatched tracks
            unmatched_tracks = [unmatched_tracks[i] for i in unmatched_tracks_2]

        # Third association with lost tracklets
        if len(unmatched_dets) > 0 and len(self.lost_stracks) > 0:
            # Create FrameBoxes from unmatched detections
            if unmatched_dets:
                unmatched_high_array = high_dets.array[unmatched_dets]
                unmatched_high_dets_boxes = FrameBoxes(
                    unmatched_high_array, ptype=boxes.ptype, **boxes.metadata_kwargs
                )
            else:
                unmatched_high_dets_boxes = boxes[:0]
            
            matched_tracks_3, unmatched_dets_3, unmatched_lost = self._associate(
                self.lost_stracks, unmatched_high_dets_boxes, 0.5
            )
            
            # Re-activate matched lost tracklets
            for track_idx, det_idx in matched_tracks_3:
                self.lost_stracks[track_idx].re_activate(
                    unmatched_high_dets_boxes.get_single_box(det_idx), self.frame_index
                )
                self.tracked_stracks.append(self.lost_stracks[track_idx])
            
            # Update unmatched detections and lost tracks
            unmatched_dets = [unmatched_dets[i] for i in unmatched_dets_3]
            lost_to_remove = [self.lost_stracks[i] for i in unmatched_lost]
            self.lost_stracks = [track for i, track in enumerate(self.lost_stracks) 
                               if i not in set(matched_tracks_3[:, 0])]

        # Mark unmatched tracked as lost
        for track_idx in unmatched_tracks:
            self.tracked_stracks[track_idx].mark_lost()
            self.lost_stracks.append(self.tracked_stracks[track_idx])

        # Remove unmatched tracked from tracked list
        self.tracked_stracks = [track for i, track in enumerate(self.tracked_stracks) 
                              if i not in set(unmatched_tracks)]

        # Create new tracklets for unmatched high confidence detections
        for det_idx in unmatched_dets:
            new_track = Tracklet(self.frame_index, high_dets.get_single_box(det_idx))
            new_track.activate(self.frame_index)
            self.tracked_stracks.append(new_track)

        # Remove old lost tracks
        self._remove_old_tracks()

        # Update previous boxes for compatibility
        if len(self.possible_tracklets) > 0:
            pred_boxes_array = []
            for track in self.possible_tracklets:
                # Get predicted box and ensure it's 4D
                current_box = track.current_box[:4]
                pred_boxes_array.append(current_box)
            
            if pred_boxes_array:
                pred_array = np.array(pred_boxes_array)
                self.previous_boxes = FrameBoxes(
                    pred_array, ptype=boxes.ptype, **boxes.metadata_kwargs
                )
            else:
                self.previous_boxes = self.empty_previous_boxes(boxes)
        else:
            self.previous_boxes = self.empty_previous_boxes(boxes)

        self.frame_index += 1

    def _associate(self, tracks: list[Tracklet], detections: FrameBoxes, 
                   threshold: float) -> tuple[np.ndarray, list[int], list[int]]:
        """Associate tracklets with detections."""
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), list(range(len(tracks)))

        # Create predicted boxes from tracks
        pred_boxes = []
        for track in tracks:
            current_box = track.current_box
            # Ensure we only use 4 dimensions [x, y, width, height]
            box_4d = current_box[:4]
            pred_boxes.append(box_4d)

        # Create FrameBoxes directly from numpy array
        if pred_boxes:
            pred_array = np.array(pred_boxes)
            track_boxes = FrameBoxes(
                pred_array, ptype=detections.ptype, **detections.metadata_kwargs
            )
        else:
            track_boxes = FrameBoxes(
                np.zeros((0, 4)), ptype=detections.ptype, **detections.metadata_kwargs
            )

        # Compute distance matrix
        distances = self.distance_function(track_boxes, detections)
        
        # Handle invalid distances
        distances = np.nan_to_num(distances, nan=1.0)
        
        # Apply threshold
        distances[distances > threshold] = 1.0

        # Solve assignment problem
        if distances.size > 0:
            track_indices, det_indices = linear_sum_assignment(distances)
            
            # Filter out assignments above threshold
            valid_mask = distances[track_indices, det_indices] < threshold
            track_indices = track_indices[valid_mask]
            det_indices = det_indices[valid_mask]
            
            matches = np.column_stack([track_indices, det_indices])
        else:
            matches = np.empty((0, 2), dtype=int)
            track_indices = np.array([], dtype=int)
            det_indices = np.array([], dtype=int)

        # Find unmatched detections and tracks
        unmatched_dets = [i for i in range(len(detections)) if i not in det_indices]
        unmatched_tracks = [i for i in range(len(tracks)) if i not in track_indices]

        return matches, unmatched_dets, unmatched_tracks

    def _remove_old_tracks(self) -> None:
        """Remove old lost tracks."""
        current_lost = []
        for track in self.lost_stracks:
            if self.frame_index - track.frame_id > self.missing_threshold:
                track.mark_removed()
                self.removed_stracks.append(track)
            else:
                current_lost.append(track)
        self.lost_stracks = current_lost

    @property
    def tracklets(self) -> list[Tracklet]:
        """Return the list of all tracklets."""
        all_tracklets = self.removed_stracks + self.possible_tracklets
        return sorted(all_tracklets, key=lambda t: t.start_index)

    def empty_previous_boxes(self, boxes: FrameBoxes) -> FrameBoxes:
        """Initialize empty FrameBoxes for previous_boxes attribute."""
        return FrameBoxes(
            np.zeros((0, 4)),
            ptype=boxes.ptype,
            **boxes.metadata_kwargs,
        )

    def make_chunk(self, fps: float, min_tracklet_length: int = 1) -> BoxTracking:
        """Consolidate tracklets to BoxTracking chunk."""
        tracklets = [
            t
            for t in self.tracklets
            if len(t) >= min_tracklet_length
            and t.start_index + len(t) > self.start_frame
        ]
        n_objects = len(tracklets)
        metadata: dict[str, Any]
        n_frames = self.frame_index - self.start_frame
        if len(tracklets) == 0:
            ptype = PType.unknown
            metadata = {"fps": fps}
        else:
            ptype = tracklets[0].boxes.ptype
            metadata = tracklets[0].boxes.metadata_kwargs
            metadata["fps"] = fps
        metadata["object_ids"] = [t.object_id for t in tracklets]
        metadata["start_time"] = self.start_frame / fps
        data = np.zeros((n_frames, n_objects, 4)) * np.nan
        for object_idx, tracklet in enumerate(tracklets):
            start = max(tracklet.start_index - self.start_frame, 0)
            end = tracklet.start_index + len(tracklet) - self.start_frame
            n_tracklet_frames = end - start
            data[start:end, object_idx] = tracklet.boxes.array[-n_tracklet_frames:]
        chunk = BoxTracking(
            data,
            ptype=ptype,
            **metadata,
        )
        # Clear removed tracklets that are completely consumed by this chunk
        self.removed_stracks = [
            t for t in self.removed_stracks
            if t.start_index + len(t) > self.start_frame + len(chunk)
        ]
        self.start_frame += len(chunk)
        return chunk


# For backward compatibility, alias ByteTracker as Tracker
Tracker = ByteTracker