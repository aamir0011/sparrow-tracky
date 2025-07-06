from __future__ import annotations

import uuid
from typing import Optional, Union

import numpy as np
from sparrow_datums import FrameBoxes, SingleBox


class Tracklet:
    """Store the location history for an object."""

    def __init__(
        self,
        start_index: int,
        box: Union[SingleBox, FrameBoxes],
        object_id: Optional[str] = None,
    ) -> None:
        """
        Store the location history for an object.

        Parameters
        ----------
        start_index
            The frame index that starts the tracklet
        box
            A NumPy array with shape (4,)
        object_id
            An ID for the tracklet
        """
        self.start_index = start_index
        self.boxes = FrameBoxes.from_single_box(box)
        self.missing_boxes = FrameBoxes(
            np.zeros((0, 4)),
            ptype=self.boxes.ptype,
            **self.boxes.metadata_kwargs,
        )
        self.object_id = object_id if object_id else str(uuid.uuid4())
        
        # ByteTrack specific attributes
        self.score = getattr(box, 'score', 1.0) if hasattr(box, 'score') else 1.0
        self.track_id = self.object_id
        self.frame_id = start_index
        self.tracklet_len = 0
        self.is_activated = False
        self.state = 'new'  # 'new', 'tracked', 'lost', 'removed'
        
        # Kalman filter state (simplified)
        self.mean = np.zeros(8)  # [x, y, a, h, vx, vy, va, vh]
        self.covariance = np.eye(8)
        self._initialize_kalman_state(box)

    def _initialize_kalman_state(self, box: Union[SingleBox, FrameBoxes]) -> None:
        """Initialize Kalman filter state from bounding box."""
        if hasattr(box, 'array'):
            bbox = box.array.flatten()
        else:
            bbox = np.array([box.x, box.y, box.width, box.height])
        
        # Convert to center x, center y, aspect ratio, height format
        w, h = bbox[2], bbox[3]
        cx, cy = bbox[0] + w/2, bbox[1] + h/2
        a = w / max(h, 1e-6)
        
        self.mean[:4] = [cx, cy, a, h]
        self.mean[4:] = 0  # velocities start at 0

    def predict(self) -> None:
        """Predict next state using Kalman filter."""
        # Simple constant velocity model
        dt = 1.0
        F = np.eye(8)
        F[0, 4] = dt  # x += vx * dt
        F[1, 5] = dt  # y += vy * dt
        F[2, 6] = dt  # a += va * dt
        F[3, 7] = dt  # h += vh * dt
        
        self.mean = F @ self.mean
        
        # Process noise
        Q = np.eye(8) * 0.1
        Q[4:, 4:] *= 0.01  # lower noise for velocities
        
        self.covariance = F @ self.covariance @ F.T + Q

    def update(self, box: SingleBox) -> None:
        """Update tracklet with new detection."""
        self.add_box(box)
        self.score = getattr(box, 'score', 1.0) if hasattr(box, 'score') else 1.0
        self.tracklet_len += 1
        
        # Update Kalman filter
        if hasattr(box, 'array'):
            bbox = box.array.flatten()
        else:
            bbox = np.array([box.x, box.y, box.width, box.height])
        
        # Convert to center x, center y, aspect ratio, height format
        w, h = bbox[2], bbox[3]
        cx, cy = bbox[0] + w/2, bbox[1] + h/2
        a = w / max(h, 1e-6)
        
        z = np.array([cx, cy, a, h])
        
        # Update step
        H = np.eye(4, 8)  # observation matrix
        R = np.eye(4) * 0.1  # measurement noise
        
        y = z - H @ self.mean
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        self.mean = self.mean + K @ y
        self.covariance = (np.eye(8) - K @ H) @ self.covariance

    def activate(self, frame_id: int) -> None:
        """Activate tracklet."""
        self.is_activated = True
        self.track_id = self.object_id
        self.frame_id = frame_id
        self.state = 'tracked'

    def re_activate(self, box: SingleBox, frame_id: int) -> None:
        """Re-activate lost tracklet."""
        self.update(box)
        self.state = 'tracked'
        self.is_activated = True
        self.frame_id = frame_id

    def mark_lost(self) -> None:
        """Mark tracklet as lost."""
        self.state = 'lost'

    def mark_removed(self) -> None:
        """Mark tracklet as removed."""
        self.state = 'removed'

    @property
    def current_box(self) -> np.ndarray:
        """Get current bounding box in [x, y, w, h] format."""
        # Convert from [cx, cy, a, h] to [x, y, w, h]
        cx, cy, a, h = self.mean[:4]
        w = a * h
        x = cx - w/2
        y = cy - h/2
        return np.array([x, y, w, h])

    def __len__(self) -> int:
        """Check number of boxes in the tracklet."""
        return len(self.boxes)

    def add_box(self, box: SingleBox) -> None:
        """Append a box to the end of the array."""
        self.boxes = self.boxes.add_box(box)

    def add_missing_box(self) -> None:
        """Append a box to the missing box list."""
        self.missing_boxes = self.missing_boxes.add_box(self.previous_box)

    def scratch_missing_boxes(self) -> None:
        """Clear the missing box list."""
        self.missing_boxes = FrameBoxes(
            np.zeros((0, 4)),
            ptype=self.boxes.ptype,
            **self.boxes.metadata_kwargs,
        )

    def finalize_missing_boxes(self) -> None:
        """Finish the missing box list."""
        for box in self.missing_boxes:
            self.add_box(box)
        self.missing_boxes = FrameBoxes(
            np.zeros((0, 4)),
            ptype=self.boxes.ptype,
            **self.boxes.metadata_kwargs,
        )

    @property
    def possible_boxes(self) -> FrameBoxes:
        """Return the list of all possible boxes."""
        return FrameBoxes(
            np.concatenate([self.boxes.array, self.missing_boxes.array]),
            ptype=self.boxes.ptype,
            **self.boxes.metadata_kwargs,
        )

    @property
    def previous_box(self) -> SingleBox:
        """Return the most recent addition."""
        if len(self.missing_boxes) > 0:
            return self.missing_boxes.get_single_box(-1)
        return self.boxes.get_single_box(-1)

    @property
    def n_missing(self) -> int:
        """Return the number of missing boxes."""
        return len(self.missing_boxes)