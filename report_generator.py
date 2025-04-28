import os
import csv
import json
import time
from datetime import timedelta
import numpy as np
import supervision as sv
from collections import defaultdict

try:
    from rfdetr.util.coco_classes import COCO_CLASSES
except ImportError:
    # Fallback if COCO_CLASSES can't be imported
    COCO_CLASSES = {}


class TrackingReporter:
    def __init__(self, source_video_path):
        self.source_video_path = source_video_path
        self.video_name = os.path.basename(source_video_path).split(".")[0]
        self.reports_dir = "reports"

        # Ensure reports directory exists
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)

        # Initialize tracking data
        self.tracking_data = defaultdict(
            lambda: {
                "class_id": None,
                "class_name": None,
                "first_seen": None,
                "last_seen": None,
                "total_frames": 0,
                "bounding_boxes": [],
                "confidence_scores": [],
            }
        )

        # Get video FPS for time calculations
        self.fps = self._get_video_fps()

    def _get_video_fps(self):
        """Get the FPS of the video or default to 30 if not available"""
        try:
            import cv2

            cap = cv2.VideoCapture(self.source_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return fps if fps > 0 else 30.0
        except:
            return 30.0  # Default FPS if unable to determine

    def update(self, detections, frame_index):
        """Update tracking data with new detections from the current frame"""
        for i in range(len(detections)):
            track_id = detections.tracker_id[i]

            # Skip tracks that don't have an ID yet
            if track_id == -1:
                continue

            class_id = detections.class_id[i]
            confidence = detections.confidence[i]
            bbox = detections.xyxy[i]

            # Update tracking info
            if self.tracking_data[track_id]["class_id"] is None:
                self.tracking_data[track_id]["class_id"] = class_id
                self.tracking_data[track_id]["class_name"] = COCO_CLASSES.get(
                    class_id, f"class_{class_id}"
                )
                self.tracking_data[track_id]["first_seen"] = frame_index

            self.tracking_data[track_id]["last_seen"] = frame_index
            self.tracking_data[track_id]["total_frames"] += 1
            self.tracking_data[track_id]["bounding_boxes"].append(bbox.tolist())
            self.tracking_data[track_id]["confidence_scores"].append(float(confidence))

    def generate_reports(self):
        """Generate various reports based on collected tracking data"""
        # Ensure reports directory exists before generating reports
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)
            print(f"Created reports directory: {self.reports_dir}")
            
        # Generate summary report
        self._generate_summary_report(f"{self.reports_dir}/summary.csv")

        # Generate detailed JSON report
        self._generate_detailed_report(f"{self.reports_dir}/report.json")

        print(f"Reports generated in the '{self.reports_dir}' directory")

    def _generate_summary_report(self, filepath):
        """Generate a CSV summary report with basic tracking statistics"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Track ID",
                    "Class",
                    "First Seen (frame)",
                    "Last Seen (frame)",
                    "Duration (frames)",
                    "Duration (seconds)",
                    "Avg Confidence",
                ]
            )

            for track_id, data in self.tracking_data.items():
                duration_frames = data["total_frames"]
                duration_seconds = duration_frames / self.fps
                avg_confidence = (
                    np.mean(data["confidence_scores"])
                    if data["confidence_scores"]
                    else 0
                )

                writer.writerow(
                    [
                        track_id,
                        data["class_name"],
                        data["first_seen"],
                        data["last_seen"],
                        duration_frames,
                        f"{duration_seconds:.2f}",
                        f"{avg_confidence:.4f}",
                    ]
                )

    def _generate_detailed_report(self, filepath):
        """Generate a detailed JSON report with full tracking data"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        report_data = {
            "video_info": {
                "filename": self.video_name,
                "fps": self.fps,
                "path": self.source_video_path,
            },
            "tracks": {},
        }

        for track_id, data in self.tracking_data.items():
            # Calculate additional metrics
            duration_frames = data["total_frames"]
            duration_seconds = duration_frames / self.fps

            track_data = {
                "class_id": (
                    int(data["class_id"]) if data["class_id"] is not None else None
                ),
                "class_name": data["class_name"],
                "first_seen_frame": data["first_seen"],
                "last_seen_frame": data["last_seen"],
                "total_frames": duration_frames,
                "duration_seconds": duration_seconds,
                "duration_formatted": str(timedelta(seconds=duration_seconds)),
                "avg_confidence": (
                    float(np.mean(data["confidence_scores"]))
                    if data["confidence_scores"]
                    else 0
                ),
                "max_confidence": (
                    float(np.max(data["confidence_scores"]))
                    if data["confidence_scores"]
                    else 0
                ),
                "min_confidence": (
                    float(np.min(data["confidence_scores"]))
                    if data["confidence_scores"]
                    else 0
                ),
            }

            report_data["tracks"][str(track_id)] = track_data

        with open(filepath, "w") as f:
            json.dump(report_data, f, indent=2)


# Function to print a simple console summary
def print_tracking_summary(reporter):
    """Print a simple summary of tracking results to the console"""
    print("\n=== TRACKING SUMMARY ===")
    print(f"Video: {reporter.video_name}")
    print(f"Total unique tracked objects: {len(reporter.tracking_data)}")

    # Count by class
    class_counts = defaultdict(int)
    for track_id, data in reporter.tracking_data.items():
        class_counts[data["class_name"]] += 1

    print("\nObjects by class:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")

    # Calculate avg duration
    durations = [
        data["total_frames"] / reporter.fps for data in reporter.tracking_data.values()
    ]
    if durations:
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        print(f"\nAverage tracking duration: {avg_duration:.2f} seconds")
        print(f"Longest tracking duration: {max_duration:.2f} seconds")

    print("======================\n")
