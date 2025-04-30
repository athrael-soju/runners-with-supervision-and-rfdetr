import supervision as sv
from supervision.utils.video import VideoInfo, VideoSink, get_video_frames_generator
import os
import cv2
import numpy as np


class VideoProcessor:
    def __init__(
        self,
        source_path,
        target_path=None,
        model=None,
        tracker=None,
        confidence_threshold=0.5,
        reporter=None,
    ):
        """
        Initialize the video processor

        Args:
            source_path: Path to the source video
            target_path: Path to the output video (if None, will use ./output/[source_filename])
            model: Object detection model
            tracker: Object tracker
            confidence_threshold: Confidence threshold for detections
            reporter: Reporter for tracking statistics (optional)
        """
        self.source_path = source_path

        # Set default target path if not provided
        if target_path is None:
            self.target_path = "./output/" + source_path.split("/")[-1]
        else:
            self.target_path = target_path

        self.model = model
        self.tracker = tracker
        self.confidence_threshold = confidence_threshold
        self.reporter = reporter

        # Create annotators
        self.box_annotator = None
        self.label_annotator = None
        if model is not None:
            self._setup_annotators()

    def _setup_annotators(self):
        """Set up the box and label annotators"""
        color = sv.ColorPalette.from_hex(
            [
                "#ffff00",
                "#ff9b00",
                "#ff8080",
                "#ff66b2",
                "#ff66ff",
                "#b266ff",
                "#9999ff",
                "#3399ff",
                "#66ffff",
                "#33ff99",
                "#66ff66",
                "#99ff00",
            ]
        )

        self.box_annotator = sv.BoxAnnotator(
            color=color, color_lookup=sv.ColorLookup.TRACK
        )
        self.label_annotator = sv.LabelAnnotator(
            color=color,
            color_lookup=sv.ColorLookup.TRACK,
            text_color=sv.Color.BLACK,
            text_scale=1.0,
        )

    def callback(self, frame, index):
        """
        Process a single frame

        Args:
            frame: The frame to process
            index: The frame index

        Returns:
            The processed frame with annotations
        """
        if self.model is None or self.tracker is None:
            return frame

        # Obtain bounding box predictions from model
        detections = self.model.predict(frame, threshold=self.confidence_threshold)

        # Update tracker with new detections and retrieve updated IDs
        detections = self.tracker.update(detections, frame)

        # Filter out detections with IDs of -1 (fresh tracks not yet confirmed)
        detections = detections[detections.tracker_id != -1]

        # Update tracking reporter if enabled
        if self.reporter is not None:
            self.reporter.update(detections, index)

        # Get classes dictionary from the model if available
        classes_dict = getattr(self.model, "classes_dict", None)

        if classes_dict:
            # Add class names to labels
            labels = [
                f"#{t_id} | {classes_dict[class_id]}"
                for t_id, class_id in zip(detections.tracker_id, detections.class_id)
            ]
        else:
            # Use just IDs if classes dictionary not available
            labels = [f"#{t_id}" for t_id in detections.tracker_id]

        annotated_image = frame.copy()
        annotated_image = self.box_annotator.annotate(annotated_image, detections)
        annotated_image = self.label_annotator.annotate(
            annotated_image, detections, labels
        )

        return annotated_image

    def process_video(self, seconds=None, output_fps=None, start_frame=0):
        """
        Process the video with the given parameters

        Args:
            seconds: Number of seconds to process (None for entire video)
            output_fps: Output FPS (None for same as input)
            start_frame: Frame to start processing from

        Returns:
            Path to the processed video
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(self.target_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Check video frame count and FPS
        video_info = sv.VideoInfo.from_video_path(self.source_path)
        input_fps = video_info.fps
        total_frames = video_info.total_frames

        # Fallback for when total_frames is None
        if total_frames is None:
            video = cv2.VideoCapture(self.source_path)
            if video.isOpened():
                total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                video.release()
            else:
                print(
                    "Warning: Could not determine total frame count, processing the entire video"
                )
                total_frames = float("inf")

        # Handle None value for start_frame
        if start_frame is None:
            start_frame = 0

        # Validate start_frame
        start_frame = max(
            0, min(start_frame, total_frames - 1 if total_frames != float("inf") else 0)
        )

        # Calculate frames to process based on seconds
        end_frame = None
        if seconds is not None:
            end_frame = min(start_frame + int(seconds * input_fps), total_frames)

        # Calculate stride to adjust output FPS if needed
        stride = 1
        if output_fps is not None and output_fps < input_fps:
            stride = max(1, round(input_fps / output_fps))
            print(f"Using stride of {stride} to achieve target FPS")

        # Calculate effective output FPS
        effective_fps = input_fps / stride
        fps_to_use = effective_fps if output_fps is not None else input_fps

        frames_to_process = total_frames if end_frame is None else end_frame
        print(f"Input video: {total_frames} frames at {input_fps} FPS")
        print(
            f"Processing frames from {start_frame} to {frames_to_process}, output at {fps_to_use:.2f} FPS"
        )

        # Use supervision's VideoSink and frame generator
        with sv.VideoSink(
            self.target_path,
            video_info=sv.VideoInfo(
                width=video_info.width,
                height=video_info.height,
                fps=fps_to_use,
                total_frames=None,
            ),
        ) as sink:

            # Process frames using the generator with stride and range control
            for frame_index, frame in enumerate(
                sv.get_video_frames_generator(
                    source_path=self.source_path,
                    stride=stride,
                    start=start_frame,
                    end=end_frame,
                )
            ):
                # Calculate absolute frame index
                absolute_frame_idx = start_frame + (frame_index * stride)

                # Process frame with callback
                processed_frame = self.callback(frame, absolute_frame_idx)
                sink.write_frame(processed_frame)

                print(f"Processed frame {absolute_frame_idx}/{frames_to_process}")

        print(f"Completed processing video to {self.target_path}")
        return self.target_path
