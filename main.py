import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import supervision as sv
from trackers import DeepSORTFeatureExtractor, DeepSORTTracker
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import csv
from collections import defaultdict

# Set confidence threshold
CONFIDENCE_THRESHOLD = 0.5

feature_extractor = DeepSORTFeatureExtractor.from_timm(
    model_name="mobilenetv4_conv_small.e1200_r224_in1k"
)
tracker = DeepSORTTracker(feature_extractor=feature_extractor)
model = RFDETRBase()
box_annotator = sv.BoxAnnotator()
annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

# For tracking objects' appearance duration
object_first_seen = {}  # {track_id: first_frame}
object_last_seen = {}   # {track_id: last_frame}
object_class = {}       # {track_id: class_name}
fps = None

def callback(frame, frame_index):
    global fps
    
    detections = model.predict(frame)
    
    # Filter detections by confidence threshold
    if detections.confidence is not None:
        mask = detections.confidence >= CONFIDENCE_THRESHOLD
        detections = detections[mask]
    
    detections = tracker.update(detections, frame)
    
    # Track objects
    if detections.tracker_id is not None:
        for track_id, class_id in zip(detections.tracker_id, detections.class_id):
            class_name = COCO_CLASSES.get(int(class_id), str(class_id))
            
            # Record object appearances
            if track_id not in object_first_seen:
                object_first_seen[track_id] = frame_index
                object_class[track_id] = class_name
            
            object_last_seen[track_id] = frame_index
    
    # Annotate the frame
    frame = box_annotator.annotate(frame, detections)
    if detections.class_id is not None:
        labels = []
        for track_id, class_id, confidence in zip(
            detections.tracker_id, detections.class_id, detections.confidence
        ):
            class_name = COCO_CLASSES.get(int(class_id), str(class_id))
            labels.append(f"{class_name} #{track_id} {confidence:.2f}")
    else:
        labels = None
    
    return annotator.annotate(frame, detections, labels=labels)

def generate_report(fps, output_path="./output/tracking_report.csv"):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Object ID', 'Class', 'First Frame', 'Last Frame', 'Duration (frames)', 'Duration (seconds)'])
        
        for track_id in sorted(object_first_seen.keys()):
            first_frame = object_first_seen[track_id]
            last_frame = object_last_seen[track_id]
            duration_frames = last_frame - first_frame + 1
            duration_seconds = duration_frames / fps
            
            writer.writerow([
                track_id,
                object_class[track_id],
                first_frame,
                last_frame,
                duration_frames,
                f"{duration_seconds:.2f}"
            ])
    
    print(f"Tracking report generated at {output_path}")
    print(f"Using confidence threshold: {CONFIDENCE_THRESHOLD}")

# Get video information and process
source_path = "./input/bikes-1280x720-2.mp4"
target_path = "./output/bikes-1280x720-2-output.mp4"

# First, get the video info to obtain FPS
video_info = sv.VideoInfo.from_video_path(source_path)
fps = video_info.fps
print(f"Video FPS: {fps}")
print(f"Using confidence threshold: {CONFIDENCE_THRESHOLD}")

# Process the video
sv.process_video(
    source_path=source_path,
    target_path=target_path,
    callback=callback,
)

# Generate tracking report
generate_report(fps)