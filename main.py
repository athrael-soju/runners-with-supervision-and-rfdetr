import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import supervision as sv
from trackers import DeepSORTFeatureExtractor, DeepSORTTracker
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

feature_extractor = DeepSORTFeatureExtractor.from_timm(
    model_name="mobilenetv4_conv_small.e1200_r224_in1k"
)
tracker = DeepSORTTracker(feature_extractor=feature_extractor)
model = RFDETRBase()
box_annotator = sv.BoxAnnotator()
annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

def callback(frame, _):
    detections = model.predict(frame)
    detections = tracker.update(detections, frame)
    frame = box_annotator.annotate(frame, detections)
    if detections.class_id is not None:
        labels = [COCO_CLASSES.get(int(class_id), str(class_id)) for class_id in detections.class_id]
    else:
        labels = None
    return annotator.annotate(frame, detections, labels=labels)

sv.process_video(
    source_path="./input/bikes-1280x720-2.mp4",
    target_path="./output/bikes-1280x720-2-output.mp4",
    callback=callback,
)