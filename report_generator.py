import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

class ReportGenerator:
    def __init__(self, coco_classes):
        self.frame_counts = {}  # Frame number -> object counts
        self.object_tracks = defaultdict(list)  # Track ID -> list of frame appearances
        self.class_counts = Counter()  # Class ID -> total count
        self.track_class_map = {}  # Track ID -> class ID (to count unique objects)
        self.frame_number = 0
        self.track_history = defaultdict(list)  # Track ID -> list of positions
        self.coco_classes = coco_classes
        
    def update(self, detections, frame_number):
        # Update frame count
        self.frame_number = frame_number
        
        # Count objects per frame
        self.frame_counts[frame_number] = len(detections)
        
        # Track objects across frames and their classes
        for track_id, class_id, bbox in zip(
            detections.tracker_id, 
            detections.class_id, 
            detections.xyxy
        ):
            # Record this track's class if we haven't seen it before
            if track_id not in self.track_class_map:
                self.track_class_map[track_id] = class_id
                # Only increment class count once per unique tracked object
                self.class_counts[class_id] += 1
                
            self.object_tracks[track_id].append(frame_number)
            # Calculate center point of the bbox for trajectory tracking
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            self.track_history[track_id].append((center_x, center_y))
    
    def generate_report(self, output_dir="./report"):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate summary statistics
        total_unique_objects = len(self.object_tracks)
        max_objects_per_frame = max(self.frame_counts.values()) if self.frame_counts else 0
        avg_objects_per_frame = np.mean(list(self.frame_counts.values())) if self.frame_counts else 0
        
        # Class distribution (now based on unique tracked objects)
        class_distribution = {self.coco_classes[class_id]: count 
                             for class_id, count in self.class_counts.items()}
        
        # Generate track duration analysis
        track_durations = {track_id: len(frames) for track_id, frames in self.object_tracks.items()}
        avg_track_duration = np.mean(list(track_durations.values())) if track_durations else 0
        
        # Save reports
        with open(f"{output_dir}/summary_report.txt", "w") as f:
            f.write(f"Object Tracking Report\n")
            f.write(f"=====================\n\n")
            f.write(f"Total unique objects tracked: {total_unique_objects}\n")
            f.write(f"Maximum objects per frame: {max_objects_per_frame}\n")
            f.write(f"Average objects per frame: {avg_objects_per_frame:.2f}\n")
            f.write(f"Average track duration (frames): {avg_track_duration:.2f}\n\n")
            f.write(f"Class distribution (unique tracked objects):\n")
            for class_name, count in class_distribution.items():
                f.write(f"  - {class_name}: {count}\n")
        
        # Generate additional data about track class distribution
        track_classes = pd.DataFrame({
            "track_id": list(self.track_class_map.keys()),
            "class": [self.coco_classes[class_id] for class_id in self.track_class_map.values()],
            "duration": [len(self.object_tracks[track_id]) for track_id in self.track_class_map.keys()]
        })
        track_classes.to_csv(f"{output_dir}/track_classes.csv", index=False)
        
        # Generate dataframes and save to CSV
        track_df = pd.DataFrame({"track_id": list(track_durations.keys()),
                                 "duration": list(track_durations.values())})
        track_df.to_csv(f"{output_dir}/track_durations.csv", index=False)
        
        # Object count per frame
        frame_df = pd.DataFrame({"frame": list(self.frame_counts.keys()),
                                "object_count": list(self.frame_counts.values())})
        frame_df.to_csv(f"{output_dir}/object_counts_per_frame.csv", index=False)
        
        # Create plots
        self._generate_plots(output_dir)
        
        return f"Report generated in {output_dir}/"
    
    def _generate_plots(self, output_dir):
        # Plot 1: Objects per frame
        plt.figure(figsize=(10, 6))
        frames = list(self.frame_counts.keys())
        counts = list(self.frame_counts.values())
        plt.plot(frames, counts)
        plt.title("Objects Detected Per Frame")
        plt.xlabel("Frame Number")
        plt.ylabel("Number of Objects")
        plt.grid(True)
        plt.savefig(f"{output_dir}/objects_per_frame.png")
        plt.close()
        
        # Plot 2: Class distribution pie chart
        if self.class_counts:
            plt.figure(figsize=(10, 10))
            labels = [self.coco_classes[class_id] for class_id in self.class_counts.keys()]
            sizes = list(self.class_counts.values())
            plt.pie(sizes, labels=labels, autopct='%1.1f%%')
            plt.title("Distribution of Object Classes (Unique Tracked Objects)")
            plt.savefig(f"{output_dir}/class_distribution.png")
            plt.close()
        
        # Plot 3: Track trajectories (for up to 10 tracks to avoid overcrowding)
        plt.figure(figsize=(10, 8))
        colors = plt.cm.jet(np.linspace(0, 1, min(10, len(self.track_history))))
        for i, (track_id, positions) in enumerate(list(self.track_history.items())[:10]):
            if positions:
                xs, ys = zip(*positions)
                class_id = self.track_class_map.get(track_id)
                class_name = self.coco_classes[class_id] if class_id is not None else "Unknown"
                plt.plot(xs, ys, 'o-', color=colors[i], markersize=2, label=f"#{track_id} ({class_name})")
        plt.title("Object Trajectories (up to 10 tracks)")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
        plt.savefig(f"{output_dir}/trajectories.png")
        plt.close()
        
        # Plot 4: Track duration histogram
        if self.object_tracks:
            plt.figure(figsize=(10, 6))
            durations = [len(frames) for frames in self.object_tracks.values()]
            plt.hist(durations, bins=20)
            plt.title("Track Duration Distribution")
            plt.xlabel("Duration (frames)")
            plt.ylabel("Number of Tracks")
            plt.grid(True)
            plt.savefig(f"{output_dir}/track_durations.png")
            plt.close()

# Additional visualization functions can be added here
def generate_heatmap(detections_history, frame_shape, output_dir="./report"):
    """
    Generate a heatmap of object locations from detection history
    
    Args:
        detections_history: List of detections across frames
        frame_shape: (height, width) of video frames
        output_dir: Directory to save the heatmap
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a heatmap of all object locations
    heatmap = np.zeros((frame_shape[0], frame_shape[1]))
    
    # Populate the heatmap with detections
    for detections in detections_history:
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            # Ensure bounds are within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_shape[1]-1, x2), min(frame_shape[0]-1, y2)
            # Add to heatmap in the box area
            heatmap[y1:y2, x1:x2] += 1
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Object Frequency')
    plt.title('Object Detection Heatmap')
    plt.savefig(f"{output_dir}/detection_heatmap.png")
    plt.close()
    
    return f"{output_dir}/detection_heatmap.png" 