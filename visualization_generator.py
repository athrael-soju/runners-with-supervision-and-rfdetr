import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import pandas as pd
from collections import defaultdict
import seaborn as sns


class ReportVisualizer:
    def __init__(self, report_path="reports/report.json", output_dir="visualizations"):
        """
        Initialize the visualizer with a path to the report JSON file

        Args:
            report_path: Path to the report JSON file
            output_dir: Directory to save visualizations
        """
        self.report_path = report_path
        self.output_dir = output_dir

        # Load report data
        with open(report_path, "r") as f:
            self.report_data = json.load(f)

        self.video_name = self.report_data["video_info"]["filename"]
        self.fps = self.report_data["video_info"]["fps"]

        # Parse tracking data
        self.tracking_data = {}
        for track_id, data in self.report_data["tracks"].items():
            self.tracking_data[track_id] = data

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_all_visualizations(self):
        """Generate all available visualizations"""
        print(f"Generating visualizations for {self.video_name}...")

        # Generate basic visualizations
        self.plot_objects_by_class()
        self.plot_tracking_duration_histogram()
        self.plot_object_timeline()
        self.plot_confidence_distribution()

        print(f"Visualizations generated")

    def plot_objects_by_class(self):
        """Generate a bar chart of object counts by class"""
        # Count objects by class
        class_counts = defaultdict(int)
        for track_id, data in self.tracking_data.items():
            class_counts[data["class_name"]] += 1

        # Sort classes by count (descending)
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        classes, counts = zip(*sorted_classes) if sorted_classes else ([], [])

        # Create the plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            classes, counts, color=sns.color_palette("viridis", len(classes))
        )

        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        plt.title(f"Objects Detected by Class - {self.video_name}")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"{self.output_dir}/objects_by_class.png", dpi=300)
        plt.close()

    def plot_tracking_duration_histogram(self):
        """Create a histogram of tracking durations"""
        # Extract tracking durations in seconds
        durations = [data["duration_seconds"] for data in self.tracking_data.values()]

        if not durations:
            print("No tracking data available for duration histogram")
            return

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Create histogram with automatic bin determination
        n, bins, patches = plt.hist(
            durations, bins="auto", color="skyblue", edgecolor="black", alpha=0.7
        )

        # Add a kernel density estimate
        if len(durations) > 1:  # KDE requires at least 2 points
            sns.kdeplot(durations, color="red", label="Density")

        # Add vertical lines for mean and median
        plt.axvline(
            np.mean(durations),
            color="r",
            linestyle="dashed",
            linewidth=1,
            label=f"Mean: {np.mean(durations):.2f}s",
        )
        plt.axvline(
            np.median(durations),
            color="g",
            linestyle="dashed",
            linewidth=1,
            label=f"Median: {np.median(durations):.2f}s",
        )

        plt.title(f"Distribution of Object Tracking Durations - {self.video_name}")
        plt.xlabel("Duration (seconds)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"{self.output_dir}/duration_histogram.png", dpi=300)
        plt.close()

    def plot_object_timeline(self):
        """Create a timeline showing when each object appears in the video"""
        # Sort tracks by first appearance
        tracks_sorted = sorted(
            self.tracking_data.items(), key=lambda x: float(x[1]["first_seen_frame"])
        )

        if not tracks_sorted:
            print("No tracking data available for timeline visualization")
            return

        # Get fps and calculate video duration
        fps = self.fps
        max_frame = max([float(data["last_seen_frame"]) for _, data in tracks_sorted])
        video_duration = max_frame / fps

        # Setup the plot
        plt.figure(figsize=(12, 8))

        # Define colormap based on object classes
        classes = {data["class_name"] for _, data in tracks_sorted}
        class_colors = dict(zip(classes, sns.color_palette("husl", len(classes))))

        # Plot each track as a horizontal line
        y_ticks = []
        y_positions = []

        for i, (track_id, data) in enumerate(tracks_sorted):
            start_time = float(data["first_seen_frame"]) / fps
            end_time = float(data["last_seen_frame"]) / fps
            duration = end_time - start_time

            plt.plot(
                [start_time, end_time],
                [i, i],
                linewidth=6,
                solid_capstyle="butt",
                color=class_colors[data["class_name"]],
            )

            # Add track info to y-axis labels
            y_ticks.append(f"#{track_id} ({data['class_name']})")
            y_positions.append(i)

        # Set labels and title
        plt.yticks(y_positions, y_ticks)
        plt.xlabel("Time (seconds)")
        plt.title(f"Object Appearance Timeline - {self.video_name}")

        # Add grid lines
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        # Add legend for classes
        legend_elements = [
            plt.Line2D([0], [0], color=color, lw=4, label=class_name)
            for class_name, color in class_colors.items()
        ]
        plt.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()

        # Save the figure
        plt.savefig(f"{self.output_dir}/object_timeline.png", dpi=300)
        plt.close()

    def plot_confidence_distribution(self):
        """Plot the distribution of confidence scores by class"""
        # Collect confidence scores by class
        class_confidences = defaultdict(list)

        for track_id, data in self.tracking_data.items():
            class_name = data["class_name"]
            # Use the avg_confidence value since we don't have individual confidence scores in the JSON
            confidences = [data["avg_confidence"]]
            class_confidences[class_name].extend(confidences)

        if not class_confidences:
            print("No confidence data available")
            return

        # Set up the plot
        plt.figure(figsize=(12, 6))

        # Create violin plots for each class
        classes = list(class_confidences.keys())
        confidence_data = [class_confidences[cls] for cls in classes]

        # If we have enough data points, create violin plots
        if sum(len(conf) for conf in confidence_data) > 3:
            parts = plt.violinplot(confidence_data, showmeans=True, showmedians=True)

            # Customize violin plot appearance
            for pc in parts["bodies"]:
                pc.set_facecolor("skyblue")
                pc.set_edgecolor("black")
                pc.set_alpha(0.7)

        # Add box plots inside or instead of violins if not enough data for violin
        plt.boxplot(
            confidence_data,
            labels=classes,
            patch_artist=True,
            boxprops=dict(facecolor="lightgreen", alpha=0.5),
            medianprops=dict(color="red"),
            vert=True,
        )

        # Add scatter points for small datasets
        for i, (cls, conf_values) in enumerate(zip(classes, confidence_data)):
            # Add jitter to x positions for better visibility
            x_jitter = np.random.normal(i + 1, 0.04, size=len(conf_values))
            plt.scatter(x_jitter, conf_values, color="blue", alpha=0.5)

        plt.title(f"Confidence Score Distribution by Class - {self.video_name}")
        plt.ylabel("Confidence Score")
        plt.ylim(0, 1.05)  # Confidence scores are between 0 and 1
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"{self.output_dir}/confidence_distribution.png", dpi=300)
        plt.close()


def generate_visualizations(report_path="reports/report.json"):
    """Generate all visualizations for the given report file"""
    visualizer = ReportVisualizer(report_path)
    visualizer.generate_all_visualizations()
    return visualizer.output_dir


if __name__ == "__main__":
    # When the script is run directly, generate visualizations from the default report
    output_dir = generate_visualizations()
