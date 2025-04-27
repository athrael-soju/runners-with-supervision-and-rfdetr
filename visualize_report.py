import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import random

def generate_color():
    """Generate a random color."""
    return (random.random(), random.random(), random.random())

def visualize_tracking_report(report_path="./output/tracking_report.csv", output_path="./output/tracking_visualization.png", confidence_threshold=0.5):
    """
    Create a Gantt-like visualization of object appearances in the video.
    
    Args:
        report_path: Path to the tracking report CSV
        output_path: Path to save the visualization image
        confidence_threshold: Confidence threshold used for detections
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read the tracking report
    df = pd.read_csv(report_path)
    
    if len(df) == 0:
        print("No tracking data found in the report.")
        return
    
    # Sort by class and duration
    df = df.sort_values(['Class', 'Duration (frames)'], ascending=[True, False])
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.4)))
    
    # Create a color map for different classes
    classes = df['Class'].unique()
    class_colors = {cls: generate_color() for cls in classes}
    
    # Plot each object as a horizontal bar
    y_ticks = []
    y_labels = []
    
    for i, (_, row) in enumerate(df.iterrows()):
        ax.barh(i, 
                row['Duration (frames)'], 
                left=row['First Frame'], 
                color=class_colors[row['Class']], 
                alpha=0.7,
                edgecolor='black')
        
        # Add object ID and duration as text
        text_x = row['First Frame'] + row['Duration (frames)'] / 2
        ax.text(text_x, i, f"ID: {int(row['Object ID'])} ({row['Duration (seconds)']}s)", 
                ha='center', va='center', fontsize=8)
        
        y_ticks.append(i)
        y_labels.append(f"{row['Class']}")
    
    # Create a legend for classes
    legend_patches = [mpatches.Patch(color=color, label=cls) 
                      for cls, color in class_colors.items()]
    ax.legend(handles=legend_patches, loc='upper right')
    
    # Set axes labels and title
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Frame Number')
    ax.set_title(f'Object Tracking Timeline (Confidence threshold: {confidence_threshold})')
    
    # Add grid lines for better readability
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")
    
    # Show additional statistics
    print("\nTracking Statistics:")
    stats = df.groupby('Class').agg({
        'Object ID': 'count',
        'Duration (seconds)': ['mean', 'max', 'min', 'sum']
    })
    print(stats)
    
    # Create a bar chart of count by class
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = df['Class'].value_counts()
    counts.plot(kind='bar', ax=ax, color=[class_colors[cls] for cls in counts.index])
    ax.set_title(f'Object Count by Class (Confidence threshold: {confidence_threshold})')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    
    # Save the count chart
    count_chart_path = os.path.join(os.path.dirname(output_path), 'class_count_chart.png')
    plt.tight_layout()
    plt.savefig(count_chart_path, dpi=300)
    print(f"Class count chart saved to {count_chart_path}")
    
    # Create a pie chart showing total duration by class
    fig, ax = plt.subplots(figsize=(10, 8))
    duration_by_class = df.groupby('Class')['Duration (seconds)'].sum()
    ax.pie(duration_by_class, labels=duration_by_class.index, autopct='%1.1f%%', 
           colors=[class_colors[cls] for cls in duration_by_class.index], shadow=True)
    ax.set_title(f'Total Screen Time by Class (Confidence threshold: {confidence_threshold})')
    
    # Save the pie chart
    pie_chart_path = os.path.join(os.path.dirname(output_path), 'screen_time_chart.png')
    plt.tight_layout()
    plt.savefig(pie_chart_path, dpi=300)
    print(f"Screen time chart saved to {pie_chart_path}")

if __name__ == "__main__":
    # Use same threshold as in main.py
    try:
        from main import CONFIDENCE_THRESHOLD
        visualize_tracking_report(confidence_threshold=CONFIDENCE_THRESHOLD)
    except ImportError:
        visualize_tracking_report() 