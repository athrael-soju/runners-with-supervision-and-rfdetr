import subprocess
import os
import time

def run_tracker_and_visualize():
    """
    Runs the tracking script and then visualizes the results.
    """
    print("Starting video tracking...")
    tracker_start = time.time()
    
    # Run the main tracking script
    subprocess.run(["python", "main.py"], check=True)
    
    tracker_end = time.time()
    print(f"Tracking completed in {tracker_end - tracker_start:.2f} seconds")
    
    # Check if the report was generated
    report_path = "./output/tracking_report.csv"
    if not os.path.exists(report_path):
        print(f"Error: Tracking report not found at {report_path}")
        return
    
    print("\nGenerating visualizations...")
    viz_start = time.time()
    
    # Run the visualization script
    subprocess.run(["python", "visualize_report.py"], check=True)
    
    viz_end = time.time()
    print(f"Visualization completed in {viz_end - viz_start:.2f} seconds")
    
    print("\nAll processing complete!")
    print("Generated files:")
    print("1. Annotated video: ./output/bikes-1280x720-2-output.mp4")
    print("2. Tracking report: ./output/tracking_report.csv")
    print("3. Timeline visualization: ./output/tracking_visualization.png")
    print("4. Class count chart: ./output/class_count_chart.png")
    print("5. Screen time chart: ./output/screen_time_chart.png")

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("./output", exist_ok=True)
    run_tracker_and_visualize() 