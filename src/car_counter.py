#!/usr/bin/env python3
"""
Tracks vehicles in a video using a YOLO model and ByteTrack.

This script processes a video to detect and track vehicles. It's built to
handle challenges like temporary occlusions or fleeting detections.

A vehicle is considered "unique" and worth saving if it meets specific criteria:
- It's moving away from the camera and getting smaller.
- It occupies a significant portion of the screen.
- Or, as a fallback, if it appears in the bottom half of the frame.

To avoid saving the same car twice, a 3-second cooldown is used between saved events.
Clips and a summary report are saved for all unique vehicles found.
"""

import cv2
import os
import argparse
import csv
import json
import base64
import mimetypes
from tqdm import tqdm
import threading
from queue import Queue
import numpy as np
from openai import OpenAI
import torch

# Ensure ultralytics is installed
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: 'ultralytics' package not found.")
    print("Please install it with: pip install ultralytics")
    exit(1)


def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    try:
        # Infer MIME type, defaulting to JPEG if unknown.
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image'):
            mime_type = 'image/jpeg'

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def analyze_image_with_openai(image_path, client):
    """
    Analyzes a single image using OpenAI GPT-4o to find vehicle info.
    Returns a tuple: (extracted_vehicle_details_dict, raw_response_json_str)
    """
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        error_response = {"error": f"Failed to encode image {image_path}"}
        return None, json.dumps(error_response)

    prompt_text = '''
    Analyze the provided image to identify vehicle information.
    If a primary vehicle is visible, please provide the following details in a JSON format:
    - "vehicle_type": (e.g., "car", "truck", "motorcycle", "bus", "N/A" if not clear)
    - "make": The inferred make/brand of the vehicle (e.g., "Toyota", "Ford", "N/A").
    - "model": The inferred model of the vehicle (e.g., "Camry", "F-150", "N/A").
    - "color": The dominant color(s) of the vehicle (e.g., "Red", "Blue and White", "N/A").
    - "license_plate": The text of any visible license plate. If multiple, list them or pick the clearest. (e.g., "ABC 123", "N/A").
    - "poi": The state, country, or region that issued the license plate (e.g., "California", "Nevada", "Ontario", "N/A" if not identifiable).
    - "stickers": Does the vehicle have any visible stickers? (e.g., "yes", "no", "N/A").
    - "external_accessory": Describe any notable external accessories on the vehicle (e.g., "bike rack", "roof box", "running boards", "N/A" if none or not clear). Example: "bike rack".

    If no vehicle is clearly identifiable, or if specific details cannot be determined,
    please use "N/A" for the respective fields in the JSON structure.
    Ensure the output is ONLY a valid JSON object.
    '''

    extracted_details = {
        "vehicle_type": "N/A", "make": "N/A", "model": "N/A", "color": "N/A",
        "license_plate": "N/A", "poi": "N/A", "stickers": "N/A", "external_accessory": "N/A"
    }
    raw_response_content = None

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": base64_image}},
                ],}
            ],
            max_tokens=500,
            temperature=0.2,
        )
        
        raw_response_content = response.choices[0].message.content

        try:
            # The response can sometimes include markdown, so we strip it.
            if raw_response_content.strip().startswith("```json"):
                json_str = raw_response_content.strip()[7:-3].strip()
            elif raw_response_content.strip().startswith("```"):
                 json_str = raw_response_content.strip()[3:-3].strip()
            else:
                json_str = raw_response_content.strip()
            
            parsed_json = json.loads(json_str)
            for key in extracted_details:
                extracted_details[key] = parsed_json.get(key, "N/A")

        except json.JSONDecodeError as je:
            print(f"  Error: Could not parse JSON from OpenAI response: {je}")
        except Exception as e:
            print(f"  An unexpected error occurred during JSON parsing: {e}")

    except Exception as e:
        print(f"Error calling OpenAI API for {os.path.basename(image_path)}: {e}")
        error_response_obj = {"error": f"OpenAI API call failed: {str(e)}", "details_attempted": extracted_details}
        return None, json.dumps(error_response_obj)

    final_raw_response_str = raw_response_content if raw_response_content is not None else json.dumps({"error": "No response content received from API."})
    
    return extracted_details, final_raw_response_str


class VehicleTracker:
    """Handles the business logic for tracking a vehicle and deciding when to save its clip."""

    def __init__(self, output_dir, video_properties, video_filename, client):
        """
        Initializes the VehicleTracker.

        Args:
            output_dir (str): The directory where output clips will be saved.
            video_properties (dict): A dict with 'fps', 'codec', and 'frame_size' for the output videos.
            video_filename (str): The original video filename to extract naming components from.
            client: The OpenAI client instance for analysis.
        """
        self.output_dir = output_dir
        self.client = client
        
        # Set up the main output folder for this specific video.
        video_name = os.path.splitext(os.path.basename(video_filename))[0]
        self.video_output_dir = os.path.join(output_dir, video_name)

        # Create subfolders for counted cars and clips needing review.
        self.counted_cars_dir = os.path.join(self.video_output_dir, 'counted_cars')
        self.review_dir = os.path.join(self.video_output_dir, 'review')
        os.makedirs(self.counted_cars_dir, exist_ok=True)
        os.makedirs(self.review_dir, exist_ok=True)
        
        # Store video properties for writing new clips.
        self.video_properties = video_properties
        
        # Parse the video's filename to get a timestamp for naming.
        self.video_prefix, self.video_base_timestamp = self._extract_video_prefix(video_filename)
        
        # Get the total screen area to check if a car is large enough.
        frame_w, frame_h = video_properties['frame_size']
        self.total_screen_area = frame_w * frame_h
        self.min_area_threshold = self.total_screen_area * 0.1  # 10% of screen is a good starting point

        # Dictionaries to maintain the state of each track over time.
        self.track_positions = {}
        self.track_sizes = {}
        self.video_writers = {}
        self.frame_buffers = {}
        self.recording_started = {}
        self.total_cars_found = 0
        self.track_first_positions = {}
        self.track_bottom_history = {}
        self.saved_car_timestamps = []
        self.deduplication_window = 3.0  # Cooldown in seconds to prevent saving the same car twice
        self.active_clip_timestamps = {}
        self.clip_merge_window = 3.0
        
        self.position_history_size = 10
        self.min_frames_before_recording = 5
        
        # Grace period to avoid splitting a clip if a car is lost for a moment.
        self.single_track_grace_start_time = None
        self.grace_period_duration = 2.0
        
        # Threshold for the "bottom half of the screen" rule.
        self.bottom_half_threshold = frame_h * 0.5

        # State for the license plate detection and review process.
        self.lp_model = YOLO('license_plate_detector.pt')
        self.analysis_results = []
        self.post_detection_buffers = {}
        self.pre_detection_buffers = {}
        self.recent_frames_buffer = []
        self.valid_tracks = set()
        self.track_start_timestamps = {}

    def process_detections(self, detections, frame, timestamp, frame_number):
        """Processes a list of tracked detections for a single frame, managing video clip recording."""
        active_track_ids = {det.get('track_id') for det in detections}
        self.current_frame_detections = len(detections)

        # Keep a rolling buffer of the last few seconds of frames.
        self.recent_frames_buffer.append((frame, timestamp, frame_number))
        cutoff_timestamp = timestamp - 2.0
        self.recent_frames_buffer = [b for b in self.recent_frames_buffer if b[1] >= cutoff_timestamp]
            
        # Process all currently visible tracks.
        for det in detections:
            track_id = det.get('track_id')
            if track_id:
                self._process_single_track(track_id, det, frame, timestamp, frame_number)

        # Figure out which tracks disappeared in this frame.
        all_managed_tracks = set(self.frame_buffers.keys())
        lost_tracks = all_managed_tracks - active_track_ids
        
        # If a single valid car disappears, start a grace period to see if it comes back.
        if len(self.valid_tracks) == 1 and self.current_frame_detections == 0:
            if self.single_track_grace_start_time is None:
                self.single_track_grace_start_time = timestamp
        else:
            # Otherwise, no grace period is needed.
            self.single_track_grace_start_time = None

        writers_to_close = {tid for tid in lost_tracks if tid in self.valid_tracks}

        # If the grace period is active, don't close the clip yet.
        if self.single_track_grace_start_time is not None:
            time_in_grace = timestamp - self.single_track_grace_start_time
            if time_in_grace < self.grace_period_duration:
                writers_to_close.clear()
            else:
                # The grace period has expired, so we should close the clip.
                self.single_track_grace_start_time = None
        
        # Finalize and save clips for any tracks that are truly lost.
        for track_id in writers_to_close:
            # Grab the frames that occurred *after* the car was lost.
            if track_id not in self.post_detection_buffers:
                self.post_detection_buffers[track_id] = list(self.recent_frames_buffer)
            self._process_lost_track(track_id)
            
        # Get rid of any lost tracks that never met the save criteria.
        for track_id in lost_tracks:
            if track_id not in self.valid_tracks and track_id in self.frame_buffers:
                del self.frame_buffers[track_id]
                
    def _process_single_track(self, track_id, detection, frame, timestamp, frame_number):
        """Handles a single tracked vehicle each time it's detected in a frame."""
        bbox = detection['bbox']
        
        # Update the vehicle's position and size history.
        self._update_track_state(track_id, bbox)
        
        # Draw the box for visualization.
        frame_with_box = frame.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_with_box, f"ID: {track_id}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # If the tracker assigns a new ID to a car it's already tracking,
        # take over the old track's buffer and state.
        is_new_track = track_id not in self.frame_buffers
        
        if len(self.valid_tracks) == 1 and self.current_frame_detections == 1 and is_new_track:
            existing_track_id = list(self.valid_tracks)[0]
            
            # Transfer all state to the new track_id.
            if existing_track_id in self.frame_buffers:
                self.frame_buffers[track_id] = self.frame_buffers.pop(existing_track_id)
            if existing_track_id in self.pre_detection_buffers:
                 self.pre_detection_buffers[track_id] = self.pre_detection_buffers.pop(existing_track_id)
            if existing_track_id in self.track_start_timestamps:
                self.track_start_timestamps[track_id] = self.track_start_timestamps.pop(existing_track_id)
                 
            self.valid_tracks.remove(existing_track_id)
            self.valid_tracks.add(track_id)
            
        # If it's a new track, start a frame buffer for it.
        if track_id not in self.frame_buffers:
            self.pre_detection_buffers[track_id] = list(self.recent_frames_buffer)
            self.frame_buffers.setdefault(track_id, [])

        # Log the vehicle's initial position.
        if track_id not in self.track_first_positions:
            center_x, center_y = self._get_center(bbox)
            self.track_first_positions[track_id] = (center_x, center_y)
        
        # Note if the vehicle appears in the bottom half of the screen.
        x1, y1, x2, y2 = bbox
        if y2 >= self.bottom_half_threshold:
            self.track_bottom_history[track_id] = True
        
        # Add the current frame to this vehicle's buffer.
        self.frame_buffers.setdefault(track_id, []).append((frame.copy(), bbox, frame_number))

        # If we're already saving this track, just add the frame and move on.
        if track_id in self.valid_tracks:
            return

        # Decide if this track has become "valid" and should be saved.
        is_moving_correctly = self._is_moving_away_and_up(track_id)
        is_big_enough = self._is_large_enough(track_id)
        has_enough_buffer = len(self.frame_buffers.get(track_id, [])) >= self.min_frames_before_recording
        has_been_in_bottom_half = self._has_been_in_bottom_half(track_id)
        
        should_save = ((is_moving_correctly and is_big_enough) or has_been_in_bottom_half) and has_enough_buffer
        
        if should_save and track_id not in self.valid_tracks:
            self.valid_tracks.add(track_id)
            self.track_start_timestamps[track_id] = timestamp
            self.total_cars_found += 1
    
    def _process_lost_track(self, track_id):
        """
        Processes a track that has been lost. It runs license plate detection on its
        buffered frames. If a plate is found, it saves the best frame. If not, it saves
        a longer "review" clip.
        """
        # This track was never deemed valid, so just throw it away.
        if track_id not in self.valid_tracks:
            if track_id in self.frame_buffers: del self.frame_buffers[track_id]
            if track_id in self.pre_detection_buffers: del self.pre_detection_buffers[track_id]
            return

        buffered_data = self.frame_buffers.get(track_id, [])
        if not buffered_data:
            return

        best_frame_raw = None
        max_lp_conf = 0.0

        # Go through the buffered frames to find the one with the clearest license plate.
        for frame, _, _ in buffered_data:
            lp_results = self.lp_model(frame, verbose=False)[0]
            if lp_results.boxes:
                for box in lp_results.boxes:
                    if box.conf[0] > max_lp_conf:
                        max_lp_conf = box.conf[0]
                        best_frame_raw = frame

        timestamp_str = self._format_timestamp(self.track_start_timestamps.get(track_id, 0))
        
        if best_frame_raw is not None and max_lp_conf > 0.1:
            # Save the single best frame we found.
            filename = f"{timestamp_str}_car_{track_id}.jpg"
            output_path = os.path.join(self.counted_cars_dir, filename)
            cv2.imwrite(output_path, best_frame_raw)

            # Queue this image for later analysis by OpenAI.
            self.analysis_results.append({
                "filename": filename, 
                "status": "counted", 
                "path": output_path
            })

        else:
            # If no good license plate was found, save a "review" clip.
            pre_frames_data = self.pre_detection_buffers.get(track_id, [])
            car_frames_data = buffered_data
            after_frames_data = self.post_detection_buffers.get(track_id, [])

            # To build the review clip, figure out the start and end frames.
            first_car_frame_num = car_frames_data[0][2] if car_frames_data else -1
            last_car_frame_num = car_frames_data[-1][2] if car_frames_data else -1

            # Stitch together the pre-detection, detection, and post-detection frames.
            final_frames = [f for f, ts, fn in pre_frames_data if fn < first_car_frame_num]
            final_frames.extend([f for f, bbox, fn in car_frames_data])
            final_frames.extend([f for f, ts, fn in after_frames_data if fn > last_car_frame_num])
            
            if final_frames:
                filename = f"{timestamp_str}_car_{track_id}_review.mp4"
                output_path = os.path.join(self.review_dir, filename)
                writer = cv2.VideoWriter(output_path, 
                                         self.video_properties['codec'], 
                                         self.video_properties['fps'], 
                                         self.video_properties['frame_size'])
                for frame in final_frames:
                    writer.write(frame)
                writer.release()
                
                # Log the review clip for the final report.
                self.analysis_results.append({
                    "filename": filename, "status": "review", "license_plate": "NEEDS_REVIEW"
                })

        # Clear out all memory associated with this track ID.
        if track_id in self.valid_tracks: self.valid_tracks.remove(track_id)
        if track_id in self.frame_buffers: del self.frame_buffers[track_id]
        if track_id in self.pre_detection_buffers: del self.pre_detection_buffers[track_id]
        if track_id in self.post_detection_buffers: del self.post_detection_buffers[track_id]
        if track_id in self.track_start_timestamps: del self.track_start_timestamps[track_id]
        if track_id in self.track_first_positions: del self.track_first_positions[track_id]
        if track_id in self.track_bottom_history: del self.track_bottom_history[track_id]

    def finish_clip(self, track_id):
        """Processes a lost track to finalize its output."""
        self._process_lost_track(track_id)

    def _get_center(self, bbox):
        """Calculates the center coordinates of a bounding box."""
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _is_moving_away_and_up(self, track_id):
        """Checks if an object is trending away from the camera (smaller) and upwards."""
        positions = self.track_positions.get(track_id, [])
        sizes = self.track_sizes.get(track_id, [])
        
        history_needed = 4
        if len(positions) < history_needed:
            return False
            
        # Check the trend over the last few frames.
        
        # Positional Trend (is it moving up?)
        y_coords = [p[1] for p in positions[-history_needed:]]
        upward_steps = sum(1 for i in range(1, len(y_coords)) if y_coords[i] < y_coords[i-1])
        
        # Size Trend (is it getting smaller?)
        recent_sizes = sizes[-history_needed:]
        shrinking_steps = sum(1 for i in range(1, len(recent_sizes)) if recent_sizes[i] < recent_sizes[i-1])

        # A lenient check: just needs to show some movement in the right direction.
        has_some_upward_movement = upward_steps >= 1
        has_some_shrinking = shrinking_steps >= 1
        
        # Also check if the overall trend is correct (first vs. last point).
        overall_moving_up = y_coords[-1] < y_coords[0]
        overall_shrinking = recent_sizes[-1] < recent_sizes[0]
        
        return (has_some_upward_movement and has_some_shrinking) or (overall_moving_up and overall_shrinking)

    def _is_large_enough(self, track_id):
        """Checks if the car is currently, or ever was, larger than a certain screen area percentage."""
        sizes = self.track_sizes.get(track_id, [])
        if not sizes:
            return False
        
        current_area = sizes[-1]
        max_area = max(sizes)
        
        return current_area >= self.min_area_threshold or max_area >= self.min_area_threshold

    def _has_been_in_bottom_half(self, track_id):
        """Checks our history to see if this car ever appeared in the bottom half."""
        return self.track_bottom_history.get(track_id, False)

    def _is_duplicate_car(self, current_timestamp):
        """Checks if we just saved a car within the cooldown period."""
        # Clean up old timestamps that are outside the deduplication window.
        cutoff_time = current_timestamp - self.deduplication_window
        self.saved_car_timestamps = [ts for ts in self.saved_car_timestamps if ts >= cutoff_time]
        
        for saved_timestamp in self.saved_car_timestamps:
            if abs(current_timestamp - saved_timestamp) < self.deduplication_window:
                return True
        
        return False

    def _find_mergeable_clip(self, current_timestamp):
        """Finds an existing clip to merge with if a new car appears soon after."""
        for track_id, start_timestamp in self.active_clip_timestamps.items():
            if track_id in self.video_writers:
                time_diff = abs(current_timestamp - start_timestamp)
                if time_diff <= self.clip_merge_window:
                    return self.video_writers[track_id], track_id
        
        return None, None

    def _update_track_state(self, track_id, bbox):
        """Logs the latest position and size for a given track ID."""
        # Update positions, keeping only the last few.
        positions = self.track_positions.setdefault(track_id, [])
        positions.append(self._get_center(bbox))
        if len(positions) > self.position_history_size:
            positions.pop(0)
            
        # Update sizes, keeping only the last few.
        sizes = self.track_sizes.setdefault(track_id, [])
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        sizes.append(area)
        if len(sizes) > self.position_history_size:
            sizes.pop(0)
        
    def finish(self):
        """Called at the end of the video to process any remaining tracks."""
        for track_id in list(self.valid_tracks):
            # The "post" buffer is whatever was left in the recent frames buffer.
            self.post_detection_buffers[track_id] = list(self.recent_frames_buffer)
            self._process_lost_track(track_id)
        
        # Clear any remaining buffers that never met the save criteria.
        self.frame_buffers.clear()
        self.pre_detection_buffers.clear()
        
    def get_total_cars_found(self):
        """Returns the total number of unique cars that started recording."""
        return self.total_cars_found

    def _extract_video_prefix(self, video_filename):
        """
        Parses a filename like 'garage_lane1_20250624_000000' to get the prefix and timestamp.
        """
        import os
        import re
        from datetime import datetime
        
        base_name = os.path.splitext(os.path.basename(video_filename))[0]
        
        # Find a timestamp like YYYYMMDD_HHMMSS at the end of the filename.
        timestamp_pattern = r'(\d{8}_\d{6})$'
        match = re.search(timestamp_pattern, base_name)
        
        if match:
            timestamp_str = match.group(1)
            prefix = base_name[:match.start()].rstrip('_')
            
            try:
                dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                # Convert to seconds since midnight for easier time math.
                base_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
                return prefix, base_seconds
            except ValueError:
                return prefix, 0
        else:
            # If no timestamp is found, use the whole name as the prefix.
            return base_name, 0

    def _format_timestamp(self, timestamp_seconds):
        """
        Converts a video time offset (in seconds) into a HHMMSS timestamp string.
        """
        actual_seconds = self.video_base_timestamp + int(timestamp_seconds)
        
        # Handle cases where the video crosses midnight.
        actual_seconds = actual_seconds % (24 * 3600)
        
        hours = int(actual_seconds // 3600)
        minutes = int((actual_seconds % 3600) // 60)
        seconds = int(actual_seconds % 60)
        
        return f"{hours:02d}-{minutes:02d}-{seconds:02d}"

def run_tracker(video_path, model_path, output_dir, conf_thresh):
    """
    Main function to run the object tracking pipeline on a video.

    Args:
        video_path (str): Path to the input video file.
        model_path (str): Path to the YOLOv11 model file (e.g., 'yolo11m.pt').
        output_dir (str): Directory to save the output images.
        conf_thresh (float): Confidence threshold for object detection.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return
    if not os.path.exists('bytetrack.yaml'):
        print("Error: 'bytetrack.yaml' not found in the current directory.")
        return
    
    # Initialize the OpenAI client if an API key is available.
    openai_api_key = os.getenv('OPENAI_API_KEY')
    client = None
    if openai_api_key:
        try:
            client = OpenAI(api_key=openai_api_key)
            print("OpenAI client initialized.")
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI client: {e}")
    else:
        print("Warning: OPENAI_API_KEY not set. Skipping OpenAI analysis.")
    
    try:
        model = YOLO(model_path)
        # Ultralytics will automatically use CUDA or MPS if available.
        print("YOLO model loaded. Device will be selected automatically (CUDA/MPS/CPU).")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    
    # We only care about tracking vehicle-like objects.
    # These are the class IDs from the COCO dataset.
    classes_to_track = [2, 7, 1, 3] # car, truck, bicycle, motorcycle

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set up the properties for any new video clips we create.
    video_properties = {
        'fps': fps,
        'codec': cv2.VideoWriter_fourcc(*'mp4v'),
        'frame_size': (frame_w, frame_h)
    }

    tracker_manager = VehicleTracker(output_dir, video_properties, video_path, client)
    
    print("Starting object tracking...")

    # Frame skipping helps us process videos faster.
    # Skip more frames when nothing is happening, fewer when cars are present.
    fast_skip = max(1, int(fps * 0.3))
    slow_skip = max(1, int(fps * 0.1))
    current_skip = fast_skip
    
    print(f"Video FPS: {fps:.1f}, Fast skip: {fast_skip} frames, Slow skip: {slow_skip} frames")
    frame_number = 0

    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while frame_number < total_frames:
            # Only read the full frame if we're going to process it.
            if frame_number % current_skip == 0:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_number / fps
                
                # Get detections and track IDs for the current frame.
                tracked_results = model.track(
                    source=frame,
                    tracker='bytetrack.yaml',
                    persist=True,
                    verbose=False,
                    conf=conf_thresh,
                    imgsz=640,
                    half=True
                )
                
                detections = []
                if tracked_results and getattr(tracked_results[0].boxes, 'id', None) is not None:
                    for box in tracked_results[0].boxes:
                        if int(box.cls[0]) in classes_to_track:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'conf': float(box.conf[0]),
                                'track_id': int(box.id[0])
                            })

                # Send the latest tracking info to our manager class.
                tracker_manager.process_detections(detections, frame, timestamp, frame_number)

                # Slow down processing if there are potential cars.
                if detections:
                    current_skip = slow_skip
                else:
                    current_skip = fast_skip
                
                pbar.set_postfix({
                    'active_tracks': len(tracker_manager.valid_tracks),
                    'tot_cars': tracker_manager.total_cars_found,
                    'skip_rate': current_skip
                })
            else:
                # Even if we skip processing, we have to advance the video stream.
                ret = cap.grab()
                if not ret:
                    break

            pbar.update(1)
            frame_number += 1
    
    cap.release()
    tracker_manager.finish()
    
    # After tracking, send saved images to OpenAI for analysis.
    if client and any(r.get('status') == 'counted' for r in tracker_manager.analysis_results):
        print("\n" + "="*30)
        print("   Analyzing Counted Images")
        print("="*30)
        
        counted_entries = [r for r in tracker_manager.analysis_results if r.get('status') == 'counted']
        
        with tqdm(total=len(counted_entries), desc="Analyzing with OpenAI") as pbar:
            for entry in counted_entries:
                path = entry.get('path')
                if path and os.path.exists(path):
                    details, raw_response = analyze_image_with_openai(path, client)
                    entry['raw_response'] = raw_response
                    if details:
                        entry.update(details)
                pbar.update(1)

    # Print a final report.
    total_cars_found = len(tracker_manager.analysis_results)
    counted_cars_count = len([r for r in tracker_manager.analysis_results if 'counted' in r.get('status', '')])
    review_clips_count = len([r for r in tracker_manager.analysis_results if r.get('status') == 'review'])

    print("\n" + "="*30)
    print("      Tracking Complete")
    print("="*30)
    print(f"Total unique vehicles processed: {total_cars_found}")
    print(f"  - Images saved in 'counted_cars': {counted_cars_count}")
    print(f"  - Review clips saved in 'review': {review_clips_count}")
    print(f"Results saved in: '{tracker_manager.video_output_dir}'")

    # Write a summary of the results to a CSV file.
    if tracker_manager.analysis_results:
        csv_path = os.path.join(tracker_manager.video_output_dir, 'vehicle_analysis_results.csv')
        csv_fieldnames = [
            "filename", "status", "vehicle_type", "make", "model", "color", 
            "license_plate", "poi", "stickers", "external_accessory", "raw_response"
        ]
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                summary_writer = csv.writer(f)
                summary_writer.writerow(['Total Vehicles', total_cars_found])
                summary_writer.writerow(['Found LPs', counted_cars_count])
                summary_writer.writerow(['Needs Review', review_clips_count])
                summary_writer.writerow([])

                writer = csv.DictWriter(f, fieldnames=csv_fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(tracker_manager.analysis_results)
            print(f"\nAnalysis summary saved to '{csv_path}'")
        except IOError as e:
            print(f"\nError writing summary CSV: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 and ByteTrack Object Tracking Pipeline")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--model-path", type=str, default="yolo11m.pt", help="Path to the YOLOv11 model file.")
    parser.add_argument("--output-dir", type=str, default="tracking_output", help="Directory to save the output images.")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for object detection.")
    
    args = parser.parse_args()
    
    run_tracker(args.video_path, args.model_path, args.output_dir, args.conf) 