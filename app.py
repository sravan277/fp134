# IMPORTANT: Eventlet monkey patching must happen FIRST
import eventlet
eventlet.monkey_patch()

# Now import other modules
import time
import math
import threading
import cv2
import numpy as np
from flask import send_from_directory
from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException
from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
from pymavlink import mavutil
import os
import gc
import psutil
import signal # For graceful shutdown

# --- Jetson Optimization Imports ---
try:
    # Enable GPU acceleration if available
    import tensorrt as trt
    TRT_AVAILABLE = True
    print("TensorRT available for GPU acceleration")
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT not available, using CPU inference")

# --- Flask App Setup ---
app = Flask(__name__)
# IMPORTANT: Change this secret key for production environments!
app.config['SECRET_KEY'] = 'jetson_mango_drone_2024_secret_key!'
# Use eventlet as the async mode
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# --- Global Variables ---
vehicle = None
telemetry_data = {}
update_interval = 1.0 # Telemetry update interval in seconds
running = True  # Flag to control background threads

# --- Global Variables ---
# ... (existing globals)
video_writer = None
is_recording = False
current_recording_filename = None
RECORDING_PATH = "recordings"  # Folder to store recordings on Jetson
# ...
# --- Computer Vision Variables ---
cap = None  # Camera capture object
frame = None  # Current frame for streaming
mango_detected = False # Overall state of mango detection
mango_position = (0, 0)  # (x, y) center of detected mango
frame_center = (0, 0)  # Will be calculated when camera starts
mango_tracking_active = False # Master switch for mango tracking behavior
cv_lock = threading.Lock()  # Lock for thread-safe access to video frames and CV variables

# --- Jetson Optimized Parameters ---
# Camera settings optimized for Jetson
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_BUFFER_SIZE = 1 # Minimal buffer size for fresh frames

search_state = "idle"  # idle, waiting, rotating, moving_forward
search_start_time = 0
rotation_start_time = 0
forward_movement_start_time = 0
current_search_yaw = 0
total_rotation_degrees = 0
forward_movement_distance = 0
# New global variables for incremental rotation
ROTATION_INCREMENT_DEGREES = 20.0  # Rotate 20 degrees at a time
DETECTION_WAIT_TIME = 10.0         # Wait 10 seconds at each position for detection
incremental_rotation_target = 0    # Target degrees for current rotation increment
detection_wait_start_time = 0      # Time when detection wait period started
current_rotation_degrees = 0       # Track total rotation completed
# Detection parameters optimized for real-time performance
# Search parameters
SEARCH_WAIT_TIME = 6.0  # Wait 15 seconds before starting search
ROTATION_SPEED_DEG_S = 40.0  # Rotation speed in degrees per second
FULL_ROTATION_DEGREES = 360.0  # Complete 360-degree rotation
FORWARD_MOVEMENT_SPEED = 0.5  # Forward movement speed in m/s
FORWARD_MOVEMENT_DISTANCE = 1.0  # Move 1 meter forward
FORWARD_MOVEMENT_TIME = FORWARD_MOVEMENT_DISTANCE / FORWARD_MOVEMENT_SPEED 

DETECTION_CONFIDENCE_THRESHOLD = 0.6
MOVEMENT_DEAD_ZONE_PERCENTAGE = 0.12
MAX_VELOCITY = 0.3 # Adjusted for smoother movements
VELOCITY_SCALING_FACTOR = 0.008 # Adjusted for Jetson performance
ALTITUDE_HOLD_VELOCITY = 0.0 # vz, typically managed by flight controller in guided
YAW_CONTROL_ENABLED = True
YAW_SCALING_FACTOR = 0.020 # Adjusted yaw rate

# Jetson-specific optimizations
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame for better performance
MAX_MANGO_LOST_FRAMES = 20  # Increased buffer for tracking stability
MEMORY_CLEANUP_INTERVAL = 100  # Clean memory every 100 frames

# Tracking state variables
last_mango_position = (0, 0)
mango_lost_counter = 0
frame_count = 0

# --- Jetson Camera Initialization ---
def get_jetson_camera_pipeline(camera_id=0, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS):
    """
    Create GStreamer pipeline for Jetson camera (CSI or USB)
    """
    csi_pipeline = (
        f"nvarguscamerasrc sensor-id={camera_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method=0 ! " # flip-method=0 for no flip, 2 for vertical flip
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=1"
    )
    usb_pipeline = (
        f"v4l2src device=/dev/video{camera_id} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, framerate=(fraction){fps}/1 ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=1"
    )
    return csi_pipeline, usb_pipeline

def initialize_jetson_camera():
    """Initialize camera with Jetson-optimized settings."""
    global cap, frame_center
    print("Initializing Jetson camera...")
    csi_pipeline, usb_pipeline = get_jetson_camera_pipeline()

    try:
        print("Attempting CSI camera connection...")
        cap = cv2.VideoCapture(csi_pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print("CSI camera failed, attempting GStreamer USB camera connection...")
            cap = cv2.VideoCapture(usb_pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                print("GStreamer USB failed, trying direct USB access (camera 0)...")
                cap = cv2.VideoCapture(0) # Try default USB camera
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
                    print("✓ Direct USB camera initialized successfully.")
                else:
                    raise Exception("All camera initialization attempts failed.")
            else:
                print("✓ GStreamer USB camera initialized successfully.")
        else:
            print("✓ CSI camera initialized successfully.")

        # Test camera and get frame dimensions
        ret, test_frame = cap.read()
        if ret:
            height, width = test_frame.shape[:2]
            frame_center = (width // 2, height // 2)
            print(f"Camera frame size: {width}x{height}, Center: {frame_center}")
            # Warm up camera
            for _ in range(5): cap.read()
            return True
        else:
            print("Error: Failed to read frame from camera during initialization.")
            cap.release()
            cap = None
            return False
    except Exception as e:
        print(f"Error initializing camera: {e}")
        if cap: cap.release()
        cap = None
        return False

# --- YOLO Model Loading with Jetson Optimization ---
def load_optimized_yolo_model():
    """Load YOLO model with Jetson optimizations."""
    global model # Ensure 'model' is treated as a global to be assigned
    try:
        from ultralytics import YOLO
        model_paths = [r"C:\Users\srava\Downloads\best1.pt"]
        loaded_model = None
        for model_path in model_paths:
            try:
                print(f"Attempting to load model: {model_path}")
                # Check if model file exists, Ultralytics YOLO might download if not found by exact path
                if not os.path.exists(model_path) and model_path == 'yolov8n.pt':
                    print(f"{model_path} not found locally, YOLO will attempt to download.")
                elif not os.path.exists(model_path):
                    print(f"{model_path} not found locally, skipping.")
                    continue

                loaded_model = YOLO(r"C:\Users\srava\Downloads\best1.pt")
                print(f"✓ Model preliminarily loaded: {model_path}")

                if TRT_AVAILABLE and hasattr(loaded_model, 'export'):
                    try:
                        print("Attempting TensorRT optimization...")
                        engine_path = model_path.replace('.pt', f'_trt_engine_fp16_gpu0_ws1.engine') # More specific name
                        # Check if engine file already exists and is compatible
                        if not os.path.exists(engine_path):
                             print(f"Exporting to TensorRT engine: {engine_path} (this may take a while)...")
                             # Ensure device is explicitly set for export if necessary for your Ultralytics version
                             loaded_model.export(format='engine', device='0', half=True, workspace=1) # Use FP16, device 0
                             # The exported engine name might vary slightly, ensure the actual name is used
                             # Ultralytics might append details, so locate the generated .engine file.
                             # For simplicity, we assume 'model_path.replace' works or user adjusts.
                             print(f"✓ TensorRT engine exported to {engine_path}")
                             # Reload model from engine file
                             # loaded_model = YOLO(engine_path) # Some versions require this
                        else:
                            print(f"Using existing TensorRT engine: {engine_path}")
                        # After export or if engine exists, ensure the model object is the engine-loaded one.
                        # This might mean re-initializing: model = YOLO(engine_path)
                        # For now, assume direct use or that Ultralytics handles it internally.
                        # It's safer to load the engine explicitly if Ultralytics version requires:
                        # model = YOLO(engine_path, task=loaded_model.task) # task might be needed
                        print("✓ TensorRT optimization/check completed.")
                    except Exception as trt_e:
                        print(f"TensorRT optimization failed: {trt_e}. Using PyTorch model on GPU if available.")
                else:
                     print("TensorRT not available or model does not support export. Using PyTorch model.")

                # Configure device for inference
                if TRT_AVAILABLE: # Even if export failed, try to run PyTorch on CUDA
                    loaded_model.to('cuda:0') # Move PyTorch model to GPU
                print(f"✓ Model ready for inference on {'GPU (CUDA)' if TRT_AVAILABLE else 'CPU'}.")
                model = loaded_model # Assign to global model
                return True
            except Exception as e:
                print(f"Failed to load or optimize {model_path}: {e}")
        print("ERROR: No YOLO model could be loaded.")
        return False
    except ImportError:
        print("Error: Ultralytics not installed. Please install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"General error loading YOLO model: {e}")
        return False

# Initialize model
model = None # Global model variable
model_loaded = load_optimized_yolo_model()


# --- Enhanced Computer Vision Functions ---
def detect_mango_jetson(image_to_detect):
    """
    Jetson-optimized mango detection using YOLOv8.
    Returns (success, center_x, center_y, width, height, confidence) of the best detected mango.
    """
    global model
    if not model_loaded or model is None:
        return False, 0, 0, 0, 0, 0.0

    try:
        # Perform inference
        results = model(image_to_detect,
                        verbose=False,
                        conf=DETECTION_CONFIDENCE_THRESHOLD,
                        device='cuda:0' if TRT_AVAILABLE else 'cpu',
                        half=True if TRT_AVAILABLE else False) # FP16 only on CUDA

        if not results: # Handles model returning None or an empty list
            # print("DEBUG: YOLO model returned no results list.")
            return False, 0, 0, 0, 0, 0.0

        result = results[0] # Get the first Results object from the list

        # Corrected check:
        # 1. Check if the result object itself is None (e.g., if results was [None])
        # 2. Check if result.boxes is None (this is the key fix for the reported error)
        # 3. Check if result.boxes is an empty collection
        if result is None or result.boxes is None or len(result.boxes) == 0:
            # Optional: Add specific debug prints if you want to know which condition was met
            # if result is None:
            #     print("DEBUG: YOLO detection result object is None.")
            # elif result.boxes is None:
            #     print("DEBUG: YOLO result.boxes is None.")
            # else: # len(result.boxes) == 0
            #     print("DEBUG: YOLO result.boxes is empty (no detections).")
            return False, 0, 0, 0, 0, 0.0

        best_detection_box = None
        highest_confidence = 0.0
        
        target_classes = ['mango'] 
        found_target_fruit = False

        for box in result.boxes: # Now result.boxes is guaranteed to be a non-None, iterable object
            confidence = float(box.conf.item())
            # No need to re-check DETECTION_CONFIDENCE_THRESHOLD here if model(conf=...) is respected
            # but it doesn't hurt as a safeguard if conf in model() is just a hint.

            cls_id = int(box.cls.item())
            class_name = model.names[cls_id].lower() if model.names and cls_id < len(model.names) else "unknown"

            if any(fruit_name in class_name for fruit_name in target_classes):
                if confidence > highest_confidence: # Ensure confidence is indeed for this box.
                    highest_confidence = confidence
                    best_detection_box = box
                    found_target_fruit = True
            elif not found_target_fruit and confidence > highest_confidence:
                highest_confidence = confidence
                best_detection_box = box
        
        if best_detection_box is None:
            return False, 0, 0, 0, 0, 0.0

        # Ensure we are using the confidence of the 'best_detection_box'
        # The highest_confidence variable already tracks this.
        final_confidence = float(best_detection_box.conf.item()) 
        if final_confidence < DETECTION_CONFIDENCE_THRESHOLD: # Final check on the chosen box
            return False, 0, 0, 0, 0, 0.0

        x1, y1, x2, y2 = map(int, best_detection_box.xyxy[0].tolist())
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        obj_width = x2 - x1
        obj_height = y2 - y1
        
        return True, center_x, center_y, obj_width, obj_height, final_confidence

    except Exception as e:
        print(f"Error during YOLO detection: {e}") # This is where your error message originates
        import traceback # Optional: for more detailed error logging
        traceback.print_exc() # Optional: prints full stack trace
        return False, 0, 0, 0, 0, 0.0
    

def jetson_video_processing_loop():
    """Jetson-optimized video processing loop for continuous feed and conditional mango detection."""
    global frame, mango_detected, mango_position, mango_tracking_active
    global last_mango_position, mango_lost_counter, frame_count, model_loaded, cv_lock

    print("Jetson video processing loop started.")
    
    while running:
        if not cap or not cap.isOpened():
            # print("Video loop: Camera not available.") # Can be noisy
            eventlet.sleep(1.0) # Wait if camera is not ready
            continue

        ret, current_raw_frame = cap.read()
        if not ret:
            # print("Video loop: Failed to read frame.") # Can be noisy
            eventlet.sleep(0.1)
            continue

        frame_count += 1
        # Always make a copy to draw on, to keep raw_frame if needed elsewhere (though not currently)
        processed_frame_for_stream = current_raw_frame.copy()

        # Variables for this frame's detection results
        detected_this_frame = False
        cx, cy, conf = 0, 0, 0.0
        global video_writer, is_recording # Make sure these are accessible
        if is_recording and video_writer is not None:
            try:
            # Ensure the frame being written is the one with all drawings
                video_writer.write(processed_frame_for_stream)
            except Exception as e:
                print(f"Error writing frame to video: {e}")

        if mango_tracking_active and model_loaded:
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                # Perform detection on the raw frame (or a resized version if preferred)
                is_detected, det_x, det_y, det_w, det_h, det_conf = detect_mango_jetson(current_raw_frame)
                
                with cv_lock: # Lock when updating shared mango state
                    if is_detected:
                        mango_detected = True
                        mango_position = (det_x, det_y)
                        frame = processed_frame_for_stream
                        last_mango_position = mango_position
                        mango_lost_counter = 0
                        
                        detected_this_frame = True
                        cx, cy, conf = det_x, det_y, det_conf

                        # Draw detection on the frame for streaming
                        cv2.rectangle(processed_frame_for_stream, (det_x - det_w//2, det_y - det_h//2),
                                      (det_x + det_w//2, det_y + det_h//2), (0, 255, 0), 2)
                        cv2.circle(processed_frame_for_stream, (det_x, det_y), 5, (0, 0, 255), -1)
                        cv2.putText(processed_frame_for_stream, f'Mango: {det_conf:.2f}',
                                    (det_x - det_w//2, det_y - det_h//2 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(processed_frame_for_stream, f'Temp: {cpu_temp}C', (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    else:
                        mango_lost_counter += 1
                        if mango_lost_counter > MAX_MANGO_LOST_FRAMES:
                            mango_detected = False
                        # detected_this_frame remains False
            else: # Not processing detection on this frame, use last known status for drawing if needed
                with cv_lock: # Access shared state
                    if mango_detected: # If still considered detected (within lost_frames)
                        detected_this_frame = True
                        cx, cy = mango_position[0], mango_position[1]
                        # Optionally draw last known position with a different style
                        cv2.circle(processed_frame_for_stream, mango_position, 7, (255, 255, 0), 1) # Cyan for stale
        else: # Mango tracking is not active or model not loaded
            with cv_lock:
                mango_detected = False # Ensure state is false

        # Draw UI elements (center crosshair, status text, temp) always
        if frame_center[0] > 0 and frame_center[1] > 0: # Ensure frame_center is initialized
            cv2.line(processed_frame_for_stream, (frame_center[0] - 20, frame_center[1]),
                     (frame_center[0] + 20, frame_center[1]), (255, 0, 0), 1)
            cv2.line(processed_frame_for_stream, (frame_center[0], frame_center[1] - 20),
                     (frame_center[0], frame_center[1] + 20), (255, 0, 0), 1)

        tracking_status_text = "TRACKING ACTIVE" if mango_tracking_active else "TRACKING INACTIVE"
        tracking_status_color = (0, 255, 0) if mango_tracking_active else (0, 0, 255)
        cv2.putText(processed_frame_for_stream, tracking_status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, tracking_status_color, 2)
        
        if not model_loaded:
            cv2.putText(processed_frame_for_stream, "MODEL NOT LOADED", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)


        cpu_temp = get_jetson_temperature()
        if cpu_temp is not None:
            cv2.putText(processed_frame_for_stream, f'Temp: {cpu_temp}C', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        with cv_lock: # Lock when updating the global frame used for streaming
            frame = processed_frame_for_stream
        
        # Emit WebSocket update with the status of mango detection relevant to tracking
        # If tracking is active, detected_this_frame reflects if a mango is currently being seen/held.
        # If tracking is inactive, detection is effectively off.
        socketio.emit('mango_update', {
            'detected': detected_this_frame if mango_tracking_active else False,
            'x': cx if mango_tracking_active and detected_this_frame else 0,
            'y': cy if mango_tracking_active and detected_this_frame else 0,
            'confidence': conf if mango_tracking_active and detected_this_frame else 0.0,
            'frame_width': processed_frame_for_stream.shape[1],
            'frame_height': processed_frame_for_stream.shape[0],
            'tracking_active': mango_tracking_active,
            'cpu_temp': cpu_temp if cpu_temp is not None else "N/A"
        })

        if frame_count % MEMORY_CLEANUP_INTERVAL == 0:
            gc.collect()
            # print(f"Memory usage: {psutil.virtual_memory().percent}%") # Optional: for debugging

        eventlet.sleep(1/CAMERA_FPS - 0.005) # Adjust sleep to match FPS, with a small buffer

    print("Video processing loop stopped.")


def get_jetson_temperature():
    """Get Jetson CPU/SoC temperature for monitoring."""
    temp_paths = [
        '/sys/class/thermal/thermal_zone0/temp',
        '/sys/class/thermal/thermal_zone1/temp', # Often GPU or another part of SoC
        '/sys/devices/virtual/thermal/thermal_zone0/temp',
        '/sys/devices/platform/coretemp.0/hwmon/hwmon1/temp1_input', # For some x86 Jetsons
    ]
    for path in temp_paths:
        try:
            with open(path, 'r') as f:
                temp_raw = int(f.read().strip())
            # Temperatures can be in millidegrees Celsius or degrees Celsius
            # If value is large (e.g., > 1000), assume millidegrees
            temp_c = temp_raw / 1000.0 if temp_raw > 1000 else float(temp_raw)
            if 10 < temp_c < 110:  # Reasonable temperature range in Celsius
                return round(temp_c, 1)
        except (FileNotFoundError, ValueError, IOError):
            continue
    return None


@app.route('/command/start_recording', methods=['POST'])
def command_start_recording():
    global is_recording, video_writer, current_recording_filename, frame_center, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, frame, cv_lock, RECORDING_PATH

    if is_recording:
        return jsonify({"status": "info", "message": "Recording is already active"}), 200

    with cv_lock: # Access global frame safely
        if frame is None: # Check if video feed is active (frame is being produced)
            return jsonify({"status": "error", "message": "Video feed not active. Cannot start recording."}), 400
        current_frame_for_dims = frame.copy() # Use a copy for dimension checking

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Ensure RECORDING_PATH exists, though it should have been created at startup
    if not os.path.exists(RECORDING_PATH):
        try:
            os.makedirs(RECORDING_PATH)
        except OSError as e:
             return jsonify({"status": "error", "message": f"Failed to create recording directory: {str(e)}"}), 500

    current_recording_filename = os.path.join(RECORDING_PATH, f"recording_{timestamp}.mp4")
    
    height, width = current_frame_for_dims.shape[:2]
    
    # Standard MP4 codec. Others like 'XVID' for .avi might also work.
    # 'mp4v' is generally good for .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # Cap recording FPS to ensure stability, adjust as needed
    recording_fps = min(CAMERA_FPS, 20.0) # Use float for FPS

    try:
        video_writer = cv2.VideoWriter(current_recording_filename, fourcc, recording_fps, (width, height))
        if not video_writer.isOpened():
            # Try to clean up before raising error
            if os.path.exists(current_recording_filename):
                 try: os.remove(current_recording_filename)
                 except: pass
            raise Exception(f"Failed to open VideoWriter for {current_recording_filename}. Check codec and permissions.")
        
        is_recording = True
        print(f"Recording started: {current_recording_filename} at {width}x{height} @{recording_fps}fps")
        socketio.emit('status_update', {'message': f"REC ON: {os.path.basename(current_recording_filename)}"})
        return jsonify({"status": "success", "message": f"Recording started: {os.path.basename(current_recording_filename)}"}), 200
    except Exception as e:
        print(f"Error starting recording: {e}")
        is_recording = False
        if video_writer: # Ensure it's released if partially opened
            video_writer.release()
        video_writer = None
        # Clean up potentially created (but unusable) file
        if current_recording_filename and os.path.exists(current_recording_filename):
            try: os.remove(current_recording_filename)
            except: pass
        current_recording_filename = None
        return jsonify({"status": "error", "message": f"Failed to start recording: {str(e)}"}), 500

@app.route('/command/stop_recording', methods=['POST'])
def command_stop_recording():
    global is_recording, video_writer, current_recording_filename
    if not is_recording:
        return jsonify({"status": "info", "message": "Recording is not active"}), 200

    active_filename_to_return = current_recording_filename # Store before reset
    is_recording = False # Signal processing loop to stop writing frames
    
    # Brief pause to ensure the last few frames are potentially written
    eventlet.sleep(0.5) 

    if video_writer:
        video_writer.release()
        video_writer = None
    
    print(f"Recording stopped: {active_filename_to_return}")
    
    if active_filename_to_return and os.path.exists(active_filename_to_return):
        socketio.emit('status_update', {'message': f"REC OFF: {os.path.basename(active_filename_to_return)}. Ready for download."})
        response_data = {
            "status": "success", 
            "message": f"Recording stopped: {os.path.basename(active_filename_to_return)}.",
            "filename": os.path.basename(active_filename_to_return)
        }
        current_recording_filename = None # Reset for the next recording session
        return jsonify(response_data), 200
    else:
        socketio.emit('status_update', {'message': "Recording stopped, but file not found."})
        current_recording_filename = None
        return jsonify({"status": "error", "message": "Recording file not found after stopping."}), 500

@app.route('/download_video/<path:filename>')
def download_video(filename):
    global RECORDING_PATH
    try:
        # Security: os.path.basename ensures filename doesn't try to navigate up directories
        safe_filename = os.path.basename(filename)
        print(f"Download request for: {safe_filename} from path {RECORDING_PATH}")
        if not os.path.exists(os.path.join(RECORDING_PATH, safe_filename)):
            return jsonify({"status": "error", "message": "File not found for download."}), 404

        return send_from_directory(RECORDING_PATH, safe_filename, as_attachment=True)
    except FileNotFoundError: # Should be caught by the check above, but as a fallback
        print(f"Error: File not found for download - {filename}")
        return jsonify({"status": "error", "message": "File not found."}), 404
    except Exception as e:
        print(f"Error during video download: {e}")
        return jsonify({"status": "error", "message": f"Server error during download: {str(e)}"}), 500

def jetson_memory_monitor():
    """Monitor Jetson memory usage and clean up if needed."""
    print("Memory monitor thread started.")
    while running:
        try:
            memory = psutil.virtual_memory()
            if memory.percent > 85: # If memory usage > 85%
                print(f"Warning: High memory usage ({memory.percent}%). Triggering GC cleanup...")
                gc.collect()
            # Optional: Log memory usage periodically for diagnostics
            # print(f"Current memory usage: {memory.percent}%")
        except Exception as e:
            print(f"Error in memory monitor: {e}")
        
        # Check memory every 30 seconds, less frequently than frame processing
        for _ in range(30): 
            if not running: break
            eventlet.sleep(1) 
    print("Memory monitor thread stopped.")


# --- Modified Drone Connection for Jetson ---
def connect_vehicle_jetson():
    """Connect to vehicle with Jetson-specific connection strings."""
    global vehicle
    
    # Common Jetson serial port for Pixhawk/flight controller
    # Ensure user has permissions: sudo usermod -a -G dialout $USER
    connection_strings = [
        '/dev/ttyACM0',
        'tcp:127.0.0.1:5762',
        '/dev/ttyTHS1',      # Jetson Nano/NX/Xavier Serial Port (often for Pixhawk)
        'udp:127.0.0.1:14550',# Common for SITL or local UDP proxy
        '/dev/ttyUSB0',      # First USB-Serial adapter
        '/dev/ttyACM0',      # Common for USB-CDC devices like some flight controllers
        'tcp:127.0.0.1:5760', # MAVProxy default TCP
         # Alternative TCP
    ]
    
    for connection_string in connection_strings:
        if not running: break # Exit if app is shutting down
        print(f"Attempting vehicle connection: {connection_string}")
        try:
            baud_rate = 115200 if connection_string.startswith('/dev/ttyTHS') else 57600
            if connection_string.startswith('/dev/'):
                print(f"Using baud rate: {baud_rate} for {connection_string}")
                vehicle_instance = connect(connection_string, baud=baud_rate, wait_ready=True, timeout=20)
            else:
                vehicle_instance = connect(connection_string, wait_ready=True, timeout=20)
            
            if vehicle_instance:
                vehicle = vehicle_instance # Assign to global vehicle
                print(f"✓ Vehicle connected successfully on {connection_string}")
                print(f"  Firmware: {vehicle.version}")
                print(f"  Mode: {vehicle.mode.name}, Armed: {vehicle.armed}")

                # Start telemetry thread (ensure it's an eventlet green thread if possible, or manage carefully)
                # For simplicity with dronekit, standard thread is often used.
                active_threads = [t.name for t in threading.enumerate()]
                if 'TelemetryThread' not in active_threads:
                    telemetry_thread = threading.Thread(target=telemetry_update_loop, name='TelemetryThread', daemon=True)
                    telemetry_thread.start()
                    print("Telemetry thread started.")
                else:
                    print("Telemetry thread already running.")
                return # Exit connection loop on success
            
        except APIException as api_e:
            print(f"Dronekit API Error connecting to {connection_string}: {api_e}")
        except Exception as e:
            print(f"General error connecting to {connection_string}: {e}")
        
        vehicle = None # Ensure vehicle is None if connection failed
        if running: # Don't sleep if we are shutting down
             eventlet.sleep(2) # Wait before trying next connection string

    if vehicle is None and running:
        print("Failed to connect to vehicle on all attempted connection strings.")
        print("Please check drone power, connections, and MAVLink stream.")


def get_telemetry():
    """Enhanced telemetry data collection."""
    if not vehicle or not hasattr(vehicle, 'location'):
        return {}

    try:
        if not all([
            vehicle.location, vehicle.location.global_relative_frame,
            vehicle.attitude, vehicle.battery, vehicle.mode,
            hasattr(vehicle, 'armed'), hasattr(vehicle, 'is_armable'),
            hasattr(vehicle, 'system_status'), hasattr(vehicle, 'gps_0')
        ]):
            return {}
            
        loc = vehicle.location.global_relative_frame
        att = vehicle.attitude
        bat = vehicle.battery
        gps = vehicle.gps_0

        telemetry = {
            "latitude": getattr(loc, 'lat', 0),
            "longitude": getattr(loc, 'lon', 0),
            "altitude": round(getattr(loc, 'alt', 0), 2),
            "groundspeed": round(getattr(vehicle, 'groundspeed', 0), 2),
            "airspeed": round(getattr(vehicle, 'airspeed', 0), 2),
            "heading": getattr(vehicle, 'heading', 0),
            "roll": round(math.degrees(getattr(att, 'roll', 0)), 2),
            "pitch": round(math.degrees(getattr(att, 'pitch', 0)), 2),
            "yaw": round(math.degrees(getattr(att, 'yaw', 0)), 2),
            "battery_voltage": round(getattr(bat, 'voltage', 0), 2),
            "battery_current": getattr(bat, 'current', 0),
            "battery_level": getattr(bat, 'level', 0),
            "mode": getattr(vehicle.mode, 'name', "UNKNOWN"),
            "armed": vehicle.armed,
            "is_armable": vehicle.is_armable,
            "system_status": getattr(vehicle.system_status, 'state', "UNKNOWN"),
            "gps_fix": getattr(gps, 'fix_type', 0),
            "gps_satellites": getattr(gps, 'satellites_visible', 0),
            "ekf_ok": vehicle.ekf_ok if hasattr(vehicle, 'ekf_ok') else None,
            "mango_tracking_active": mango_tracking_active,
            "mango_detected": mango_detected,
            "search_state": search_state,  # Add search state to telemetry
            "cpu_temperature": get_jetson_temperature(),
            "memory_usage": round(psutil.virtual_memory().percent,1)
        }
        return telemetry
    except Exception as e:
        print(f"Error accessing vehicle attributes for telemetry: {e}")
        return {}

def telemetry_update_loop():
    """Telemetry loop with Jetson monitoring."""
    global telemetry_data
    print("Telemetry loop started.")
    while running:
        if vehicle:
            try:
                current_telemetry = get_telemetry()
                if current_telemetry: # Only update if data is valid
                    telemetry_data = current_telemetry
                    socketio.emit('telemetry_update', telemetry_data)
            except Exception as e:
                print(f"Error in telemetry loop: {e}")
        
        # Sleep using eventlet for cooperative multitasking
        # Update interval can be adjusted based on needs
        for _ in range(int(update_interval * 10)): # Sleep in smaller chunks
            if not running: break
            eventlet.sleep(0.1)
    print("Telemetry loop stopped.")

def send_enhanced_movement_command(vx, vy, vz, yaw_rate):
    """Send movement command to the vehicle using MAVLink."""
    try:
        msg = vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000011111000111,
            0, 0, 0, vx, vy, vz, 0, 0, 0, 0, yaw_rate)
        vehicle.send_mavlink(msg)
    except Exception as e:
        print(f"Error sending movement command: {e}")

def mango_tracking_loop():
    global mango_detected, mango_position, frame_center, mango_tracking_active, cv_lock, vehicle
    global search_state, search_start_time, rotation_start_time, forward_movement_start_time
    global total_rotation_degrees, incremental_rotation_target, detection_wait_start_time
    global current_rotation_degrees

    print("Enhanced mango tracking control loop started.")
    while running:
        if not (vehicle and vehicle.armed and mango_tracking_active and frame_center[0] > 0):
            search_state = "idle"
            eventlet.sleep(0.2) 
            continue

        with cv_lock:
            current_mango_detected = mango_detected
            current_mango_position_x, current_mango_position_y = mango_position

        current_time = time.time()

        if current_mango_detected:
            search_state = "idle"
            current_rotation_degrees = 0
            incremental_rotation_target = 0
            
            try:
                x_offset = current_mango_position_x - frame_center[0]
                y_offset = current_mango_position_y - frame_center[1]
                x_dead_zone = frame_center[0] * MOVEMENT_DEAD_ZONE_PERCENTAGE
                y_dead_zone = frame_center[1] * MOVEMENT_DEAD_ZONE_PERCENTAGE
                vx, vy, vz, yaw_rate_deg_s = 0.0, 0.0, ALTITUDE_HOLD_VELOCITY, 0.0

                if abs(x_offset) > x_dead_zone:
                    vy = x_offset * VELOCITY_SCALING_FACTOR
                    vy = max(min(vy, MAX_VELOCITY), -MAX_VELOCITY)

                if abs(y_offset) > y_dead_zone:
                    vx = -y_offset * VELOCITY_SCALING_FACTOR
                    vx = max(min(vx, MAX_VELOCITY), -MAX_VELOCITY)

                if YAW_CONTROL_ENABLED and abs(x_offset) > x_dead_zone * 1.5:
                    yaw_rate_deg_s = x_offset * YAW_SCALING_FACTOR
                    yaw_rate_deg_s = max(min(yaw_rate_deg_s, 20), -20)

                if abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vz) > 0.01 or abs(yaw_rate_deg_s) > 0.5:
                    print(f"Tracking mango: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, yaw_rate={yaw_rate_deg_s:.2f}")
                    send_enhanced_movement_command(vx, vy, vz, math.radians(yaw_rate_deg_s))
                else:
                    # Send hover command when mango is centered
                    send_enhanced_movement_command(0, 0, 0, 0)
            except Exception as e:
                print(f"Error in mango tracking movement: {e}")
        else:
            # Enhanced search pattern with incremental rotation and detection pauses
            try:
                if search_state == "idle":
                    search_state = "waiting"
                    search_start_time = current_time
                    current_rotation_degrees = 0
                    incremental_rotation_target = 0
                    print("Mango lost. Starting 15-second wait period...")
                    socketio.emit('status_update', {'message': "Mango lost. Waiting 15 seconds before search..."})
                            
                elif search_state == "waiting":
                    if current_time - search_start_time >= SEARCH_WAIT_TIME:
                        search_state = "rotating_incremental"
                        rotation_start_time = current_time
                        current_rotation_degrees = 0
                        incremental_rotation_target = ROTATION_INCREMENT_DEGREES
                        print("Wait period complete. Starting incremental rotation search...")
                        print(f"Target: Rotate {ROTATION_INCREMENT_DEGREES}° then wait {DETECTION_WAIT_TIME}s for detection")
                        socketio.emit('status_update', {'message': "Starting incremental rotation search..."})
                    else:
                        remaining_wait = SEARCH_WAIT_TIME - (current_time - search_start_time)
                        if int(remaining_wait) % 5 == 0 and int(remaining_wait * 10) % 50 == 0:  # Print every 5 seconds
                            print(f"Waiting for mango... {remaining_wait:.0f} seconds remaining")
                        send_enhanced_movement_command(0, 0, 0, 0)
                        
                elif search_state == "rotating_incremental":
                    rotation_duration = current_time - rotation_start_time
                    degrees_rotated_this_increment = rotation_duration * ROTATION_SPEED_DEG_S
                    
                    if degrees_rotated_this_increment < ROTATION_INCREMENT_DEGREES:
                        # Still rotating for this increment
                        total_degrees_so_far = current_rotation_degrees + degrees_rotated_this_increment
                        print(f"Rotating: {degrees_rotated_this_increment:.1f}°/{ROTATION_INCREMENT_DEGREES}° (Total: {total_degrees_so_far:.1f}°)")
                        send_enhanced_movement_command(0, 0, 0, math.radians(ROTATION_SPEED_DEG_S))
                    else:
                        # Completed this rotation increment, start detection wait
                        search_state = "detecting"
                        detection_wait_start_time = current_time
                        current_rotation_degrees += ROTATION_INCREMENT_DEGREES
                        print(f"Completed {ROTATION_INCREMENT_DEGREES}° rotation. Total rotated: {current_rotation_degrees}°")
                        print(f"Stopping for {DETECTION_WAIT_TIME}s to allow mango detection...")
                        socketio.emit('status_update', {
                            'message': f"Rotated {current_rotation_degrees}°. Waiting {DETECTION_WAIT_TIME}s for detection..."
                        })
                        # Stop all movement during detection period
                        send_enhanced_movement_command(0, 0, 0, 0)
                        
                elif search_state == "detecting":
                    detection_wait_duration = current_time - detection_wait_start_time
                    remaining_detection_time = DETECTION_WAIT_TIME - detection_wait_duration
                    
                    if detection_wait_duration < DETECTION_WAIT_TIME:
                        # Still in detection wait period
                        if int(remaining_detection_time) % 2 == 0 and int(remaining_detection_time * 10) % 20 == 0:  # Print every 2 seconds
                            print(f"Detection wait: {remaining_detection_time:.1f}s remaining at {current_rotation_degrees}° position")
                        # Ensure drone stays still during detection
                        send_enhanced_movement_command(0, 0, 0, 0)
                    else:
                        # Detection wait complete, check if we need to continue rotating or move forward
                        if current_rotation_degrees >= FULL_ROTATION_DEGREES:
                            # Completed full 360° rotation, move forward
                            search_state = "moving_forward"
                            forward_movement_start_time = current_time
                            current_rotation_degrees = 0  # Reset for next search cycle
                            print("360° rotation with detection stops complete. Moving 1 meter forward...")
                            socketio.emit('status_update', {'message': "Full rotation complete. Moving forward to new search area..."})
                        else:
                            # Continue with next rotation increment
                            search_state = "rotating_incremental"
                            rotation_start_time = current_time
                            incremental_rotation_target = current_rotation_degrees + ROTATION_INCREMENT_DEGREES
                            next_target = min(incremental_rotation_target, FULL_ROTATION_DEGREES)
                            degrees_to_rotate = next_target - current_rotation_degrees
                            print(f"Starting next rotation: {degrees_to_rotate}° (to {next_target}° total)")
                            
                elif search_state == "moving_forward":
                    movement_duration = current_time - forward_movement_start_time
                    if movement_duration < FORWARD_MOVEMENT_TIME:
                        distance_moved = movement_duration * FORWARD_MOVEMENT_SPEED
                        remaining_distance = FORWARD_MOVEMENT_DISTANCE - distance_moved
                        print(f"Moving forward: {distance_moved:.1f}m/{FORWARD_MOVEMENT_DISTANCE}m (remaining: {remaining_distance:.1f}m)")
                        send_enhanced_movement_command(FORWARD_MOVEMENT_SPEED, 0, 0, 0)
                    else:
                        # Forward movement complete, start new incremental rotation cycle
                        search_state = "rotating_incremental"
                        rotation_start_time = current_time
                        current_rotation_degrees = 0
                        incremental_rotation_target = ROTATION_INCREMENT_DEGREES
                        print("Forward movement complete. Starting new incremental rotation search cycle...")
                        socketio.emit('status_update', {'message': "Moved 1m forward. Starting new incremental rotation search..."})
                        
            except Exception as e:
                print(f"Error in incremental search pattern: {e}")
                search_state = "idle"
                current_rotation_degrees = 0
                incremental_rotation_target = 0

        eventlet.sleep(0.1)

    print("Enhanced mango tracking control loop stopped.")

def generate_frames():
    """Generate frames for video feed."""
    global frame, cv_lock
    while running:
        with cv_lock: # Ensure thread-safe access to the frame
            if frame is None:
                # Create a placeholder frame if camera is not ready
                placeholder = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No Camera Feed", (CAMERA_WIDTH//2 - 100, CAMERA_HEIGHT//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                current_display_frame = placeholder
            else:
                current_display_frame = frame.copy() # Make a copy to avoid issues if frame is updated

        try:
            ret, buffer = cv2.imencode('.jpg', current_display_frame, [cv2.IMWRITE_JPEG_QUALITY, 75]) # Quality 70-80
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # print("JPEG encoding failed") # Can be noisy
                pass # Handle encoding error, maybe yield a default error image
        except Exception as e:
            print(f"Error generating video frame: {e}")
            # Fallback: yield a small placeholder to keep stream alive on error
            placeholder_error = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.putText(placeholder_error, "Error", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            ret, buffer = cv2.imencode('.jpg', placeholder_error)
            if ret:
                 yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


        eventlet.sleep(1/CAMERA_FPS) # Stream at roughly camera FPS

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index1.html') # Ensure index1.html exists in a 'templates' folder

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/command/arm', methods=['POST'])
def command_arm():
    if not vehicle: return jsonify({"status": "error", "message": "Vehicle not connected"}), 500
    try:
        if vehicle.armed: return jsonify({"status": "info", "message": "Vehicle already armed"})
        if not vehicle.is_armable: return jsonify({"status": "error", "message": "Vehicle not armable. Check pre-arm checks."}), 400
        
        print("Arming command received. Setting mode to GUIDED.")
        if vehicle.mode.name != "GUIDED":
            vehicle.mode = VehicleMode("GUIDED")
            # Wait for mode change confirmation
            timeout = time.time() + 5 # 5 seconds timeout
            while vehicle.mode.name != "GUIDED" and time.time() < timeout:
                eventlet.sleep(0.1)
            if vehicle.mode.name != "GUIDED":
                return jsonify({"status": "error", "message": "Failed to set mode to GUIDED"}), 500
        
        vehicle.armed = True
        timeout = time.time() + 10 # 10 seconds timeout for arming
        while not vehicle.armed and time.time() < timeout:
            eventlet.sleep(0.2)
            
        if vehicle.armed:
            print("Vehicle armed successfully.")
            return jsonify({"status": "success", "message": "Vehicle armed successfully"})
        else:
            print("Arming timed out.")
            return jsonify({"status": "error", "message": "Arming timed out. Check pre-arm status."}), 500
    except Exception as e:
        print(f"Arming failed: {e}")
        return jsonify({"status": "error", "message": f"Arming failed: {str(e)}"}), 500

@app.route('/command/disarm', methods=['POST'])
def command_disarm():
    if not vehicle: return jsonify({"status": "error", "message": "Vehicle not connected"}), 500
    try:
        global mango_tracking_active
        if mango_tracking_active:
            print("Disarming: Stopping mango tracking first.")
            mango_tracking_active = False # Stop tracking before disarming
            eventlet.sleep(0.2) # Give a moment for tracking loop to react

        print("Disarm command received.")
        vehicle.armed = False
        timeout = time.time() + 10
        while vehicle.armed and time.time() < timeout:
            eventlet.sleep(0.2)
            
        if not vehicle.armed:
            print("Vehicle disarmed successfully.")
            return jsonify({"status": "success", "message": "Vehicle disarmed successfully"})
        else:
            print("Disarming timed out.")
            return jsonify({"status": "error", "message": "Disarming timed out"}), 500
    except Exception as e:
        print(f"Disarming failed: {e}")
        return jsonify({"status": "error", "message": f"Disarming failed: {str(e)}"}), 500

@app.route('/command/takeoff', methods=['POST'])
def command_takeoff():
    if not vehicle: return jsonify({"status": "error", "message": "Vehicle not connected"}), 500
    if not vehicle.armed: return jsonify({"status": "error", "message": "Vehicle not armed"}), 400
    if vehicle.mode.name != "GUIDED": return jsonify({"status": "error", "message": "Vehicle must be in GUIDED mode for takeoff"}), 400

    try:
        data = request.get_json()
        altitude = float(data.get('altitude', 3.0)) # Default takeoff altitude 3m
        if not (1.0 <= altitude <= 20.0):
            return jsonify({"status": "error", "message": "Altitude must be between 1m and 20m"}), 400
        
        print(f"Takeoff command received. Target altitude: {altitude}m")
        vehicle.simple_takeoff(altitude)
        # Add a small delay to allow the command to be processed by the drone
        # and then monitor altitude in telemetry.
        socketio.emit('status_update', {'message': f"Takeoff to {altitude}m initiated."})
        return jsonify({"status": "success", "message": f"Takeoff command issued to {altitude}m"})
    except Exception as e:
        print(f"Takeoff failed: {e}")
        return jsonify({"status": "error", "message": f"Takeoff failed: {str(e)}"}), 500

@app.route('/command/land', methods=['POST'])
def command_land():
    if not vehicle: return jsonify({"status": "error", "message": "Vehicle not connected"}), 500
    try:
        global mango_tracking_active
        if mango_tracking_active:
            print("Landing: Stopping mango tracking first.")
            mango_tracking_active = False
            eventlet.sleep(0.2)

        print("Land command received. Setting mode to LAND.")
        vehicle.mode = VehicleMode("LAND")
        socketio.emit('status_update', {'message': "Landing initiated."})
        return jsonify({"status": "success", "message": "Landing initiated"})
    except Exception as e:
        print(f"Landing failed: {e}")
        return jsonify({"status": "error", "message": f"Landing failed: {str(e)}"}), 500

@app.route('/command/rtl', methods=['POST'])
def command_rtl():
    if not vehicle: return jsonify({"status": "error", "message": "Vehicle not connected"}), 500
    try:
        global mango_tracking_active
        if mango_tracking_active:
            print("RTL: Stopping mango tracking first.")
            mango_tracking_active = False
            eventlet.sleep(0.2)

        print("RTL command received. Setting mode to RTL.")
        vehicle.mode = VehicleMode("RTL")
        socketio.emit('status_update', {'message': "Return to Launch initiated."})
        return jsonify({"status": "success", "message": "Return to Launch initiated"})
    except Exception as e:
        print(f"RTL failed: {e}")
        return jsonify({"status": "error", "message": f"RTL failed: {str(e)}"}), 500

@app.route('/command/start_tracking', methods=['POST'])
def command_start_tracking():
    global mango_tracking_active, model_loaded, cap, vehicle
    if not cap or not cap.isOpened():
        return jsonify({"status": "error", "message": "Camera not initialized or not open."}), 500
    if not model_loaded:
        return jsonify({"status": "error", "message": "Object detection model not loaded."}), 500
    if not vehicle:
         return jsonify({"status": "error", "message":"Vehicle not connected. Cannot start tracking."}), 400
    if not vehicle.armed:
         return jsonify({"status": "error", "message": "Vehicle not armed. Arm before starting tracking."}), 400
    if vehicle.mode.name != "GUIDED":
        return jsonify({"status": "error", "message": f"Vehicle must be in GUIDED mode. Current: {vehicle.mode.name}"}), 400
    
    if vehicle.location.global_relative_frame.alt < 1.0 : # Min altitude check
         return jsonify({"status": "error", "message": f"Vehicle too low (alt: {vehicle.location.global_relative_frame.alt:.1f}m). Takeoff to a safe altitude first."}), 400


    mango_tracking_active = True
    print("✅ Mango tracking activated via command.")
    socketio.emit('status_update', {'message': "Mango tracking activated."})
    return jsonify({"status": "success", "message": "Mango tracking activated"})

@app.route('/command/stop_tracking', methods=['POST'])
def command_stop_tracking():
    global mango_tracking_active, search_state
    mango_tracking_active = False
    search_state = "idle"  # Reset search state
    print("🛑 Mango tracking deactivated via command. Search state reset.")
    
    # Send hover command if vehicle is armed
    if vehicle and vehicle.armed:
        try:
            print("Sending hover command as tracking stopped.")
            msg = vehicle.message_factory.set_position_target_local_ned_encode(
                0, 0, 0, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                0b0000011111000111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            vehicle.send_mavlink(msg)
        except Exception as e:
            print(f"Error sending hover command after stopping tracking: {e}")
            
    socketio.emit('status_update', {'message': "Mango tracking deactivated."})
    return jsonify({"status": "success", "message": "Mango tracking deactivated"})

# --- Graceful Shutdown ---
def signal_handler(sig, frame_obj):
    global running
    print(f'Signal {sig} received! Shutting down gracefully...')
    if not running: # Already shutting down
        print("Already attempting to shut down. Force exiting if stuck.")
        os._exit(1) # Force exit if shutdown is stuck
    running = False

# --- Main Execution ---
if __name__ == '__main__':
    video_writer = None
    is_recording = False
    current_recording_filename = None


    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("🚀 Starting Mango Tracking Drone Application...")

    if not os.path.exists(RECORDING_PATH):
        try:
            os.makedirs(RECORDING_PATH)
            print(f"Created recording directory: {RECORDING_PATH}")
        except OSError as e:
            print(f"Error creating recording directory {RECORDING_PATH}: {e}")
            # Handle error appropriately, maybe exit or disable recording

    # Initialize Camera (essential for video feed and CV)
    if not initialize_jetson_camera():
        print("⚠ CRITICAL: Failed to initialize camera. Video stream and CV features will be disabled.")
        # Application can continue, but video-dependent features won't work.
        # User should be informed via UI if possible.


    
    # Model is loaded globally at startup. model_loaded flag indicates success.
    if not model_loaded:
        print("⚠ WARNING: YOLO Model not loaded. Mango detection and tracking will be disabled.")

    # Attempt to connect to the vehicle in a non-blocking way using eventlet.spawn
    # This allows the web server to start even if the drone isn't immediately available.
    print("🚁 Attempting to connect to vehicle in background...")
    eventlet.spawn(connect_vehicle_jetson) # connect_vehicle_jetson starts telemetry_thread on success

    print("💡 Starting background threads using eventlet.spawn for cooperative multitasking...")
    eventlet.spawn(jetson_video_processing_loop) # For camera feed and CV
    eventlet.spawn(mango_tracking_loop)          # For drone movement based on CV
    eventlet.spawn(jetson_memory_monitor)        # For Jetson resource monitoring

    print(f"🌍 Starting Flask-SocketIO server on http://0.0.0.0:5000")
    print("ℹ  Press Ctrl+C to exit.")
    
    try:
        # use_reloader=False is important with threaded/eventlet setups to avoid multiple initializations
        socketio.run(app, host='0.0.0.0', port=5000, use_reloader=False, debug=False)
    except Exception as e:
        print(f"💥 Failed to start Flask-SocketIO server: {e}")
    finally:

        
        print("\nApplication is shutting down...")
        running = False # Signal all threads to stop

        print("Waiting for threads to complete (max 2 seconds)...")
        # Give threads a moment to notice the 'running' flag and clean up
        # This relies on threads checking running periodically.
        # For eventlet, tasks should yield. For standard threads, they must check running.
        # A more robust shutdown might involve joining threads with timeouts.
        eventlet.sleep(2.0) 
        # ... (before print("Cleanup complete. Exiting now."))
        # ensure globals are accessible
        if is_recording and video_writer:
            print("Stopping active recording due to shutdown...")
            video_writer.release()
            is_recording = False
            print(f"Recording {current_recording_filename if current_recording_filename else 'N/A'} finalized.")
        video_writer = None


        if vehicle:
            print("Closing vehicle connection...")
            if vehicle.armed:
                print("Disarming vehicle before exit...")
                try:
                    vehicle.armed = False
                    # Wait briefly for disarm
                    _t_start = time.time()
                    while vehicle.armed and (time.time() - _t_start < 3):
                        eventlet.sleep(0.1)
                except Exception as e_disarm:
                    print(f"Error during final disarm: {e_disarm}")
            try:
                vehicle.close()
                print("Vehicle connection closed.")
            except Exception as e_close:
                print(f"Error closing vehicle connection: {e_close}")
        
        if cap:
            print("Releasing camera...")
            cap.release()
            print("Camera released.")
        
        # Perform a final garbage collection
        gc.collect()
        print("Cleanup complete. Exiting now.")
        # os._exit(0) # Consider if a forceful exit is needed if threads don't stop