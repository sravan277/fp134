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
from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException #, Command
from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
from pymavlink import mavutil
import os
import gc
import psutil
import signal # For graceful shutdown

# --- Jetson Optimization Imports ---
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
    print("TensorRT available for GPU acceleration")
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT not available, using CPU inference")

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'jetson_mango_drone_2024_secret_key!'
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# --- Global Variables ---
vehicle = None
telemetry_data = {}
update_interval = 1.0
running = True

video_writer = None
is_recording = False
current_recording_filename = None
RECORDING_PATH = "recordings"

cap = None
frame = None
mango_detected = False
mango_position = (0, 0)
frame_center = (0, 0)
mango_tracking_active = False
cv_lock = threading.Lock()

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_BUFFER_SIZE = 1

search_state = "idle"
search_start_time = 0
rotation_start_time = 0
forward_movement_start_time = 0
current_search_yaw = 0
total_rotation_degrees = 0
forward_movement_distance = 0
ROTATION_INCREMENT_DEGREES = 20.0
DETECTION_WAIT_TIME = 10.0
incremental_rotation_target = 0
detection_wait_start_time = 0
current_rotation_degrees = 0

SEARCH_WAIT_TIME = 6.0
ROTATION_SPEED_DEG_S = 40.0 # Reduced for smoother search turns
FULL_ROTATION_DEGREES = 360.0
FORWARD_MOVEMENT_SPEED = 0.5
FORWARD_MOVEMENT_DISTANCE = 1.0
FORWARD_MOVEMENT_TIME = FORWARD_MOVEMENT_DISTANCE / FORWARD_MOVEMENT_SPEED

DETECTION_CONFIDENCE_THRESHOLD = 0.6
MOVEMENT_DEAD_ZONE_PERCENTAGE = 0.12
MAX_VELOCITY = 0.3
VELOCITY_SCALING_FACTOR = 0.008
ALTITUDE_HOLD_VELOCITY = 0.0
YAW_CONTROL_ENABLED = True
YAW_SCALING_FACTOR = 0.020

PROCESS_EVERY_N_FRAMES = 2
MAX_MANGO_LOST_FRAMES = 20
MEMORY_CLEANUP_INTERVAL = 100

last_mango_position = (0, 0)
mango_lost_counter = 0
frame_count = 0

# --- Geofence Global Variables ---
geofence_polygon = None  # Will store a list of [lat, lon] tuples
geofence_active = False
GEOFENCE_BOUNDARY_THRESHOLD_METERS = 1.5 # How close to boundary before reacting
GEOFENCE_TURN_YAW_RATE_DEG_S = 30 # Degrees per second for boundary turn
GEOFENCE_TURN_DURATION_S = 3.0 # Turn for 3 seconds (30 deg/s * 3s = 90 deg)
geofence_breach_recovery_state = None # e.g., "turning_right", "moving_inwards"
geofence_breach_recovery_start_time = 0

# Initialize model
model = None
model_loaded = False # Will be set by load_optimized_yolo_model

# --- Point-in-Polygon Function ---
def is_point_in_polygon(point_lat, point_lon, polygon_vertices):
    """
    Checks if a point (lat, lon) is inside a polygon.
    Ray casting algorithm.
    Args:
        point_lat: Latitude of the point.
        point_lon: Longitude of the point.
        polygon_vertices: A list of [lat, lon] tuples representing the polygon.
    Returns:
        True if the point is in the polygon, False otherwise.
    """
    if not polygon_vertices or len(polygon_vertices) < 3:
        return False # Not a valid polygon

    num_vertices = len(polygon_vertices)
    inside = False
    p1_lat, p1_lon = polygon_vertices[0]

    for i in range(num_vertices + 1):
        p2_lat, p2_lon = polygon_vertices[i % num_vertices]
        if point_lon > min(p1_lon, p2_lon):
            if point_lon <= max(p1_lon, p2_lon):
                if point_lat <= max(p1_lat, p2_lat):
                    if p1_lon != p2_lon:
                        x_intersection = (point_lon - p1_lon) * (p2_lat - p1_lat) / (p2_lon - p1_lon) + p1_lat
                        if p1_lat == p2_lat or point_lat <= x_intersection:
                            inside = not inside
        p1_lat, p1_lon = p2_lat, p2_lon
    return inside

# --- Helper function to get distance (Haversine) ---
def get_distance_metres(aLocation1, aLocation2):
    """
    Returns the ground distance in metres between two LocationGlobal objects.
    This method is an approximation, and will not be accurate over large distances and close to the
    earth's poles. It comes from the ArduPilot test code:
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5


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

# --- (Your existing functions: get_jetson_camera_pipeline, initialize_jetson_camera, load_optimized_yolo_model, etc.)
# Make sure load_optimized_yolo_model sets the global `model_loaded = True` on success.
def load_optimized_yolo_model():
    """Load YOLO model with Jetson optimizations."""
    global model, model_loaded # Ensure 'model' is treated as a global to be assigned
    try:
        from ultralytics import YOLO
        # Use a more specific path if needed, or ensure 'yolov8n.pt' is in the right place
        # For your custom model:
        model_path = r"C:\Users\srava\Downloads\best1.pt" # Make sure this path is correct on the Jetson
        
        print(f"Attempting to load model: {model_path}")
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}")
            model_loaded = False
            return False

        # Try loading with error handling for this specific step
        try:
            loaded_model_instance = YOLO(model_path)
        except Exception as yolo_load_e:
            print(f"ERROR: Failed to initialize YOLO model with path '{model_path}': {yolo_load_e}")
            model_loaded = False
            return False
        
        print(f"✓ Model preliminarily loaded: {model_path}")

        if TRT_AVAILABLE and hasattr(loaded_model_instance, 'export'):
            try:
                print("Attempting TensorRT optimization...")
                engine_path = model_path.replace('.pt', f'_trt_engine_fp16_gpu0_ws1.engine')
                if not os.path.exists(engine_path):
                    print(f"Exporting to TensorRT engine: {engine_path} (this may take a while)...")
                    loaded_model_instance.export(format='engine', device='0', half=True, workspace=1)
                    print(f"✓ TensorRT engine exported to {engine_path}")
                else:
                    print(f"Using existing TensorRT engine: {engine_path}")
                # Reload model from engine for some Ultralytics versions (ensure this logic)
                # model = YOLO(engine_path, task=loaded_model_instance.task) # Example if task needed
                print("✓ TensorRT optimization/check completed.")
            except Exception as trt_e:
                print(f"TensorRT optimization failed: {trt_e}. Using PyTorch model on GPU if available.")
        else:
            print("TensorRT not available or model does not support export. Using PyTorch model.")

        if TRT_AVAILABLE : # or torch.cuda.is_available()
            loaded_model_instance.to('cuda:0')
        
        model = loaded_model_instance # Assign to global model
        model_loaded = True # Set flag
        print(f"✓ Model '{os.path.basename(model_path)}' ready for inference on {'GPU (CUDA)' if TRT_AVAILABLE else 'CPU'}.")
        return True
    except ImportError:
        print("Error: Ultralytics not installed. Please install with: pip install ultralytics")
        model_loaded = False
        return False
    except Exception as e:
        print(f"General error loading YOLO model: {e}")
        model_loaded = False
        return False

# Call it at startup
if not model_loaded: # Check if not already loaded (e.g. if script reloads)
    load_optimized_yolo_model()
# --- (Your existing functions: detect_mango_jetson, jetson_video_processing_loop, get_jetson_temperature, etc.)

def telemetry_update_loop():
    """Telemetry loop with Jetson monitoring."""
    global telemetry_data, geofence_polygon, geofence_active
    print("Telemetry loop started.")
    while running:
        if vehicle:
            try:
                current_telemetry = get_telemetry() # get_telemetry already includes geofence related keys
                if current_telemetry:
                    telemetry_data = current_telemetry
                    
                    # Add geofence status to telemetry sent to client
                    telemetry_data["geofence_status"] = "Active" if geofence_active and geofence_polygon else "Not Defined"
                    telemetry_data["is_inside_geofence"] = False
                    if geofence_active and geofence_polygon and vehicle.location.global_relative_frame.lat is not None:
                         telemetry_data["is_inside_geofence"] = is_point_in_polygon(
                             vehicle.location.global_relative_frame.lat,
                             vehicle.location.global_relative_frame.lon,
                             geofence_polygon
                         )
                    socketio.emit('telemetry_update', telemetry_data)
            except Exception as e:
                print(f"Error in telemetry loop: {e}")
        
        for _ in range(int(update_interval * 10)):
            if not running: break
            eventlet.sleep(0.1)
    print("Telemetry loop stopped.")


def mango_tracking_loop():
    global mango_detected, mango_position, frame_center, mango_tracking_active, cv_lock, vehicle
    global search_state, search_start_time, rotation_start_time, forward_movement_start_time
    global total_rotation_degrees, incremental_rotation_target, detection_wait_start_time
    global current_rotation_degrees
    global geofence_active, geofence_polygon, geofence_breach_recovery_state, geofence_breach_recovery_start_time

    print("Enhanced mango tracking control loop started.")
    while running:
        if not (vehicle and vehicle.armed and mango_tracking_active and frame_center[0] > 0 and vehicle.mode.name == "GUIDED"):
            search_state = "idle"
            geofence_breach_recovery_state = None # Reset recovery if tracking stops or disarmed
            eventlet.sleep(0.2)
            continue

        current_time = time.time()
        drone_lat = vehicle.location.global_relative_frame.lat
        drone_lon = vehicle.location.global_relative_frame.lon

        # --- Geofence Check and Recovery ---
        if geofence_active and geofence_polygon and drone_lat is not None:
            if not is_point_in_polygon(drone_lat, drone_lon, geofence_polygon):
                if geofence_breach_recovery_state is None: # Just breached
                    print("GEOFENCE BREACH! Attempting recovery.")
                    socketio.emit('geofence_boundary_alert', {'message': 'Drone outside defined area! Attempting recovery.'})
                    search_state = "geofence_recovery" # Prioritize recovery
                    geofence_breach_recovery_state = "stopping_and_turning"
                    geofence_breach_recovery_start_time = current_time
                    # Stop any current movement and initiate turn
                    send_enhanced_movement_command(0, 0, 0, math.radians(GEOFENCE_TURN_YAW_RATE_DEG_S))
                    eventlet.sleep(0.1) # Give time for command to take effect
                    continue # Skip rest of tracking logic for this iteration

            if geofence_breach_recovery_state == "stopping_and_turning":
                if current_time - geofence_breach_recovery_start_time < GEOFENCE_TURN_DURATION_S:
                    # Continue turning
                    send_enhanced_movement_command(0, 0, 0, math.radians(GEOFENCE_TURN_YAW_RATE_DEG_S))
                    eventlet.sleep(0.1)
                    continue
                else:
                    # Turn complete, try to move slightly "forward" (which is now new direction)
                    print("Geofence recovery: Turn complete. Attempting to move inwards.")
                    send_enhanced_movement_command(0, 0, 0, 0) # Stop turning
                    geofence_breach_recovery_state = "moving_inwards"
                    geofence_breach_recovery_start_time = current_time # Reset timer for inward movement

            if geofence_breach_recovery_state == "moving_inwards":
                # Try to move forward (new heading) for a short duration
                if current_time - geofence_breach_recovery_start_time < 2.0: # Move for 2 seconds
                    send_enhanced_movement_command(MAX_VELOCITY * 0.5, 0, 0, 0) # Move slowly
                    eventlet.sleep(0.1)
                    # Check if back inside
                    if is_point_in_polygon(vehicle.location.global_relative_frame.lat, vehicle.location.global_relative_frame.lon, geofence_polygon):
                        print("Geofence recovery: Back inside geofence.")
                        socketio.emit('status_update', {'message': "Drone back inside geofence."})
                        send_enhanced_movement_command(0,0,0,0) # Stop
                        geofence_breach_recovery_state = None
                        search_state = "idle" # Resume normal operations
                    continue
                else:
                    # Still outside after trying to move inwards, or timed out
                    print("Geofence recovery: Failed to re-enter after turn and move. Hovering.")
                    send_enhanced_movement_command(0, 0, 0, 0) # Hover
                    # Consider landing or RTL if still outside after recovery attempts
                    # For now, it will just hover and potentially try recovery again on next cycle if still outside
                    geofence_breach_recovery_state = None # Will trigger stopping_and_turning again if still outside
                    eventlet.sleep(0.1)
                    continue
        
        # If we were in recovery but now back inside, clear recovery state
        if geofence_breach_recovery_state is not None and is_point_in_polygon(drone_lat, drone_lon, geofence_polygon):
            print("Geofence recovery: Confirmed back inside. Resuming normal operations.")
            send_enhanced_movement_command(0,0,0,0) # Stop any recovery movement
            geofence_breach_recovery_state = None
            search_state = "idle" # Reset search state to allow normal tracking/searching

        # --- Normal Mango Tracking and Search Logic (only if not in geofence recovery) ---
        if geofence_breach_recovery_state is None:
            with cv_lock:
                current_mango_detected = mango_detected
                current_mango_position_x, current_mango_position_y = mango_position

            if current_mango_detected:
                # ... (your existing mango detected logic: x_offset, y_offset, vx, vy, yaw_rate_deg_s)
                search_state = "idle" # Reset search state when mango is detected
                current_rotation_degrees = 0
                incremental_rotation_target = 0
                
                try:
                    x_offset = current_mango_position_x - frame_center[0]
                    y_offset = current_mango_position_y - frame_center[1] # y_offset positive is down in image, means drone should move up (negative vz, or positive vx if vz is altitude)
                    x_dead_zone = frame_center[0] * MOVEMENT_DEAD_ZONE_PERCENTAGE
                    y_dead_zone = frame_center[1] * MOVEMENT_DEAD_ZONE_PERCENTAGE
                    
                    vx, vy, vz, yaw_rate_deg_s = 0.0, 0.0, ALTITUDE_HOLD_VELOCITY, 0.0

                    # X offset in image (left/right) controls drone's Y velocity (sideways)
                    # or YAW RATE
                    if abs(x_offset) > x_dead_zone:
                        if YAW_CONTROL_ENABLED:
                            yaw_rate_deg_s = x_offset * YAW_SCALING_FACTOR # Positive x_offset (mango right of center) -> positive yaw_rate (turn right)
                            yaw_rate_deg_s = max(min(yaw_rate_deg_s, 20), -20) # Cap yaw rate
                        else: # Use Y velocity if yaw control is disabled for centering
                            vy = x_offset * VELOCITY_SCALING_FACTOR 
                            vy = max(min(vy, MAX_VELOCITY), -MAX_VELOCITY)
                    
                    # Y offset in image (up/down) controls drone's X velocity (forward/backward)
                    if abs(y_offset) > y_dead_zone:
                        vx = -y_offset * VELOCITY_SCALING_FACTOR # Positive y_offset (mango below center) -> negative vx (move backward to center it in view)
                                                                # Negative y_offset (mango above center) -> positive vx (move forward)
                        vx = max(min(vx, MAX_VELOCITY), -MAX_VELOCITY)
                    
                    # Only send command if there's significant movement needed
                    if abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vz) > 0.01 or abs(yaw_rate_deg_s) > 0.5:
                        print(f"Tracking mango: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, yaw_rate={yaw_rate_deg_s:.2f}")
                        send_enhanced_movement_command(vx, vy, vz, math.radians(yaw_rate_deg_s))
                    else:
                        # Mango is centered, send hover
                        send_enhanced_movement_command(0, 0, 0, 0)
                except Exception as e:
                    print(f"Error in mango tracking movement: {e}")
            else: # Mango not detected, execute search pattern
                # ... (your existing search_state logic: "idle", "waiting", "rotating_incremental", "detecting", "moving_forward")
                try:
                    if search_state == "idle":
                        search_state = "waiting"
                        search_start_time = current_time
                        current_rotation_degrees = 0
                        incremental_rotation_target = 0
                        print("Mango lost. Starting search wait period...")
                        socketio.emit('status_update', {'message': "Mango lost. Waiting before search..."})
                                
                    elif search_state == "waiting":
                        if current_time - search_start_time >= SEARCH_WAIT_TIME:
                            search_state = "rotating_incremental"
                            rotation_start_time = current_time
                            current_rotation_degrees = 0
                            # Start with the first increment
                            incremental_rotation_target = ROTATION_INCREMENT_DEGREES 
                            print("Wait period complete. Starting incremental rotation search...")
                            socketio.emit('status_update', {'message': "Starting incremental rotation search..."})
                        else:
                            remaining_wait = SEARCH_WAIT_TIME - (current_time - search_start_time)
                            # Only print every few seconds to avoid spam
                            if int(remaining_wait) % 3 == 0 and int(remaining_wait * 10) % 30 < 1: 
                                print(f"Waiting for mango... {remaining_wait:.0f}s remaining")
                            send_enhanced_movement_command(0, 0, 0, 0) # Hover
                            
                    elif search_state == "rotating_incremental":
                        # Calculate how much we should have rotated in this segment
                        # Target for this specific rotation segment is ROTATION_INCREMENT_DEGREES
                        # Time elapsed in this specific rotation segment
                        time_in_this_rotation_segment = current_time - rotation_start_time 
                        degrees_rotated_this_segment = time_in_this_rotation_segment * ROTATION_SPEED_DEG_S

                        if degrees_rotated_this_segment < ROTATION_INCREMENT_DEGREES:
                            # Still rotating for this increment
                            print(f"Rotating: {degrees_rotated_this_segment:.1f}°/{ROTATION_INCREMENT_DEGREES}° (Total: {current_rotation_degrees + degrees_rotated_this_segment:.1f}°)")
                            send_enhanced_movement_command(0, 0, 0, math.radians(ROTATION_SPEED_DEG_S))
                        else:
                            # Completed this rotation increment
                            send_enhanced_movement_command(0, 0, 0, 0) # Stop rotation
                            search_state = "detecting"
                            detection_wait_start_time = current_time
                            current_rotation_degrees += ROTATION_INCREMENT_DEGREES # Accumulate total rotation
                            print(f"Completed {ROTATION_INCREMENT_DEGREES}° rotation. Total rotated: {current_rotation_degrees}°")
                            print(f"Pausing for {DETECTION_WAIT_TIME}s for detection...")
                            socketio.emit('status_update', {
                                'message': f"Rotated {current_rotation_degrees:.0f}°. Pausing for detection..."
                            })
                            
                    elif search_state == "detecting":
                        detection_wait_duration = current_time - detection_wait_start_time
                        remaining_detection_time = DETECTION_WAIT_TIME - detection_wait_duration
                        
                        if detection_wait_duration < DETECTION_WAIT_TIME:
                            if int(remaining_detection_time) % 2 == 0 and int(remaining_detection_time * 10) % 20 < 1:
                                print(f"Detection pause: {remaining_detection_time:.1f}s remaining at {current_rotation_degrees}°")
                            send_enhanced_movement_command(0, 0, 0, 0) # Ensure hover
                        else:
                            # Detection wait complete
                            if current_rotation_degrees >= FULL_ROTATION_DEGREES:
                                search_state = "moving_forward"
                                forward_movement_start_time = current_time
                                current_rotation_degrees = 0  # Reset for next search cycle after moving
                                print("360° rotation with pauses complete. Moving forward...")
                                socketio.emit('status_update', {'message': "Full rotation done. Moving to new search area..."})
                            else:
                                # Continue with next rotation increment
                                search_state = "rotating_incremental"
                                rotation_start_time = current_time # Reset start time for the new segment
                                # incremental_rotation_target remains ROTATION_INCREMENT_DEGREES for the next segment
                                print(f"Starting next rotation increment (to {current_rotation_degrees + ROTATION_INCREMENT_DEGREES}°).")
                                
                    elif search_state == "moving_forward":
                        movement_duration = current_time - forward_movement_start_time
                        if movement_duration < FORWARD_MOVEMENT_TIME:
                            distance_moved = movement_duration * FORWARD_MOVEMENT_SPEED
                            print(f"Moving forward: {distance_moved:.1f}m / {FORWARD_MOVEMENT_DISTANCE}m")
                            send_enhanced_movement_command(FORWARD_MOVEMENT_SPEED, 0, 0, 0)
                        else:
                            send_enhanced_movement_command(0, 0, 0, 0) # Stop forward movement
                            search_state = "rotating_incremental" # Start new rotation cycle
                            rotation_start_time = current_time
                            current_rotation_degrees = 0 # Reset rotation for the new spot
                            print("Forward movement complete. Starting new incremental rotation search cycle.")
                            socketio.emit('status_update', {'message': "Moved forward. Starting new search rotation."})
                                    
                except Exception as e:
                    print(f"Error in search pattern: {e}")
                    search_state = "idle" # Reset to idle on error
                    current_rotation_degrees = 0


        eventlet.sleep(0.1) # Loop frequency
    print("Enhanced mango tracking control loop stopped.")


# --- (Your existing Flask routes: /, video_feed, /command/arm, /command/disarm, etc.) ---

@app.route('/command/set_geofence', methods=['POST'])
def command_set_geofence():
    global geofence_polygon, geofence_active
    if not vehicle:
        return jsonify({"status": "error", "message": "Vehicle not connected"}), 500
    try:
        data = request.get_json()
        new_geofence = data.get('geofence')
        if not new_geofence or len(new_geofence) < 3:
            return jsonify({"status": "error", "message": "Invalid geofence data. Polygon must have at least 3 points."}), 400
        
        geofence_polygon = [[float(lat), float(lon)] for lat, lon in new_geofence]
        geofence_active = True
        print(f"Geofence set with {len(geofence_polygon)} points.")
        socketio.emit('geofence_status_update', {'status': 'Active', 'message': 'Geofence has been set.', 'geofence': geofence_polygon})
        return jsonify({"status": "success", "message": "Geofence set successfully"})
    except Exception as e:
        print(f"Error setting geofence: {e}")
        return jsonify({"status": "error", "message": f"Failed to set geofence: {str(e)}"}), 500

@app.route('/command/clear_geofence', methods=['POST'])
def command_clear_geofence():
    global geofence_polygon, geofence_active
    geofence_polygon = None
    geofence_active = False
    print("Geofence cleared.")
    socketio.emit('geofence_status_update', {'status': 'Cleared', 'message': 'Geofence has been cleared.'})
    return jsonify({"status": "success", "message": "Geofence cleared successfully"})

# SocketIO event for frontend to query current geofence status
@socketio.on('query_geofence_status')
def handle_query_geofence_status():
    global geofence_active, geofence_polygon
    if geofence_active and geofence_polygon:
        emit('geofence_status_update', {'status': 'Active', 'message': 'Geofence is currently active.', 'geofence': geofence_polygon})
    else:
        emit('geofence_status_update', {'status': 'Not Defined', 'message': 'No geofence is currently defined.'})


# --- Graceful Shutdown & Main Execution ---
# ... (Your existing signal_handler and if __name__ == '__main__': block) ...
# Ensure model is loaded in __main__ if not done globally before
if __name__ == '__main__':
    # ...
    if not model_loaded: # Redundant if called above, but safe
        print("Attempting to load model from __main__ as it wasn't loaded earlier.")
        load_optimized_yolo_model()
    # ... rest of your __main__
    video_writer = None # Ensure these are initialized for main context
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

    if not initialize_jetson_camera():
        print("⚠ CRITICAL: Failed to initialize camera. Video stream and CV features will be disabled.")
    
    if not model_loaded: # Check after attempts
        print("⚠ WARNING: YOLO Model not loaded. Mango detection and tracking will be disabled.")

    print("🚁 Attempting to connect to vehicle in background...")
    eventlet.spawn(connect_vehicle_jetson) 

    print("💡 Starting background threads using eventlet.spawn for cooperative multitasking...")
    eventlet.spawn(jetson_video_processing_loop) 
    eventlet.spawn(mango_tracking_loop)          
    eventlet.spawn(jetson_memory_monitor)        

    print(f"🌍 Starting Flask-SocketIO server on http://0.0.0.0:5000")
    print("ℹ  Press Ctrl+C to exit.")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, use_reloader=False, debug=False)
    except Exception as e:
        print(f"💥 Failed to start Flask-SocketIO server: {e}")
    finally:
        print("\nApplication is shutting down...")
        running = False 

        print("Waiting for threads to complete (max 2 seconds)...")
        eventlet.sleep(2.0) 
        
        if is_recording and video_writer: # Check if video_writer is not None
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
        
        if cap: # Check if cap is not None
            print("Releasing camera...")
            cap.release()
            print("Camera released.")
        
        gc.collect()
        print("Cleanup complete. Exiting now.")