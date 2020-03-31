''' Configuration file for color_finalP1.py; keep in same directory '''

# =============== Camera Initialization ===============
capture_source = -1
capture_width = 800
capture_height = 600
warmup_time = 2.0
# =====================================================


# ======= Camera Measurements ======= 
surface_to_cam_cm = 107
cam_v_fov_cm = 43.3
cam_h_fov_cm = 70.42
# ===================================


# == Frames to capture ==
number_of_frames = 20
# =======================


# ======= Image Crop =======
crop_row_start = 10
crop_row_end= -10
crop_col_start = 10
crop_col_end = -210
# ==========================


# ========== HSV Bounds ==========
blue_lower_hue = 94
blue_lower_saturation = 80
blue_lower_value = 2
blue_upper_hue = 126
blue_upper_saturation = 255
blue_upper_value = 255

green_lower_hue = 25
green_lower_saturation = 52
green_lower_value = 72
green_upper_hue = 102
green_upper_saturation = 255
green_upper_value = 255

red_lower_hue = 0
red_lower_saturation = 100
red_lower_value = 100
red_upper_hue = 10
red_upper_saturation = 255
red_upper_value = 255

red_lower_hue_2 = 160
red_lower_saturation_2 = 100
red_lower_value_2 = 100
red_upper_hue_2 = 179
red_upper_saturation_2 = 255
red_upper_value_2 = 255
# ================================


# ========= Blur =========
blur_kernel_size = 3
# ========================


# ====== Circle Detection Parameters ======
inverse_resolution_ratio = 0.1
center_min_distance = 20
edge_upper_threshold = 40
center_detect_threshold = 23
min_radius = 6
max_radius = 60
# =========================================


# ======== Circle BGR Values ========
detected_circle_b = 0
detected_circle_g = 255
detected_circle_r = 0

detected_center_b = 0
detected_center_g = 0
detected_center_r = 255
# ===================================


# ========== Math ==========
distance_buffer = 7.25
angle_buffer = 4.5

object_size_cm = 4
default_surface_depth = 6
min_object_distance = 9
# ==========================
