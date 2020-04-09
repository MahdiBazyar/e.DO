# ================================== Configuration Loader ======================================
#  Developers: Wayne State University Senior Capstone Project Students          
#              Winter 2020: Hawraa Banoon, Nathaniel Smith, Kristina Stevoff, & Mahdi Bazyar
#  Directory:  /home/nvidia/jetson-reinforcement/build/aarch64/config/
#  Interface:  None
#  Purpose:    Loads the contents of config.json to be used in the main files
#  Inputs:     1 JSON file:
#              config.json
#  Outputs:    None    
# ==============================================================================================

import json

with open("/home/nvidia/jetson-reinforcement/build/aarch64/config/config.json") as f:
    config = json.load(f)

# ========================================= colorFinalP1.py =========================================
capture_source = config["color_finalP1"]["camera_initialization"]["init_capture_source"]
capture_width = config["color_finalP1"]["camera_initialization"]["init_capture_width"]
capture_height = config["color_finalP1"]["camera_initialization"]["init_capture_height"]
warmup_time = config["color_finalP1"]["camera_initialization"]["init_warmup_time"]

surface_to_cam_cm = config["color_finalP1"]["camera_measurements"]["meas_surface_to_cam_cm"]
cam_v_fov_cm = config["color_finalP1"]["camera_measurements"]["meas_cam_v_fov_cm"]
cam_h_fov_cm = config["color_finalP1"]["camera_measurements"]["meas_cam_h_fov_cm"]

number_of_frames = config["color_finalP1"]["frames"]["frames_num_to_capture"]

crop_row_start = config["color_finalP1"]["image_crop"]["crop_row_start"]
crop_row_end = config["color_finalP1"]["image_crop"]["crop_row_end"]
crop_col_start = config["color_finalP1"]["image_crop"]["crop_col_start"]
crop_col_end = config["color_finalP1"]["image_crop"]["crop_col_end"]

blue_lower_hue = config["color_finalP1"]["hsv_bounds"]["hsv_blue_lower_hue"]
blue_lower_saturation = config["color_finalP1"]["hsv_bounds"]["hsv_blue_lower_saturation"]
blue_lower_value = config["color_finalP1"]["hsv_bounds"]["hsv_blue_lower_value"]
blue_upper_hue = config["color_finalP1"]["hsv_bounds"]["hsv_blue_upper_hue"]
blue_upper_saturation = config["color_finalP1"]["hsv_bounds"]["hsv_blue_upper_saturation"]
blue_upper_value = config["color_finalP1"]["hsv_bounds"]["hsv_blue_upper_value"]

green_lower_hue = config["color_finalP1"]["hsv_bounds"]["hsv_green_lower_hue"]
green_lower_saturation = config["color_finalP1"]["hsv_bounds"]["hsv_green_lower_saturation"]
green_lower_value = config["color_finalP1"]["hsv_bounds"]["hsv_green_lower_value"]
green_upper_hue = config["color_finalP1"]["hsv_bounds"]["hsv_green_upper_hue"]
green_upper_saturation = config["color_finalP1"]["hsv_bounds"]["hsv_green_upper_saturation"]
green_upper_value = config["color_finalP1"]["hsv_bounds"]["hsv_green_upper_value"]

red_lower_hue = config["color_finalP1"]["hsv_bounds"]["hsv_red_lower_hue"]
red_lower_saturation = config["color_finalP1"]["hsv_bounds"]["hsv_red_lower_saturation"]
red_lower_value = config["color_finalP1"]["hsv_bounds"]["hsv_red_lower_value"]
red_upper_hue = config["color_finalP1"]["hsv_bounds"]["hsv_red_upper_hue"]
red_upper_saturation = config["color_finalP1"]["hsv_bounds"]["hsv_red_upper_saturation"]
red_upper_value = config["color_finalP1"]["hsv_bounds"]["hsv_red_upper_value"]

red_lower_hue_2 = config["color_finalP1"]["hsv_bounds"]["hsv_red_lower_hue_2"]
red_lower_saturation_2 = config["color_finalP1"]["hsv_bounds"]["hsv_red_lower_saturation_2"]
red_lower_value_2 = config["color_finalP1"]["hsv_bounds"]["hsv_red_lower_value_2"]
red_upper_hue_2 = config["color_finalP1"]["hsv_bounds"]["hsv_red_upper_hue_2"]
red_upper_saturation_2 = config["color_finalP1"]["hsv_bounds"]["hsv_red_upper_saturation_2"]
red_upper_value_2 = config["color_finalP1"]["hsv_bounds"]["hsv_red_upper_value_2"]

blur_kernel_size = config["color_finalP1"]["blur"]["blur_kernel_size"]

inverse_resolution_ratio = config["color_finalP1"]["circle_detection_parameters"]["cdparam_inverse_resolution_ratio"]
center_min_distance = config["color_finalP1"]["circle_detection_parameters"]["cdparam_center_min_distance"]
edge_upper_threshold = config["color_finalP1"]["circle_detection_parameters"]["cdparam_edge_upper_threshold"]
center_detect_threshold = config["color_finalP1"]["circle_detection_parameters"]["cdparam_center_threshold"]
min_radius = config["color_finalP1"]["circle_detection_parameters"]["cdparam_min_radius"]
max_radius = config["color_finalP1"]["circle_detection_parameters"]["cdparam_max_radius"]

detected_circle_b = config["color_finalP1"]["circle_bgr"]["cbgr_circle_blue"]
detected_circle_g = config["color_finalP1"]["circle_bgr"]["cbgr_circle_green"]
detected_circle_r = config["color_finalP1"]["circle_bgr"]["cbgr_circle_red"]

detected_center_b = config["color_finalP1"]["circle_bgr"]["cbgr_center_blue"]
detected_center_g = config["color_finalP1"]["circle_bgr"]["cbgr_center_green"]
detected_center_r = config["color_finalP1"]["circle_bgr"]["cbgr_center_red"]

distance_buffer = config["color_finalP1"]["math"]["math_distance_buffer"]
angle_buffer = config["color_finalP1"]["math"]["math_angle_buffer"]
object_size_cm = config["color_finalP1"]["math"]["math_object_size_cm"]
default_surface_depth = config["color_finalP1"]["math"]["math_default_surface_depth"]
min_object_distance = config["color_finalP1"]["math"]["math_min_object_distance"]
# ===================================================================================================

#a = config["color_finalP1"]
#print "color_finalP1.py contents:\n\n", a


# ========================================= armManipulation.py ======================================
main_project_directory = config["armManipulation"]["directories"]["dir_main"]
joint_angle_file = config["armManipulation"]["directories"]["dir_joint_angles"]
object_color_file = config["armManipulation"]["directories"]["dir_obj_color"]

blue_bucket_angles = config["armManipulation"]["bucket_angles"]["angles_blue"]
green_bucket_angles = config["armManipulation"]["bucket_angles"]["angles_green"]
red_bucket_angles = config["armManipulation"]["bucket_angles"]["angles_red"]

number_of_vectors = config["armManipulation"]["vectors"]["vec_number"]
print_vector_data = config["armManipulation"]["vectors"]["vec_print_data"]

lift_object = config["armManipulation"]["object_lift"]["obj_lift"]

system_message_sleep = config["armManipulation"]["sleep_timers"]["sleep_sys_msg"]
continuous_move_sleep = config["armManipulation"]["sleep_timers"]["sleep_cont_move"]
dance_sleep = config["armManipulation"]["sleep_timers"]["sleep_dance"]
# ===================================================================================================

#b = config["armManipulation"]
#print "armManipulation.py contents:\n\n", b


# ========================================= test.py ================================================
x_min = config["test"]["object_boundaries"]["bound_x_min"]
x_max = config["test"]["object_boundaries"]["bound_x_max"]
y_min = config["test"]["object_boundaries"]["bound_y_min"]
y_max = config["test"]["object_boundaries"]["bound_y_max"]
z_min = config["test"]["object_boundaries"]["bound_z_min"]
z_max = config["test"]["object_boundaries"]["bound_z_max"]

mode = config["test"]["repositioning_method"]["rmethod_mode"]

update_interval = config["test"]["repositioning_time"]["rtime_update_interval"]
# ===================================================================================================

#c = config["test"]
#print "test.py contents:\n\n", c
