''' Configuration file for armManipulation.py; keep in same directory '''

# =================================== Directories ===================================
main_project_directory = "/home/nvidia/jetson-reinforcement/build/aarch64"
joint_angle_file = "/home/nvidia/jetson-reinforcement/build/aarch64/bin/results.txt"
object_color_file = "/home/nvidia/jetson-reinforcement/build/aarch64/bin/color.txt"
# ====================================================================================


# ===== Bucket movement angles =====
blue_bucket_angles = [
     40.0,      # Joint 1
    -55.0,      # Joint 2
    -60.0,      # Joint 3
    -80.0,      # Joint 4
    -75.0,      # Joint 5
     95.0,      # Joint 6
      0.0, 0.0, 0.0, 0.0
]

green_bucket_angles = [
     30.0,      # Joint 1
    -70.0,      # Joint 2
    -20.0,      # Joint 3
    -20.0,      # Joint 4
    -50.0,      # Joint 5
     20.0,      # Joint 6
      0.0, 0.0, 0.0, 0.0
]

red_bucket_angles = [
      5.0,      # Joint 1
    -70.0,      # Joint 2
    -10.0,      # Joint 3
    -25.0,      # Joint 4
    -73.0,      # Joint 5
     20.0,      # Joint 6
      0.0, 0.0, 0.0, 0.0
]
# ==================================


# ======== Vectors ========
number_of_vectors = 1
print_vector_data = True
# =========================


# === Object lift amount ===
lift_object = 15.0
# ==========================


# ========== Sleep timers ==========
system_message_sleep = 0.25
continuous_move_sleep = 12.0
dance_sleep = 10.0
# ==================================

