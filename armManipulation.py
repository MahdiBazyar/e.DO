# ================================== Arm Manipulation ==========================================
#  Developers: Wayne State University Senior Capstone Project Students
#              Fall 2019: Adel Mohamed, Corey Simms, Mahmoud Elmasri, & Tyler Riojas 
#              Winter 2020: Hawraa Banoon, Nathaniel Smith, Kristina Stevoff, & Mahdi Bazyar
#  Directory:  /home/nvidia/jetson_ctrl_ws/src/edoProjectFinal
#  Interface:  Arm Terminal
#  Purpose:    Presents the user with various options to move e.DO
#  Inputs:     2 Text files:
#              results.txt
#              color.txt
#  Outputs:    None    
# ==============================================================================================


# ============================ IMPORTS ============================
import os
import sys
from pynput import keyboard
from termios import tcflush, TCIOFLUSH
import time
sys.path.append('/home/nvidia/jetson-reinforcement/build/aarch64/config/')
import config_load as cfg
from manipulation.manipulationWQueue import *
# ================================================================= [end imports]

sys.path.append(cfg.main_project_directory)

# ===================================== FUNCTIONS =====================================

# Function that gets a winning joint angle from Gazebo
def get_joint_angles():
    joint_data = []

    f = open(cfg.joint_angle_file, "r")

    for i in range(6):
      joint_data.append(int(f.readline()))

    # Debug: Print angles gotten from text file
   
    return joint_data

# Function that gets the joint angles to reach the bucket
def put_in_bucket():
    joint_data_bucket = []
    color = "" 
    c = open(cfg.object_color_file, "r")
    color = c.read()
    print(color)
    
    # FULL DISCLOSURE: e.DO's movements to the buckets are hard-coded here 
    # to work with our current setup in the lab.
    if (color == "blue"):
        print("Blue detected")
        joint_data_bucket = cfg.blue_bucket_angles
    if (color == "green"): 
        print("Green detected")
        joint_data_bucket = cfg.green_bucket_angles
    if (color == "red"):
        print("Red detected")
        joint_data_bucket = cfg.red_bucket_angles

    # Debug: Print angles gotten from text file
   
    return joint_data_bucket
    
# Function that performs grab attempt   
def move_arm():
      
    joint_data = get_joint_angles()

   # man.setGripper(80)      # Max gripper opening
    joint_data.append(80)
    joint_data.append(0.0)
    joint_data.append(0.0)
    joint_data.append(0.0)
    
    # Debug: Prints joint angles
    print "Joint data =", joint_data, "\n"
    
# ================================== Vectors ==================================
# "Vectors," in this context, are intermediate angles for each joint
# between their final values, divided evenly by the number of vectors.
 
    joint_data_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]      # empty copy of joint_data that
                                                                                # will contain intermediate moves
    
    vector_number = cfg.number_of_vectors   # More vectors should amount to smoother movement; must be an integer
                                             # Currently set to 1 until "fly" feature can be enabled to connect movements without pausing
                                             # Alternatively, use 1000+ vectors and figure out how to speed up e.DO.
                        
    for i in range (0, vector_number):  # Angles progressively build up to the joint_data values.
        for j in range (0, 10):
            if (joint_data_vector[j] < joint_data[j] and joint_data[j] >= 0):   # Handles positive angles
                joint_data_vector[j] += (joint_data[j] / float(vector_number))
                        
            if (joint_data_vector[j] > joint_data[j] and joint_data[j] < 0):    # Handles negative angles
                joint_data_vector[j] += (joint_data[j] / float(vector_number))
    
        # Debug: Prints the value of each vector; they should build up to the original values in joint_data
        if (cfg.print_vector_data):
            print "Vector", i + 1, '\n', joint_data_vector, '\n'
        
        # Moves each joint the new intermediate angle each iteration
        man.jointMove(joint_data_vector)
 
# **This functionality has only been implemented here; 
# extenuating circumstances have prevented us from fleshing it out. It may be best
# to make an entirely new function of this in the future so all of e.DO's moves use it.

# ============================================================================= [end vectors]    
     

    joint_data[6] = 0           # Close gripper
    man.jointMove(joint_data)
    
    # Lift object
    joint_data[1] = joint_data[1] - cfg.lift_object      # Raise joint 2 slightly
    man.jointMove(joint_data)
    
    # Put object back down
    joint_data[1] = joint_data[1] + cfg.lift_object      # Lower joint 2
    man.jointMove(joint_data)

    # Release object and return to home position
    joint_data[6] = 80                      # Open gripper
    man.jointMove(joint_data)
    joint_data[6] = 0                       # Close gripper
    man.jointMove()                         # Go home
    
def move_arm_bucket():

    joint_data = get_joint_angles()

    joint_data_bucket = put_in_bucket()
    print(joint_data_bucket)
    man.setGripper(80)

    joint_data.append(80)
    joint_data.append(0.0)
    joint_data.append(0.0)
    joint_data.append(0.0)

    # Move to and grab object
    man.jointMove(joint_data)
    joint_data[6] = 0
    man.jointMove(joint_data)
    
    # Home Position 
    man.jointMove()
    
    # Move to bucket 
    man.jointMove(joint_data_bucket)
    
    # Release object
    joint_data_bucket[6] = 80
    man.jointMove(joint_data_bucket)
    
    # Close gripper 
    joint_data_bucket[6] = 0
    man.jointMove(joint_data_bucket)
    
    # Home Position 
    man.jointMove()

# Keyboard listener function to start e.DO movement
def start_key(key):
    if key == keyboard.Key.space:
        print("\n< Continuously running e.DO... >\n")
        return False

# Keyboard listener function to stop e.DO movement
def stop_key(key):
    if key == keyboard.Key.esc:
        print("\n< Stopping e.DO... >\n")
        return False

# ===================================================================================== [end functions]

# ================================================== MAIN ==================================================

if __name__ == '__main__':

  rospy.init_node("sp_demo1", anonymous = True)
  man = Manipulation()

  os.system('cls||clear')   # clear terminal screen
  


  while True:
      
    # Flush input buffer
    sys.stdout.flush();
    tcflush(sys.stdin, TCIOFLUSH)
      
    print("=================================================================")
    print("                         Arm Terminal                            ")
    print("=================================================================") 
    
    response = raw_input("0: Exit\n1: Single grab attempt\n2: Continuous grab attempts\n3: Single bucket drop\n4: Continuous bucket drops\n5: Dance\n6: Home position\n\n")    # Action menu 

    while((response != "0") and (response != "1") and (response != "2") and (response != "3") and (response != "4") and (response != "5") and (response != "6") and (response != "b")):     # Input validation loop
        response = (raw_input("Please select a valid option.\n\n0: Exit\n1: Single grab attempt\n2: Continuous grab attempts\n3: Single bucket drop\n4: Continuous bucket drops\n5: Dance\n6: Home position\n\n"))
    
    if(response == "0"):        # Exit arm terminal
        print("< Exiting... >\n")
        time.sleep(cfg.system_message_sleep)
        exit(0)
      
    elif(response == "1"):      # Single grab attempt
      print("< Moving e.DO to object once... >\n")
      time.sleep(cfg.system_message_sleep)
      move_arm()
      
    elif(response == "2"):      # Continuous grab attempts
        

        # results.txt comparisons to determine if e.DO has won in the simulation
        current_angles = get_joint_angles()
        updated_angles = current_angles

        print("Press Space to start and Esc to stop.")
        
        #  Wait for Space key to start e.DO
        with keyboard.Listener (on_press = start_key) as listener:
            listener.join()
        
        # Wait for Esc key to stop e.DO and return to menu
        with keyboard.Listener(on_press = stop_key) as listener:
            print("e.DO will dance while waiting for a win in the simulation.\n")
            while True:         # Continuously move e.DO

                while (updated_angles == current_angles):   # Runs until there is a win in Gazebo
                    man.dance()     # Dance, e.DO, dance!
                    updated_angles = get_joint_angles()     # Breaks dance loop when angles have been updated after a win
                                                            # Otherwise, keep on dancin'...
                    if (updated_angles != current_angles):
                        break
                    time.sleep(cfg.continuous_move_sleep)  # Wait n seconds to avoid building up a large queue of movements for e.DO


             #   time.sleep(cfg.continuous_move_sleep)
                move_arm()  # Move according to the angle updates in results.txt
                current_angles = updated_angles # Reinitialize current_angles to execute dance loop again

                                
                if (not listener.running):      # Stop e.DO, return to home position,

                    man.jointMove()
                    break
                
    elif(response == "3"):          # Single bucket
        print("< Moving e.DO to bucket once... >\n")
        time.sleep(cfg.system_message_sleep)
        move_arm_bucket()
        
        
    elif(response == "4"):          # Continuous bucket
        print("< Continuously moving e.DO to bucket... >\n")    
        time.sleep(cfg.system_message_sleep)
        
        # results.txt comparisons to determine if e.DO has won in the simulation
        current_angles = get_joint_angles()
        updated_angles = current_angles

        print("Press Space to start and Esc to stop.")
            
        #  Wait for Space key to start e.DO
        with keyboard.Listener (on_press = start_key) as listener:
            listener.join()
            
        # Wait for Esc key to stop e.DO and return to menu
        with keyboard.Listener(on_press = stop_key) as listener:
            print("e.DO will dance while waiting for a win in the simulation.\n")
            while True:         # Continuously move e.DO
                while (updated_angles == current_angles):   # Runs until there is a win in Gazebo
                    man.dance()     # Dance, e.DO, dance!
                    updated_angles = get_joint_angles()     # Breaks dance loop when angles have been updated after a win
                                                            # Otherwise, keep on dancin'...
                    if (updated_angles != current_angles):
                        break
                    time.sleep(cfg.continuous_move_sleep)  # Wait n seconds to avoid building up a large queue of movements for e.DO

             #   time.sleep(cfg.continuous_move_sleep)
                move_arm_bucket()
                current_angles = updated_angles # Reinitialize current_angles to execute dance loop again

                if (not listener.running):      # Stop e.DO, return to home position,
                    man.jointMove()
                    break


    elif(response == "5"):          # Dance
        print("< e.DO, the dancing robot arm >\n")
        
        print("Press Space to start and Esc to stop.")
            
        #  Wait for Space key to start e.DO
        with keyboard.Listener (on_press = start_key) as listener:
            listener.join()
        # Wait for Esc key to stop e.DO and return to menu
        with keyboard.Listener(on_press = stop_key) as listener:
            while True:
                man.dance()
                time.sleep(cfg.dance_sleep)   # Wait n seconds before executing next loop iteration
                                               # to prevent building up large queue of dance moves
                
                if (not listener.running):     # Stop e.DO, return to home position,
                    man.jointMove()
                    break
            
            
    elif(response == "6"):          # Home position
        print("< Returning e.DO to home position... >\n")
        man.jointMove()
        time.sleep(cfg.system_message_sleep)


    elif(response == "b"):          # Hidden function to move only to bucket for testing angles
        print("< Moving to bucket... >\n")
        time.sleep(cfg.system_message_sleep)
        
        joint_data_bucket = put_in_bucket()
        man.jointMove(joint_data_bucket)
        joint_data_bucket[6] = 80
        man.jointMove(joint_data_bucket)
        joint_data_bucket[6] = 0
        man.jointMove(joint_data_bucket)
        man.jointMove()

    else:                           # Hmm....
        print("< This should never execute... >\n")
        time.sleep(cfg.system_message_sleep)


            
# ========================================================================================================== [end main]
