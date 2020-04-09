# ================================== Test File! =============================================
#  Developers: Wayne State University Senior Capstone Project Students
#              Winter 2020: Hawraa Banoon, Nathaniel Smith, Kristina Stevoff, & Mahdi Bazyar
#  Directory:  /home/nvidia/jetson-reinforcement/build/aarch64
#  Interface:  Stores information about object's position for Gazebo simulation  
#  Purpose:    This file updates the object's position in simulation model
#              while we are training the virtual e.DO              
#  Inputs:     None           
#  Outputs:    3 Text Files 
#              newX.txt  (The distance of object from base of e.DO in cm) 
#              xy.txt    (The angle that the object is in relation to e.DO in degree) 
#              newZ.txt  (The depth of the object in cm) 
# ================================== Test File! ==============================================

import time
import math
import random
import sys
sys.path.append('/home/nvidia/jetson-reinforcement/build/aarch64/config/')
import config_load as cfg

# Initial object position
newX = cfg.x_min   # Distance
xy = cfg.y_min     # Angle
newZ = cfg.z_min   # Height

# Writes object coordinates to text files to be read by Gazebo
while True: 
  f = open("newX.txt", "w")
  print("Updated newX.txt") 
  f.write('%f' % newX)
  f.close() 
                
  f = open("xy.txt", "w")
  print("Updated xy.txt") 
  f.write('%f' % xy)
  f.close()
  
  f = open("newZ.txt", "w")
  print("Updated newZ.txt") 
  f.write('%f' % newZ)
  f.close()
  
  
  time.sleep(cfg.update_interval)      # Sleeps for 1 hour (3600 seconds) to train
                                       # e.DO at current object position
                                       
  # Moves the object linearly
  if (cfg.mode == 1):
      newX = newX + 1
      xy = xy + 1 
      newZ = newZ + 1

      if (newX == cfg.x_max):     # Resets object distance once it gets out of range
        newX = cfg.x_min

      if (xy == cfg.y_max):       # Resets object angle once it gets out of range
        xy = cfg.y_min
        
       if (newZ == cfg.z_max):    # Resets object height once it gets out of range
            newZ = cfg.z_min
  
  # Randomizes object position within defined bounds
  elif (cfg.mode == 2):
      newX = random.randint(cfg.x_min, cfg.x_max)
      xy = random.randint(cfg.y_min, cfg.y_max)
      newZ = random.randint(cfg.z_min, cfg.z_max)

