# Updates object position in simulation while training
# belongs in direcrory /home/nvidia/jetson-reinforcement/build/aarch64

import time
import math
import random 

# Initial object position
newX = 30   # Distance
xy = -40    # Angle
newZ = 6    # Height

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
  
  # ========== Old functionality which caused the object to move linearly ==========
  #newX = newX + 1
  #xy = xy + 1 
  #newZ = newZ + 1

  #if (newX == 75):     # Resets object distance once it gets out of range
    #newX = 30

  #if (xy == 40):       # Resets object angle once it gets out of range
    #xy = -35
    
   #if (newZ == 75):    # Resets object height once it gets out of range
        #newZ = 6
  # ================================================================================
  
  # Randomizes object position within defined bounds
  newX = random.randint(30, 75)
  xy = random.randint(-40, 40)
  newZ = random.randint(6, 75)

  time.sleep(3600)      # Sleeps for 1 hour (3600 seconds) to train
                        # e.DO at current object position

