import time
import math
import random 

newX = 30
newZ = 6
xy = -35

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
  newX= random.randint(30, 75)
  newZ= random.randint(6, 75)
  xy= random.randint(-40, 40)

  #newZ = newZ + 1
  #xy = xy + 1 
  #newX = newX + 1

  #if (newX == 70): 
    #newX = 30

  #if (xy == 40):
    #xy = -35

  time.sleep(3600)

