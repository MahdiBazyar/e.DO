from collections import deque
from imutils.video import VideoStream
import numpy as np

import cv2
import imutils
import time
import math
import sys
import os

#Adding file to Python PATH to callable elsewhere
#sys.path.append(".")


#class ObjectDetection():

  #def __init__(self):
    #pass:

#def getObjectLocation():
pts = deque(maxlen=32)
cap = cv2.VideoCapture(1) #number is which camera on your system you want to use
cap.set(3, 800)
cap.set(4, 600)

#Allow the camera or video file to warm up
time.sleep(2.0)

#Math logic to find the height and length
depthOrAdjacent = 107

verticalFieldOfView = 43.3
verticalFieldOfViewHalf = verticalFieldOfView / 2
baseOrOppositeVFOV = math.tan(math.radians(verticalFieldOfViewHalf)) * depthOrAdjacent

horizontalFieldOfView = 70.42
horizontalFieldOfViewHalf = horizontalFieldOfView / 2
baseOrOppositeHFOV = math.tan(math.radians(horizontalFieldOfViewHalf)) * depthOrAdjacent

  
heightIn = baseOrOppositeVFOV * 2
widthIn = baseOrOppositeHFOV * 2
print(heightIn, widthIn) #Debug Statement
count = 0
runCamera = True
os.system('cls||clear')   #to clear the termnial screen
print("===============")
print("Camera Terminal")
print("===============")
print("Press Ctrl + C to exit.")

while True:
  while(runCamera):
    #print(count)
   # if(count < 0):
    #  temp = raw_input("Type (y) to get object location or (n) to exit: ")
   #   if(temp == "y"):
     #   count = 20
    #  if(temp == "n"):
    #    exit(0)
   # while(count >=0):
   #   count = count -1
      #Ask the user to continue
      #response = raw_input("To terminate put 'n': \n")
      #if(response == "n"):
        #cap.release()
        #cv2.destroyAllWindows()
        #break;
      count = count + 1
      ret, img = cap.read()
      #start code here
      #Location of variable changed to fix results
      #Croping image
      #lefty, righty, leftx,rightx
      #img = img[100:-20,10:-210]
      img = img[10:-10,10:-210]

      height, width, _ = img.shape
      inchesPerPixelH = heightIn / height #'''heightIn''' 
      inchesPerPixelW = widthIn / width	#'''widthIn'''
      #ends here
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      #
      # Convert to grayscale. 
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
      # Blur using 3 * 3 kernel. 
      gray_blurred = cv2.blur(gray, (3, 3)) 
    
      # Apply Hough transform on the blurred image. 
      detected_circles = cv2.HoughCircles(gray_blurred,  
                     	cv2.HOUGH_GRADIENT, 0.1, 50, param1 = 50, 
                 	param2 = 40, minRadius = 0, maxRadius = 55) 

    # Draw circles that are detected. 
      if detected_circles is not None: 
    
      # Convert the circle parameters a, b and r to integers. 
          detected_circles = np.uint16(np.around(detected_circles)) 
    
          for pt in detected_circles[0, :]: 
              a, b, r = pt[0], pt[1], pt[2] 
    
              # Draw the circumference of the circle. 
              cv2.circle(img, (a, b), r, (204, 193, 255), 2) 
    
              # Draw a small circle (of radius 1) to show the center. 
              cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
              pts.appendleft((a, b))
              #My code starts here.
              # loop over the set of tracked points
              for i in np.arange(1, len(pts)):
                  # coordinates to display/ forward to the ai
                  diff = 7.25#8#0; #8
                  #print(r) 
                  x = round(diff + (pts[0][0] * inchesPerPixelH), 2)
                  y = round(((heightIn/2) - (pts[0][1] * inchesPerPixelH) + 4.5), 2)
                  z = round((r * (inchesPerPixelH)) * 2, 2)
                  #BEING: MATH TO CALCULATE DEPTH OF OBJECT - Hawraa Banoon 
                  #Source: https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
                  if (r > 0):
                    #4 Cm by 4 Cm object 
                    #Depth Perception calculated by: D = (W x F) / P 
                    #D = Depth, F = Focal Length, P = pixels at certain length 
                    #In this case F = (P X D) / W , Where P = 16 pixels when on table, 
                    #D is depth of table (107 cm) and W is 4cn object 
                    newZ = round((4 * 856) / (r * 2), 2) 
                    newZ = 107 - newZ
                  else:
                    newZ = 6
                    #print(newZ) 
                  #END: MATH TO CALCULATE DEPTH OF OBJECT - Hawraa Banoon 

                  newX = round(math.sqrt((x*x) + (y*y)), 2)
                  print("x:", x) 
                  print("y:", y) 
                  xy = math.degrees(math.acos(((x*x) + (newX*newX) - (y*y)) / (2 * x * newX)))
                  print("newX:", newX) 
                  #print("xy:", xy)
                  newXY = math.degrees(math.atan((y/x)))
                  print("newXY:", newXY)
                  #cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 3)

                  # show the image coordinates
                  if(y<0):
                    xy = xy * -1 
                  cv2.putText(img, "X: {}, Y: {}, OW: {}".format(x, y, z),
                  	(10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                  	0.35, (0, 0, 255), 1)
                  if(count % 1 == 0): 
                    if(x > 9):
                      f = open("newX.txt", "w")
                      print("Updated newX.txt") 
                      f.write('%f' % newX)
                      f.close()               
                      f = open("xy.txt", "w")
                      print("Updated xy.txt") 
                      f.write('%f' % xy)
                      f.close()
                    #Write to text file the Z value if it is greater than Zero so it doesn't hit table - Hawraa Banoon 
                    if(newZ > 7.0):
                      f = open("newZ.txt", "w")
                      print("Updated newZ.txt") 
                      f.write('%f' % newZ)
                      f.close()
                    else: 
                      newZ = 6.0
                      f = open("newZ.txt", "w")
                      print("Updated newZ.txt") 

                      f.write('%f' % newZ)
                      f.close()

      cv2.imshow("Detected Circle", img)
      if(count % 50 ==0 and detected_circles is not None):
        time.sleep(10)
        cv2.imshow("Detected Circle", gray_blurred)
      #cv2.imshow("gray", gray)
      k = cv2.waitKey(30) & 0xff
      if k == 27:
           cap.release()
     # close all windows
           cv2.destroyAllWindows()
