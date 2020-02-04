from collections import deque
from imutils.video import VideoStream
import numpy as np

import cv2
import imutils
import time
import math
import sys

#Adding file to Python PATH to callable elsewhere
#sys.path.append(".")


#class ObjectDetection():

  #def __init__(self):
    #pass:

#def getObjectLocation():
pts = deque(maxlen=32)
cap = cv2.VideoCapture(1) #number is which camera on your system you want to use

# cap.set is used to set the exposure time for (3) width and (4) height
image_width = 800
image_height = 600
cap.set(3, image_width)
cap.set(4, image_height)

#Allow the camera or video file to warm up
time.sleep(2.0)

#Math logic to find the height and length of the object detected by open CV

depthOfCamera = 107  #centimeters from floor to camera

verticalFieldOfView_degrees = 43.3
verticalFieldOfViewHalf = verticalFieldOfView_degrees / 2
baseOrOppositeVFOV = math.tan(math.radians(verticalFieldOfViewHalf)) * depthOfCamera

horizontalFieldOfView_degrees = 70.42
horizontalFieldOfViewHalf = horizontalFieldOfView_degrees / 2
baseOrOppositeHFOV = math.tan(math.radians(horizontalFieldOfViewHalf)) * depthOfCamera

heightIn = baseOrOppositeVFOV * 2
widthIn = baseOrOppositeHFOV * 2
print(heightIn, widthIn) #Debug Statement


count = 20
print("===============")
print("Camera Terminal")
print("===============")

while True:
    #print(count)
    if(count < 0):
      temp = raw_input("Type (y) to get object location or (n) to exit: ")
      if(temp == "y"):
        count = 20
      if(temp == "n"):
        exit(0)
    while(count >=0):
      count = count -1
      #Ask the user to continue
      #response = raw_input("To terminate put 'n': \n")
      #if(response == "n"):
        #cap.release()
        #cv2.destroyAllWindows()
        #break;
      #count = count - 1
      ret, img = cap.read()
      #start code here
      #Location of variable changed to fix results
      #Croping image
      #lefty, righty, leftx,rightx
      #img = img[100:-20,10:-210]
      img = img[10:-10,10:-210]

      height, width, _ = img.shape
      inchesPerPixelH = heightIn / height   #'''heightIn'''
      inchesPerPixelW = widthIn / width	    #'''widthIn'''

      #ends here
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      #
      # Convert to grayscale. 
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
      # Blur using 3 * 3 kernel. 
      gray_blurred = cv2.blur(gray, (3, 3)) 
    
      # Apply Hough transform on the blurred image. 
      detected_circles = cv2.HoughCircles(gray_blurred,  
                     	cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                 	param2 = 30, minRadius = 1, maxRadius = 40) 
    
    # Draw circles that are detected. 
      if detected_circles is not None: 
    
      # Convert the circle parameters a, b and r to integers. 
          detected_circles = np.uint16(np.around(detected_circles)) 
    
          for pt in detected_circles[0, :]: 
              a, b, r = pt[0], pt[1], pt[2] 
    
              # Draw the circumference of the circle. 
              cv2.circle(img, (a, b), r, (0, 255, 0), 2) 
    
              # Draw a small circle (of radius 1) to show the center. 
              cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
              pts.appendleft((a, b))
              #My code starts here.
              # loop over the set of tracked points
              for i in np.arange(1, len(pts)):
                  # coordinates to display/ forward to the ai

                  edo_base_diff_cm = 7.25 #8#0; #8

                  x = round(edo_base_diff_cm + (pts[0][0] * inchesPerPixelH), 2) #from edo towards camera
                  y = round(((heightIn/2) - (pts[0][1] * inchesPerPixelH) + 4.5), 2) #from edo perpendicular to camera
                  z = round((r * inchesPerPixelH) * 2, 2) #from base of edo towards the ceiling

                  newX = round(math.sqrt((x**2) + (y**2)), 2) #euclidean distance to object
                  xy = math.degrees(math.acos(((x**2) + (newX * newX) - (y**2)) / (2 * x * newX))) #degrees rotation to object

                  #cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 3)

                  #if object is to the left of the e.Do, reverse the coordinates
                  if(y<0):
                    xy = xy * -1
                    # show the image coordinates
                  cv2.putText(img, "X: {}, Y: {}, OW: {}".format(x, y, z),
                  	(10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                  	0.35, (0, 0, 255), 1)

                  #if its less than 9 cm away from the origin, e.Do can not pick up the object,
                  # so only consider values greater than 9
                  if(x > 9):
                    f = open("newX.txt", "w")
                    f.write('%f' % newX)
                    f.close()               
                    f = open("xy.txt", "w")
                    f.write('%f' % xy)
                    f.close()

      cv2.imshow("Detected Circle", img)
      #cv2.imshow("blurred", gray_blurred)
      #cv2.imshow("gray", gray)


      k = cv2.waitKey(30) & 0xff
      if k == 27:
          cap.release()
          # close all windows
          cv2.destroyAllWindows()
