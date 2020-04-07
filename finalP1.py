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
#Image Resolution 
image_width = 800
image_height = 600
cap.set(3, image_width) #3 = Width
cap.set(4, image_height) # 4 = Height 

#Allow the camera or video file to warm up
time.sleep(2.0)

#Math logic to find the height and length
depthOfCamera= 107 #Depth from camera to table in CM 

verticalFieldOfView = 43.3
verticalFieldOfViewHalf = verticalFieldOfView / 2
baseOrOppositeVFOV = math.tan(math.radians(verticalFieldOfViewHalf)) * depthOfCamera

horizontalFieldOfView = 70.42
horizontalFieldOfViewHalf = horizontalFieldOfView / 2
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
      ret, img = cap.read() #Takes one picture frame and stores in numpy array img
      #start code here
      #Location of variable changed to fix results
      #Croping image
      #lefty, righty, leftx,rightx
      #img = img[100:-20,10:-210]
      img = img[10:-10,10:-210] 

      height, width, _ = img.shape # returns a tuple of the number of rows, columns, and channels (if the image is color)
      inchesPerPixelH = heightIn / height #'''heightIn''' 
      inchesPerPixelW = widthIn / width	#'''widthIn'''
      #ends here
      # Convert to grayscale. 
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
      # Blur using 3 * 3 kernel. 
      gray_blurred = cv2.blur(gray, (3, 3)) 
    
      # Apply Hough transform on the blurred image. 
      detected_circles = cv2.HoughCircles(gray_blurred,  
                     	cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                 	param2 = 30, minRadius = 1, maxRadius = 0) 
    
    # Draw circles that are detected. 
      if detected_circles is not None: 
    
      # Convert the circle parameters a, b and r to integers. 
          detected_circles = np.uint16(np.around(detected_circles)) #Rounds to zero decimals
    
          for pt in detected_circles[0, :]: 
              #a & b = center_coordinates r = radius 
              a, b, r = pt[0], pt[1], pt[2] 
    
              # Draw the circumference of the circle.
              # DEFINITION: cv2.circle(image, center_coordinates, radius, color, thickness)
              cv2.circle(img, (a, b), r, (203, 194, 244), 2) 
    
              # Draw a small circle (of radius 1) to show the center. 
              cv2.circle(img, (a, b), 1, (235, 244, 194), 3)

              pts.appendleft((a, b))
              #My code starts here.
              # loop over the set of tracked points
              for i in np.arange(1, len(pts)):
                  # coordinates to display/ forward to the ai
                  diff = 7.25#8#0; #8
                  x = round(diff + (pts[0][0] * inchesPerPixelH), 2)
                  y = round(((heightIn/2) - (pts[0][1] * inchesPerPixelH) + 4.5), 2)
                  z = round((r * inchesPerPixelH) * 2, 2)
                  newZ = z 
                  newX = round(math.sqrt((x*x) + (y*y)), 2)
                  xy = math.degrees(math.acos(((x*x) + (newX * newX) - (y*y)) / (2 * x * newX)))
                  #cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 3)

                  # show the image coordinates
                  if(y<0):
                    xy = xy * -1 
                  cv2.putText(img, "X: {}, Y: {}, OW: {}".format(x, y, z),
                  	(10, img.shape[0] - 10), cv2.FONT_HERSHEY_S 32IMPLEX,
                  	0.35, (0, 0, 255), 1)
                  
                  if(x > 9):
                    f = open("newX.txt", "w")
                    f.write('%f' % newX)
                    f.close()               
                    f = open("xy.txt", "w")
                    f.write('%f' % xy)
                    f.close()
                    f = open("newZ.txt", "w")
                    f.write('%f' % newZ)
                    f.close()
      #DEFINITION: cv2.imshow(window_name, image)
      cv2.imshow("Detected Circle", img)
      #cv2.imshow("blurred", gray_blurred)
      #cv2.imshow("gray", gray)
      key = cv2.waitKey(30) & 0xff 
      if key == 27:
           cap.release()
     # close all windows
           cv2.destroyAllWindows()
