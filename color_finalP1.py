# ================================== Camera File! ==================================
#  Developers: Wayne State University Senior Capstone Project Students
#              Fall 2019: Adel Mohamed, Corey Simms, Mahmoud Elmasri, & Tyler Riojas 
#              Winter 2020: Hawraa Banoon, Nathaniel Smith, Kristina Stevoff, & Mahdi Bazyar
#  Directory:  /home/nvidia/jetson-reinforcement/build/aarch64
#  Interface:  Runs the camera terminal 
#  Purpose:    This file runs the camera and allows the user to choose capturing a
#              single shot of 20 frames or run the camera continuously. The images 
#              will then be processed to output the color and location of the object.
#  Inputs:     Camera connection
#  Outputs:    4 Text Files
#              Color.txt (The color of the object- red, blue, or green) 
#              newX.txt  (The distance of object from base of e.DO in cm) 
#              xy.txt    (The angle that the object is in relation to e.DO in degrees) 
#              newZ.txt  (The depth of the object in cm) 
# ================================== Camera File! ==================================


# =========== Standard Imports and Necessary Packages ===========
from collections import deque # import deque module provides you with a double ended queue
from imutils.video import VideoStream # Capture a live video stream 
import numpy as np # N-Demensional array-processing package
import cv2 # OpenCV libaray for image processing 
import imutils # Series of convenience functions to make OpenCV functions easier 
import time # Time library 
import math # Math library 
import sys # System library 
import os # Operating Systems Interface library 
# =========== Standard Imports and Necessary Packages ===========

# Create double ended queue 
pts = deque(maxlen = 32)

# Open the camera to capture a video stream 
# If you are using a personal laptop, you may need to change the number from 1 to 0 or -1. 
cap = cv2.VideoCapture(1) 

# Define the height and width
cap.set(3, 800) 
cap.set(4, 600)

# Allow the camera or video file to warm up
time.sleep(2.0)

# =========== Math logic to find the height and length ===========
depthOrAdjacent = 107

verticalFieldOfView = 43.3
verticalFieldOfViewHalf = verticalFieldOfView / 2
baseOrOppositeVFOV = math.tan(math.radians(verticalFieldOfViewHalf)) * depthOrAdjacent


horizontalFieldOfView = 70.42
horizontalFieldOfViewHalf = horizontalFieldOfView / 2
baseOrOppositeHFOV = math.tan(math.radians(horizontalFieldOfViewHalf)) * depthOrAdjacent

  
heightIn = baseOrOppositeVFOV * 2
widthIn = baseOrOppositeHFOV * 2
print(heightIn, widthIn) # Debug Statement

count = 20    # Counter for frames 
zCounter = 0  # Counter for calcuating depth average 
zAvg = 0      # Average depth over a the set of frames
total = 0	  # Total of the depth values of each frame

continuousCamera = False

os.system('cls||clear')   # Clear the terminal screen 

# Run the terminal 
while True:
    print("=============================================")
    print("              Camera Terminal                ")
    print("=============================================\n")
    
    # Menu options (prompt user input) 
    response = raw_input("0: Exit\n1: Capture still image\n2: Capture video\n\n")
    
    # Input validation loop
    while((response != "0") and (response != "1") and (response != "2")):     
            response = (raw_input("Please select a valid option.\n\n0: Exit\n1: Capture still image\n2: Capture video\n\n"))
    
    # If response is 0, exit the program 
    if (response == "0"):
        print "< Exiting... >\n"
        time.sleep(0.25)
        exit(0)
    
    # If response is 2, continuously run the camera
    if (response == "2"):
        continuousCamera = True
    
    while True:
        #print(count)
        # If response is 1, capture 20 frames
        if(count < 0 and response == "1"):
          temp = raw_input("Type (y) to get object location or (n) to exit: ")
          
          if(temp == "y"):
            count = 20
            zCounter = 0
            zAvg = 0
            total = 0
            
          if(temp == "n"):
            print('\n')
            break
            
        while(count >= 0 or continuousCamera):
              
          count = count - 1
          # Capture frame-by-frame
          _, img = cap.read()
          # Crop image to a desirable size (Captures table where object would be placed) 
          img = img[10:-10,10:-210]
          
          # Defining the range of colors in HSV values for the filters 
          green_lower = np.array([25, 52, 72],np.uint8) 
          green_upper = np.array([102, 255, 255],np.uint8)
          red_lower = np.array([0, 100, 100], np.uint8)
          red_upper = np.array([10, 255, 255], np.uint8)
          red_lower_T = np.array([160, 100, 100], np.uint8)
          red_upper_T = np.array([179, 255, 255], np.uint8)
          blue_lower = np.array([94, 80, 2], np.uint8)
          blue_upper = np.array([126, 255, 255], np.uint8)
          
          height, width, _ = img.shape
          inchesPerPixelH = heightIn / height #'''heightIn''' 
          inchesPerPixelW = widthIn / width #'''widthIn'''
          
          # DETECT BLUE
          hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
          mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
          res_blue = cv2.bitwise_and(img, img, mask = mask_blue)
          gray = cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY)
          # Blur using 3 * 3 kernel. 
          gray_blurred = cv2.blur(gray, (3, 3)) 
          # Apply Hough transform on the blurred image. 
          detected_circles_blue = cv2.HoughCircles(gray_blurred,  
                            cv2.HOUGH_GRADIENT, 0.1, 20, param1 = 40, 
                        param2 = 23, minRadius = 0, maxRadius = 60) 
          
        
          # DETECT GREEN
          mask_green = cv2.inRange(hsv, green_lower, green_upper)
          res_green = cv2.bitwise_and(img, img, mask = mask_green)
          gray = cv2.cvtColor(res_green, cv2.COLOR_BGR2GRAY)
          # Blur using 3 * 3 kernel. 
          gray_blurred = cv2.blur(gray, (3, 3)) 
          # Apply Hough transform on the blurred image. 
          detected_circles_green = cv2.HoughCircles(gray_blurred,  
                            cv2.HOUGH_GRADIENT, 0.1, 20, param1 = 40, 
                        param2 = 23, minRadius = 6, maxRadius = 60) 
          
          # DETECT RED
          mask_red = cv2.inRange(hsv, red_lower, red_upper)
          mask_red_T = cv2.inRange(hsv, red_lower_T, red_upper_T)
          res_red = cv2.bitwise_and(img, img, mask = mask_red | mask_red_T)
          gray = cv2.cvtColor(res_red, cv2.COLOR_BGR2GRAY)
          # Blur using 3 * 3 kernel. 
          gray_blurred = cv2.blur(gray, (3, 3)) 
          # Apply Hough transform on the blurred image. 
          detected_circles_red = cv2.HoughCircles(gray_blurred,  
                            cv2.HOUGH_GRADIENT, 0.1, 20, param1 = 40, 
                        param2 = 23, minRadius = 6, maxRadius = 60) 
                        
          if detected_circles_green is not None: 
            f = open("color.txt", "w")
            f.write('%s' % "green")
            f.close()       
            print("Wrote green to file.")
          if detected_circles_blue is not None: 
            f = open("color.txt", "w")
            f.write('%s' % "blue")
            f.close()     
            print("Wrote blue to file.")      
          if detected_circles_red is not None: 
            f = open("color.txt", "w")
            f.write('%s' % "red")
            f.close()     
            print("Wrote red to file.")      
            
        # Draw circles that are detected. 
          if ((detected_circles_blue is not None)or(detected_circles_green is not None)or(detected_circles_red is not None)): 
              if detected_circles_blue is not None: 
                  detected_circles = np.uint16(np.around(detected_circles_blue))
              if detected_circles_green is not None:
                  detected_circles = np.uint16(np.around(detected_circles_green)) 
              if detected_circles_red is not None:
                  detected_circles = np.uint16(np.around(detected_circles_red)) 
              
              for pt in detected_circles[0, :]: 
                  a, b, r = pt[0], pt[1], pt[2] 
                  cv2.circle(img, (a, b), r, (0, 255, 0), 2) 
                  print(r)
                  # Draw a small circle (of radius 1) to show the center. 
                  cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
                  pts.appendleft((a, b))
                  
                  # loop over the set of tracked points
                  for i in np.arange(1, len(pts)):
                      x = round(7.25 + (pts[0][0] * inchesPerPixelH), 2)
                      y = round(((heightIn/2) - (pts[0][1] * inchesPerPixelH) + 4.5), 2)
                      z = round((r * inchesPerPixelH) * 2, 2)
                      # BEGIN: MATH TO CALCULATE DEPTH OF OBJECT - Hawraa Banoon 
                      # Source: https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
                      if (r > 0):
                        # 4 Cm by 4 Cm object 
                        # Depth Perception calculated by: D = (W x F) / P 
                        # D = Depth, F = Focal Length, P = pixels at certain length 
                        # In this case F = (P X D) / W , Where P = 16 pixels when on table, 
                        # D is depth of table (107 cm) and W is 4cm object 
                        newZ = round((4 * 856) / (r * 2), 2) 
                        newZ = 107 - newZ
                      else:
                        newZ = 6
                        
                      zCounter = zCounter + 1
                      total = total + newZ
                      zAvg = total / zCounter
                      print zAvg
                      #print(newZ) 
                      
                      # END: MATH TO CALCULATE DEPTH OF OBJECT - Hawraa Banoon 
                      
                      newX = round(math.sqrt((x*x) + (y*y) +(zAvg*zAvg)), 2)
                      xy = math.degrees(math.atan(y/x))
                      #xy = xy -45.0
                      #xy = math.degrees(math.acos(((x*x) + (newX * newX) - (y*y)) / (2 * x * newX)))
                      
                       
                      #cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 3)

                      # show the image coordinates
                      #if(y<0):
                        #xy = xy * -1 
                      cv2.putText(img, "X: {}, Y: {}, OW: {}".format(x, y, z),
                        (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 255), 1)
                      
                      if(x > 9):
                        f = open("newX.txt", "w")
                        f.write('%f' % newX)
                        f.close()               
                        f = open("xy.txt", "w")
                        f.write('%f' % xy)
                        f.close()
                      if(newZ > 5):
                        zAvg = zAvg + 2
                        f = open("newZ.txt", "w")
                        f.write('%f' % zAvg)
                        f.close()
                      else: 
                        zAvg = 5.0
                        f = open("newZ.txt", "w")
                        f.write('%f' % zAvg)
                        f.close()
                
          cv2.imshow("Detected Circle", img)     # Open new window to show video stream with circle detection 
          #cv2.imshow('Filter_blue', res_blue)    # Open new window to show video stream that detects blue 
          #cv2.imshow('Filter_green', res_green)  # Open new window to show video stream that detects green 
          #cv2.imshow('Filter_red', res_red)      # Open new window to show video stream that detects red
          
          k = cv2.waitKey(30) & 0xff
          if k == 27:
            cap.release()
            cv2.destroyAllWindows()     # Close all windows
            exit(0)                     # Exit the program
