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
boundaries = [
    ([17, 15, 100], [50, 56, 200]),
    ([86, 31, 4], [220, 88, 50]),
    ([25, 146, 190], [62, 174, 250]),
    ([103, 86, 65], [145, 133, 128])
]
pts = deque(maxlen = 32)
cap = cv2.VideoCapture(0) #number is which camera on your system you want to use
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
count = 20
zCounter = 0
zAvg = 0
total = 0
os.system('cls||clear')   #to clear the termnial screen

while True:
    print("=============================================")
    print("              Camera Terminal                ")
    print("=============================================\n")

    response = input("0: Exit\n1: Capture single frame\n2: Continuous capture\n\n")
    while((response != "0") and (response != "1") and (response != "2")):     # Input validation loop
            response = (raw_input("Please select a valid option.\n\n0: Exit\n1: Capture single frame\n2: Continuous capture\n\n"))

    if (response == "0"):
        exit(0)

    if (response == "2"):
        count = 200000000000

    while True:
        #print(count)
        if(count < 0):
          temp = raw_input("Type (y) to get object location or (n) to exit: ")
          if(temp == "y"):
            count = 20
            zCounter = 0
            zAvg = 0
            total = 0
          if(temp == "n"):
            print('\n')
            break
        while(count >= 0):
          count = count -1
          #Ask the user to continue
          #response = raw_input("To terminate put 'n': \n")
          #if(response == "n"):
            #cap.release()
            #cv2.destroyAllWindows()
            #break;
          #count = count - 1
          _, img = cap.read()
          #converting frame(img) from BGR (Blue-Green-Red) to HSV (hue-saturation-value)
          image = img 
          img = img[10:-10,10:-210]
          #defining the range of color
          green_lower = np.array([25, 52, 72],np.uint8) 
          green_upper = np.array([102, 255, 255],np.uint8)
          red_lower = np.array([0, 100, 100], np.uint8)
          red_upper = np.array([10, 255, 255], np.uint8)
          red_lower_T = np.array([160, 100, 100], np.uint8)
          red_upper_T = np.array([179, 255, 255], np.uint8)
          blue_lower = np.array([94, 80, 2], np.uint8)
          blue_upper = np.array([126, 255, 255], np.uint8)
          
          #finding the range  color in the image
          green = cv2.inRange(image, green_lower, green_upper)
          red = cv2.inRange(image, red_lower, red_upper)
          blue = cv2.inRange(image, blue_lower, blue_upper)

        
          #res=cv2.bitwise_and(img, img, mask = yellow)
          #track yellow and draw a rectangle. 
          #(_,Green_contours,Green_hierarchy)=cv2.findContours(green,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
          #(_,Red_contours,Red_hierarchy)=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
          #(_,Blue_contours,Blue_hierarchy)=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
          #if (_,Green_contours,Green_hierarchy) is not None: 
            #f = open("color.txt", "w")
            #f.write('%s' % "green")
            #f.close()       
            #print("Wrote green to file.")
          #if (_,Red_contours,Red_hierarchy) is not None: 
            #f = open("color.txt", "w")
            #f.write('%s' % "red")
            #f.close()    
            #print("Wrote red to file")
          #if (_,Blue_contours,Blue_hierarchy) is not None: 
            #f = open("color.txt", "w")
            #f.write('%s' % "blue")
            #f.close()     
            #print("Wrote blue to file.")      
          #start code here
          #Location of variable changed to fix results
          #Croping image
          #lefty, righty, leftx,rightx
          #img = img[100:-20,10:-210]
          height, width, _ = img.shape
          inchesPerPixelH = heightIn / height #'''heightIn''' 
          inchesPerPixelW = widthIn / width #'''widthIn'''
          #ends here

          ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) ,
          127, 255, cv2.THRESH_BINARY)
          contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

          
# ================== Detects Simple Squares/Rectancles   Kristina Stevoff ================= 
          for c in contours:
           x,y,w,h = cv2.boundingRect(c)
          cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

  # find minimum area
          rect = cv2.minAreaRect(c)
  # calculate coordinates of the minimum area rectangle
          box = cv2.boxPoints(rect)
          box = np.int0(box)
          cv2.drawContours(img, [box], 0, (0,0, 255), 3)
# ================================================================




          #DETECT BLUE
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

          
          #DETECT BLUE
          hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
          mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
          res_blue = cv2.bitwise_and(img, img, mask = mask_blue)
          gray = cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY)
          # Blur using 3 * 3 kernel. 
          gray_blurred = cv2.blur(gray, (3, 3))
          detected_square_blue = cv2.boundingRect(c)
          
        
          #DETECT GREEN                                        
          mask_green = cv2.inRange(hsv, green_lower, green_upper)
          res_green = cv2.bitwise_and(img, img, mask = mask_green)
          gray = cv2.cvtColor(res_green, cv2.COLOR_BGR2GRAY)
          # Blur using 3 * 3 kernel. 
          gray_blurred = cv2.blur(gray, (3, 3)) 
          # Apply Hough transform on the blurred image. 
          detected_circles_green = cv2.HoughCircles(gray_blurred,  
                            cv2.HOUGH_GRADIENT, 0.1, 20, param1 = 50, 
                        param2 = 25, minRadius = 6, maxRadius = 60)


          
          #DETECT RED
          mask_red = cv2.inRange(hsv, red_lower, red_upper)
          mask_red_T = cv2.inRange(hsv, red_lower_T, red_upper_T)
          res_red = cv2.bitwise_and(img, img, mask = mask_red | mask_red_T)
          gray = cv2.cvtColor(res_red, cv2.COLOR_BGR2GRAY)
          # Blur using 3 * 3 kernel. 
          gray_blurred = cv2.blur(gray, (3, 3)) 
          # Apply Hough transform on the blurred image. 
          detected_circles_red = cv2.HoughCircles(gray_blurred,  
                            cv2.HOUGH_GRADIENT, 0.1, 20, param1 = 50, 
                        param2 = 25, minRadius = 6, maxRadius = 60)


                        
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

             
                  
                  #My code starts here.
                  # loop over the set of tracked points
                  for i in np.arange(1, len(pts)):
                      x = round(7.25 + (pts[0][0] * inchesPerPixelH), 2)
                      y = round(((heightIn/2) - (pts[0][1] * inchesPerPixelH) + 4.5), 2)
                      z = round((r * inchesPerPixelH) * 2, 2)
                      #BEING: MATH TO CALCULATE DEPTH OF OBJECT - Hawraa Banoon 
                      #Source: https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
                      if (r > 0):
                        #4 Cm by 4 Cm object 
                        #Depth Perception calculated by: D = (W x F) / P 
                        #D = Depth, F = Focal Length, P = pixels at certain length 
                        #In this case F = (P X D) / W , Where P = 16 pixels when on table, 
                        #D is depth of table (107 cm) and W is 4cm object 
                        newZ = round((4 * 856) / (r * 2), 2) 
                        newZ = 107 - newZ
                      else:
                        newZ = 6
                      zCounter = zCounter +1
                      total = total + newZ
                      zAvg = total /zCounter
                      print (zAvg)
                        #print(newZ) 
                      #END: MATH TO CALCULATE DEPTH OF OBJECT - Hawraa Banoon 
                      newX = round(math.sqrt((x*x) + (y*y) +(zAvg*zAvg)), 2)
                      xy =math.degrees(math.atan(y/x))
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
                #My code ends here 
          cv2.imshow("Detected Circle", img)
          #cv2.imshow("blurred", gray_blurred)
          #cv2.imshow("gray", gray)
          cv2.imshow('Filter_blue', res_blue)
          cv2.imshow('Filter_green', res_green)
          cv2.imshow('Filter_red', res_red)
          k = cv2.waitKey(30) & 0xff
          if k == 27:

            cap.release()


         # close all windows
            cv2.destroyAllWindows()
            exit(0)
