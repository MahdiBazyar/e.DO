# ================================== Camera File! ==================================
#  Developers: Wayne State University Senior Capstone Project Students
#              Fall 2019: Adel Mohamed, Corey Simms, Mahmoud Elmasri, & Tyler Riojas 
#              Winter 2020: Hawraa Banoon, Nathaniel Smith, Kristina Stevoff, & Mahdi Bazyar
#  Directory:  /home/nvidia/jetson-reinforcement/build/aarch64
#  Interface:  Runs the camera terminal 
#  Purpose:    This file runs the camera and allows the user to choose capturing a
#              single shot of n frames or run the camera continuously. The images 
#              will then be processed to output the color and location of the object.
#  Inputs:     Camera connection
#  Outputs:    16 Text Files - 4 Text Files Per Quadrant 
#              Quadrant 1: 
#              Color1.txt (The color of the object- red, blue, or green) 
#              newX1.txt  (The distance of object from base of e.DO in cm) 
#              xy1.txt    (The angle that the object is in relation to e.DO in degrees) 
#              newZ1.txt  (The depth of the object in cm)
#              Quadrant 2: 
#              color2.txt (The color of the object- red, blue, or green) 
#              newX2.txt  (The distance of object from base of e.DO in cm) 
#              xy2.txt    (The angle that the object is in relation to e.DO in degrees) 
#              newZ2.txt  (The depth of the object in cm)
#              Quadrant 3: 
#              color3.txt (The color of the object- red, blue, or green) 
#              newX3.txt  (The distance of object from base of e.DO in cm) 
#              xy3.txt    (The angle that the object is in relation to e.DO in degrees) 
#              newZ3.txt  (The depth of the object in cm)
#              Quadrant 4: 
#              color4.txt (The color of the object- red, blue, or green) 
#              newX4.txt  (The distance of object from base of e.DO in cm) 
#              xy4.txt    (The angle that the object is in relation to e.DO in degrees) 
#              newZ4.txt  (The depth of the object in cm)
# ================================== Camera File! ==================================


# =========== Standard Imports and Necessary Packages ===========
from collections import deque # Import deque module provides you with a double ended queue
from imutils.video import VideoStream # Capture a live video stream 
import numpy as np # N-dimensional array-processing package
import cv2 # OpenCV libaray for image processing 
import imutils # Series of convenience functions to make OpenCV functions easier 
import time # Time library 
import math # Math library 
import sys # System library 
import os # Operating Systems Interface library 
import cam_config as cfg # External configuration file
# =========== Standard Imports and Necessary Packages ===========

# Create double ended queue 
pts = deque(maxlen = 32)

# Open the camera to capture a video stream 
# If you are using a personal laptop, you may need to change the number from 1 to 0 or -1. 
cap = cv2.VideoCapture(cfg.capture_source) 

# Define the height and width
cap.set(3, cfg.capture_width) 
cap.set(4, cfg.capture_height)

# Allow the camera or video file to warm up
time.sleep(cfg.warmup_time)

# =========== Math logic to find the height and length ===========
depthOrAdjacent = cfg.surface_to_cam_cm

verticalFieldOfView = cfg.cam_v_fov_cm
verticalFieldOfViewHalf = verticalFieldOfView / 2
baseOrOppositeVFOV = math.tan(math.radians(verticalFieldOfViewHalf)) * depthOrAdjacent


horizontalFieldOfView = cfg.cam_h_fov_cm
horizontalFieldOfViewHalf = horizontalFieldOfView / 2
baseOrOppositeHFOV = math.tan(math.radians(horizontalFieldOfViewHalf)) * depthOrAdjacent

  
heightIn = baseOrOppositeVFOV * 2
widthIn = baseOrOppositeHFOV * 2
print(heightIn, widthIn) # Debug Statement

count = cfg.number_of_frames    # Counter for frames 
zCounter = 0                    # Counter for calcuating depth average 
zAvg = 0                        # Average depth over a the set of frames
total = 0                       # Total of the depth values of each frame

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
        # If response is 1, capture n frames
        if(count < 0 and response == "1"):
          temp = raw_input("Type (y) to get object location or (n) to exit: ")
          
          if(temp == "y"):
            count = cfg.number_of_frames    # Counter for frames 
            zCounter = 0                    # Counter for calcuating depth average 
            zAvg = 0                        # Average depth over a the set of frames
            total = 0                       # Total of the depth values of each frame
            
          if(temp == "n"):
            print('\n')
            break
            
        while(count >= 0 or continuousCamera):
              
          count = count - 1
          # Capture frame-by-frame
          _, img = cap.read()
          # Crop image to a desirable size (Captures table where object would be placed) 
          img = img[cfg.crop_row_start:cfg.crop_row_end,cfg.crop_col_start:cfg.crop_col_end]
         #A copy of the image 
          frame = img
          
          # ================== TOP RIGHT CORNER =================
          contours = np.array ([[0, 0], [0,300], [300,300],[300, 0]])
          mask = np.zeros(img.shape, dtype=np.uint8)
          cv2.fillPoly(mask, pts=[contours], color=(255,255,255))
          # apply the mask
          masked_image1 = cv2.bitwise_and(img, mask)
          cv2.imshow("corner1", masked_image1) 


          # ================== TOP LEFT CORNER =================
          contours = np.array ([[300, 0], [300,300], [600,300],[600, 0]])
          mask2 = np.zeros(img.shape, dtype=np.uint8)
          cv2.fillPoly(mask2, pts=[contours], color=(255,255,255))
          # apply the mask
          masked_image2 = cv2.bitwise_and(img, mask2)
          cv2.imshow("corner2", masked_image2) 


          # ================== BOTTOM RIGHT CORNER =================
          contours = np.array ([[0, 300], [0,600], [300,600],[300, 300]])
          mask3 = np.zeros(img.shape, dtype=np.uint8)
          cv2.fillPoly(mask3, pts=[contours], color=(255,255,255))
          # apply the mask
          masked_image3 = cv2.bitwise_and(img, mask3)
          cv2.imshow("corner3", masked_image3) 


          # ================== BOTTOM LEFT CORNER =================
          contours = np.array ([[300, 300], [300,600], [600,600],[600, 300]])
          mask4 = np.zeros(img.shape, dtype=np.uint8)
          cv2.fillPoly(mask4, pts=[contours], color=(255,255,255))
          # apply the mask
          masked_image4 = cv2.bitwise_and(img, mask4)
          cv2.imshow("corner4", masked_image4)
          
          img_counter = 0
          masked_images = [masked_image1, masked_image2, masked_image3, masked_image4]
          for i  in masked_images: 
			  img_counter+= 1
			  img = i 
			  # Defining the range of colors in HSV values for the filters 
			  green_lower = np.array([cfg.green_lower_hue, cfg.green_lower_saturation, cfg.green_lower_value],np.uint8) 
			  green_upper = np.array([cfg.green_upper_hue, cfg.green_upper_saturation, cfg.green_upper_value],np.uint8)
			  red_lower = np.array([cfg.red_lower_hue, cfg.red_lower_saturation, cfg.red_lower_value], np.uint8)
			  red_upper = np.array([cfg.red_upper_hue, cfg.red_upper_saturation, cfg.red_upper_value], np.uint8)
			  red_lower_T = np.array([cfg.red_lower_hue_2, cfg.red_lower_saturation_2, cfg.red_lower_value_2], np.uint8)
			  red_upper_T = np.array([cfg.red_upper_hue_2, cfg.red_upper_saturation_2, cfg.red_upper_value_2], np.uint8)
			  blue_lower = np.array([cfg.blue_lower_hue, cfg.blue_lower_saturation, cfg.blue_lower_value], np.uint8)
			  blue_upper = np.array([cfg.blue_upper_hue, cfg.blue_upper_saturation, cfg.blue_upper_value], np.uint8)
			  
			  height, width, _ = img.shape
			  inchesPerPixelH = heightIn / height #'''heightIn''' 
			  inchesPerPixelW = widthIn / width #'''widthIn'''
			  
			  # ================== DETECT BLUE CIRCLE =================
			  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
			  mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
			  res_blue = cv2.bitwise_and(img, img, mask = mask_blue)
			  gray = cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY)
			  # Blur using n * n kernel. 
			  gray_blurred = cv2.blur(gray, (cfg.blur_kernel_size, cfg.blur_kernel_size)) 
			  # Apply Hough transform on the blurred image. 
			  detected_circles_blue = cv2.HoughCircles(gray_blurred,  
								cv2.HOUGH_GRADIENT, cfg.inverse_resolution_ratio, cfg.center_min_distance, 
								param1 = cfg.edge_upper_threshold, param2 = cfg.center_detect_threshold, 
								minRadius = cfg.min_radius, maxRadius = cfg.max_radius)
					 
			  # ================== DETECT GREEN CIRCLE =================
			  mask_green = cv2.inRange(hsv, green_lower, green_upper)
			  res_green = cv2.bitwise_and(img, img, mask = mask_green)
			  gray = cv2.cvtColor(res_green, cv2.COLOR_BGR2GRAY)
			  # Blur using n * n kernel. 
			  gray_blurred = cv2.blur(gray, (cfg.blur_kernel_size, cfg.blur_kernel_size)) 
			  # Apply Hough transform on the blurred image. 
			  detected_circles_green = cv2.HoughCircles(gray_blurred,  
								cv2.HOUGH_GRADIENT, cfg.inverse_resolution_ratio, cfg.center_min_distance, 
								param1 = cfg.edge_upper_threshold, param2 = cfg.center_detect_threshold, 
								minRadius = cfg.min_radius, maxRadius = cfg.max_radius)
			  
			  # ================== DETECT RED CIRCLE =================
			  mask_red = cv2.inRange(hsv, red_lower, red_upper)
			  mask_red_T = cv2.inRange(hsv, red_lower_T, red_upper_T)
			  res_red = cv2.bitwise_and(img, img, mask = mask_red | mask_red_T)
			  gray = cv2.cvtColor(res_red, cv2.COLOR_BGR2GRAY)
			  # Blur using n * n kernel. 
			  gray_blurred = cv2.blur(gray, (cfg.blur_kernel_size, cfg.blur_kernel_size)) 
			  # Apply Hough transform on the blurred image. 
			  detected_circles_red = cv2.HoughCircles(gray_blurred,  
								cv2.HOUGH_GRADIENT, cfg.inverse_resolution_ratio, cfg.center_min_distance, 
								param1 = cfg.edge_upper_threshold, param2 = cfg.center_detect_threshold, 
								minRadius = cfg.min_radius, maxRadius = cfg.max_radius)
			  
			  colorTextFile = "color" + str(img_counter) + ".txt" 
			  newXTextFile = "newX" + str(img_counter) + ".txt" 
			  xyTextFile = "xy" + str(img_counter) + ".txt" 
			  newZTextFile = "newZ" + str(img_counter) + ".txt" 
			  # WRITE COLOR DETECTED TO TEXT FILE
			  if detected_circles_green is not None: 
				f = open(colorTextFile, "w")
				f.write('%s' % "green")
				f.close()       
				#print("Wrote green to file.")
			  if detected_circles_blue is not None: 
				f = open(colorTextFile, "w")
				f.write('%s' % "blue")
				f.close()     
				#print("Wrote blue to file.")      
			  if detected_circles_red is not None: 
				f = open(colorTextFile, "w")
				f.write('%s' % "red")
				f.close()     
				#print("Wrote red to file.")      
			  if ((detected_circles_blue is None) and (detected_circles_green is None) and (detected_circles_red is None)): 
				f = open(colorTextFile, "r+")
				f.truncate(0)
				f.close()  
            
			# If any color circles detected then must perform calculations for coordinates: 
			  if ((detected_circles_blue is not None) or (detected_circles_green is not None) or (detected_circles_red is not None)): 
				  if detected_circles_blue is not None: 
					  detected_circles = np.uint16(np.around(detected_circles_blue))
				  if detected_circles_green is not None:
					  detected_circles = np.uint16(np.around(detected_circles_green)) 
				  if detected_circles_red is not None:
					  detected_circles = np.uint16(np.around(detected_circles_red)) 
				  
				  for pt in detected_circles[0, :]:
					  # Define two points and radius of circle 
					  a, b, r = pt[0], pt[1], pt[2] 
					  
					  # Draw a green circle around the circle detected 
					  cv2.circle(frame, (a, b), r, (cfg.detected_circle_b, cfg.detected_circle_g, cfg.detected_circle_r), 2) 

					  # Draw a small red circle (of radius 1) to show the center
					  cv2.circle(frame, (a, b), 1, (cfg.detected_center_b, cfg.detected_center_g, cfg.detected_center_r), 3)
					  pts.appendleft((a, b))
					  
					  # Loop over the set of tracked points
					  for i in np.arange(1, len(pts)):
						  x = round(cfg.distance_buffer + (pts[0][0] * inchesPerPixelH), 2)
						  y = round(((heightIn / 2) - (pts[0][1] * inchesPerPixelH) + cfg.angle_buffer), 2)
						  z = round((r * inchesPerPixelH) * 2, 2)
						  # BEGIN: MATH TO CALCULATE DEPTH OF OBJECT
						  # Source: https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
						  if (r > 0):
							# n x n cm object 
							# Depth Perception calculated by: D = (W x F) / P 
							# D = Depth, F = Focal Length, P = pixels at certain length 
							# In this case F = (P X D) / W , Where P = 16 pixels when on table, 
							# W is 4 cm object, F = 107 cm from table to camera
							newZ = round((cfg.object_size_cm * 856) / (r * 2), 2) 
							newZ = cfg.surface_to_cam_cm - newZ
						  else:
							newZ = cfg.default_surface_depth
							
						  # Calculate the average depth for the frames captured 
						  zCounter = zCounter + 1
						  total = total + newZ
						  zAvg = total / zCounter
						  #print zAvg
						  #print(newZ) 
						  
						  # END: MATH TO CALCULATE DEPTH OF OBJECT 
						  
						  # Convert from cartesian coordinates to 3D Polar Coordinates 
						  newX = round(math.sqrt((x * x) + (y * y) +(zAvg * zAvg)), 2)
						  xy = math.degrees(math.atan(y / x))
						  
						  # PRIOR calculation from cartesian cordiantes to 2D Polar Coordinates 
						  #xy = xy -45.0
						  #xy = math.degrees(math.acos(((x*x) + (newX * newX) - (y*y)) / (2 * x * newX)))
						  
						   
						  #cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 3)

						  # Print the image coordinates on the bottom right corner of image. 
						  cv2.putText(img, "X: {}, Y: {}, OW: {}".format(x, y, z),
							(10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
							0.35, (0, 0, 255), 1)
						  # Write values to text files to be read by ML model 
						  if(x > cfg.min_object_distance):
							f = open(newXTextFile, "w")
							f.write('%f' % newX)
							f.close()               
							f = open(xyTextFile, "w")
							f.write('%f' % xy)
							f.close()
						  if(newZ > cfg.default_surface_depth):
							zAvg = zAvg + 2
							f = open(newZTextFile, "w")
							f.write('%f' % zAvg)
							f.close()
						  else: 
							zAvg = cfg.default_surface_depth
							f = open(newZTextFile, "w")
							f.write('%f' % zAvg)
							f.close()
				  cv2.imshow("Image", frame)
				  #cv2.imshow("Detected Circle", img)     # Open new window to show video stream with circle detection
				  #cv2.imshow('Blue Filter', res_blue)    # Open new window to show video stream that detects blue 
				  #cv2.imshow('Green Filter', res_green)  # Open new window to show video stream that detects green 
				  #cv2.imshow('Red Filter', res_red)      # Open new window to show video stream that detects red
				  cv2.imshow('Processing', img) 
			  else:
				  fx = open(newXTextFile, "r+")
				  fx.truncate(0)
				  fx.close()               
				  fxy = open(xyTextFile, "r+")
				  fxy.truncate(0)
				  fxy.close()
				  fz = open(newZTextFile, "r+")
				  fz.truncate(0)
				  fz.close()
          k = cv2.waitKey(30) & 0xff
          if k == 27:
            cap.release()
            cv2.destroyAllWindows()     # Close all windows
            exit(0)                     # Exit the program
