
import numpy as np
import cv2
import cv2 as cv
from ctypes import *
import math
import random

import numpy as np
from numpy.linalg import norm
import cv2


import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
 


cap = cv.VideoCapture("Single_Subject_Static_Source_Bright_Environment.mov")



# take first frame of the video
ret,frame = cap.read()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.

out = cv2.VideoWriter('Result_Single_Subject_Static_Source_Bright_Environment.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))




 
# Create background subtractor object
background = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=180,
         detectShadows=True)
 
# Create kernel to facilitate morphological operation
    

while(True):
    
    
    # take frame-by-frame
        
    ret, frame = cap.read()
    
    # Use every frame to calculate the foreground mask as well as update
    #the corresponding background in the meanwhile
    foreground = background.apply(frame)
 
        
    # Create kernel to facilitate morphological closing operation
    
    

    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((16,16),np.uint8))
    
    
    # Remove salt and pepper noise using a median filter
    
    foreground = cv2.medianBlur(foreground, 5)
    
 
    
    
    alpha = 2.0
    beta = 0
    #increase the contrast of the image
    #foreground = cv2.convertScaleAbs(foreground, alpha=alpha, beta=beta)
         
    # Set a threshold the image to make it in binary color
    _, foreground = cv2.threshold(foreground,127,255,cv2.THRESH_BINARY)
    #Use canny edge detector to link edges
    foreground = cv2.Canny(foreground, 50, 200)
 
    # draw the bounding boxes
    contours, _= cv2.findContours(foreground,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(item) for item in contours]
 

   
    for i in range(len(areas)):
        
        cnt = contours[i]
        #check whether the bounding boxes is larger than 88
        if areas[i]>88:
            x,y,width,height = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+width,y+height),(255,255,0),2)
 
        
    cv2.imshow('frame',frame)
    out.write(frame)
       
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 

cap.release()
out.release()
cv2.destroyAllWindows()
 

