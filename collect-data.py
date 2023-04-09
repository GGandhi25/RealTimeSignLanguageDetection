import cv2
import numpy as np
import os

# Create a directory structure for storing images
if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/test")
    os.makedirs("data/train/pain")
    os.makedirs("data/train/stop")
    os.makedirs("data/train/hurts")
    #os.makedirs("data/train/iamok")
    os.makedirs("data/train/iloveyou")
    os.makedirs("data/test/pain")
    os.makedirs("data/test/stop")
    os.makedirs("data/test/hurts")
    #os.makedirs("data/test/iamok")
    os.makedirs("data/test/iloveyou")
    

# Set the mode to either "train" or "test"
mode = 'train'
directory = 'data/'+mode+'/'

# Start capturing from the default camera
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Getting count of existing images
    count = {'pain': len(os.listdir(directory+"/pain")),
             'stop': len(os.listdir(directory+"/stop")),
             'hurts': len(os.listdir(directory+"/hurts")),
             #'iamok': len(os.listdir(directory+"/iamok")),
             'iloveyou': len(os.listdir(directory+"/iloveyou"))}
    
    # Printing the image counts on the video feed
    cv2.putText(frame, "MODE : " + mode, (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "IMAGE COUNT", (10, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "PAIN : " + str(count['pain']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv2.putText(frame, "STOP : " + str(count['stop']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv2.putText(frame, "HURTS : " + str(count['hurts']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    #cv2.putText(frame, "IAMOK : " + str(count['iamok']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv2.putText(frame, "ILOVEYOU : " + str(count['iloveyou']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    # Define the coordinates of the Region of Interest (ROI)
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0),1)
    # Extract the region of interest from the frame
    roi = frame[y1:y2, x1:x2]
    #Resize the extracted ROI to 64x64 pixels
    roi = cv2.resize(roi, (64, 64)) 
 
    cv2.imshow("Sign Detection", frame)

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", roi)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory+'pain/'+str(count['pain'])+'.jpg', roi)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory+'stop/'+str(count['stop'])+'.jpg', roi)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory+'hurts/'+str(count['hurts'])+'.jpg', roi)
    # if interrupt & 0xFF == ord('3'):
    #     cv2.imwrite(directory+'iamok/'+str(count['iamok'])+'.jpg', roi)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory+'iloveyou/'+str(count['iloveyou'])+'.jpg', roi)
    
cap.release()
cv2.destroyAllWindows()

