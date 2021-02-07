import cv2
import os
 
# The video path to be extracted
video_path = 'source.mp4'
 
times = 0
name = 1   # Number of the first picture

 
# Frequency of extracting video, extract one every x frames
frameFrequency = 30
 
# Output pictures to the current directory vedio folder
outPutDirName=os.getcwd()+ '/image'
 
if not os.path.exists(outPutDirName):
    # If the file directory does not exist, create a directory
    os.makedirs(outPutDirName)
 
camera = cv2.VideoCapture('source.mp4')
 
while True:
    times+=1
    res, image = camera.read()
    if not res:
        print('not res , not image')
        break
    if times%frameFrequency == 0:
        cv2.imwrite(outPutDirName + 'img_'+ str(name)+ '.jpg', image)
        print(outPutDirName + 'img_' + str(name)+'.jpg')
        name += 1
 
print('Picture extraction finished')
camera.release()
