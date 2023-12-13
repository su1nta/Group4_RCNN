import cv2

import glob

vidcap = cv2.VideoCapture('C:/Users/SOMA HAZRA/Desktop/research/samples/Crowd.mp4')
success,image = vidcap.read()
count = 0
while success:
      cv2.imwrite('C:/Users/SOMA HAZRA/Desktop/research/MyProg/Resampling/ExtFrame'+ "\\frame%d.jpg" % count, image)   # save frame as JPEG file      
      success,image = vidcap.read()
      print('Read a new frame: ', success)
  
      count += 1
  
print ('The no. of frames',count)

dstPath='C:/Users/SOMA HAZRA/Desktop/research/MyProg/Resampling/GRFrame/'
imdir = 'C:/Users/SOMA HAZRA/Desktop/research/MyProg/Resampling/ExtFrame'
ext = ['JPG']
files = []
[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
images = [cv2.imread(file) for file in files]
try:
        
        gray = cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)
      
        cv2.imwrite(dstPath,gray)
        print ("{} is converted".format(images))
except:
        print ("{} is not converted".format(images))