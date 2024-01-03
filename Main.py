import cv2

import glob

vidcap = cv2.VideoCapture('Crowd.mp4')
success,image = vidcap.read()
count = 0
while count <= 20:
      cv2.imwrite('Resampling/ExtFrame'+ "/frame%d.jpg" % count, image)   # save frame as JPEG file      
      success,image = vidcap.read()
      print('Read a new frame: ', success)
  
      count += 1
  
print ('The no. of frames',count)

dstPath='Resampling\GRFrame/'
imdir = 'Resampling\ExtFrame/'
ext = ['JPG']
files = []

# test = [glob.glob(imdir + '*.' + e) for e in ext]
# print(test)


[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
# print("FilesL: ",files[1])
images = [cv2.imread(file) for file in files]
print(images)
try:
        
        gray = cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)
      
        # cv2.imwrite(dstPath,gray)
        cv2.imwrite(dstPath + 'gray_' + file.split("/")[-1], gray)
        print ("{} is converted".format(images))
except Exception as e:
        print ("{} is not converted".format(images))
        print(str(e))
