import cv2
import frcnn as fr
import human_tracking as ht
from torchvision import transforms as T

tensor_transform = T.ToTensor()
frcnn = fr.Faster_Rcnn()
human_track = ht.Human_Tracking()

vidcap = cv2.VideoCapture('Crowd.mp4')


success,frame = vidcap.read()
frame_rs = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
frame_ten = tensor_transform(frame_rs)

detection = frcnn.human_detection(frame_ten)
reduced_detection = human_track.extract_features(detection)
tracked_objects = human_track.track_human(frame_rs, reduced_detection)
updated_frame = human_track.update_frame(frame_rs, tracked_objects)

cv2.imshow('tracking', updated_frame)
cv2.waitKey(0)

vidcap.release()
cv2.destroyAllWindows()