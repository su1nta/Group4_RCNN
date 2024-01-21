import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

class Human_Tracking:
    def __init__(self):
        self.deepsort = DeepSort(max_age=5, embedder="mobilenet")

    # @classmethod
    def extract_features(cls, detection):
        '''
            :param:
            frame : single frame from the video
            detection : list of dictionaries from frcnn model
        '''

        # extract the bounding boxes, labels and scores
        boxes = detection[0]["boxes"].detach().numpy()
        labels = detection[0]["labels"].detach().numpy()
        scores = detection[0]["scores"].detach().numpy()

        # filter out detections less than 50% scores
        confidence = 0.5
        mask = scores >= confidence
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        # filter non-human detections
        mask = labels == 1
        boxes = boxes[mask]
        scores = scores[mask]

        # resultant detection list for tracking
        result_detection = []
        n = len(boxes)

        for i in range(n):
            xmin, ymin, xmax, ymax = boxes[i]
            result_detection.append(([int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)], scores[i], 'person'))

        return result_detection

    
    def track_human(cls, frame, detections):

        deepsort_model = cls.deepsort

        # associate the detections with existing tracks
        tracked_objects = deepsort_model.update_tracks(detections, frame=frame)

        return tracked_objects
        

    def update_frame(self, frame, tracked_objects):
        for obj in tracked_objects:
            # if not obj.is_confirmed():
            #     continue
            bbox = obj.to_ltrb()
            track_id = obj.track_id
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                          (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def track_movement(self, frame, tracked_objects):
        for obj in tracked_objects:
            xmin, ymin, xmax, ymax = obj[:4]
            x = (xmax + xmin) / 2
            y = (ymax + ymin) / 2

            obj_velocity = np.array([x[1] - x[0], y[1] - y[0]])

            # determine if the object is still moving
            if np.linalg.norm(obj_velocity) > 0.1:
                obj_status = 1
            else:
                obj_status = 0
            
            # update the status
            obj.append(obj_status)

        return tracked_objects