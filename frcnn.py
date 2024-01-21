import torch
import torchvision

class Faster_Rcnn:
    def __init__(self):
        # defining and evaluating model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    # @classmethod
    def human_detection(cls, frame):
        with torch.no_grad():
            rcnn_model = cls.model
            detection = rcnn_model([frame])

        # # extract boxes, labels and scores
        # boxes = detect[0]["boxes"].detach().numpy()
        # labels = detect[0]["labels"].detach().numpy()
        # scores = detect[0]["scores"].detach().numpy()

        # # filter out detections less than 80% scores
        # mask = scores >= 0.8
        # boxes = boxes[mask]
        # labels = labels[mask]
        # scores = scores[mask]

        # # filter non-human detections
        # mask = labels == 1
        # boxes = boxes[mask]
        # scores = scores[mask]

        # detection = []
        # detection.append(boxes)
        # detection.append(labels)
        # detection.append(scores)

        # print(detection)

        # # return the filtered detection
        return detection
