import torch
import torchvision
from torchvision import transforms as T
import cv2

# defining and evaluating model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


def detect_human(images):
    '''
        takes an list of images, passes to the model and returns the dictionary of predicted objects
        :param: image - a list of numpy images read by opencv (e.g. cv2.imread)
    '''
    # read the images and transform it to a tensor
    input_img = []
    transform = T.ToTensor()
    for image in images:    
        img = transform(image)
        input_img.append(img)

    # pass the image to the model
    with torch.no_grad():
        pred = model(input_img)

    return pred  
    

# class names of COCO dataset
coco_classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# test the function
img = cv2.imread("frame7.jpg")
pred = detect_human([img])

# separate bounding boxes, class_labels and prediction scores
bboxes, labels, scores = pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]

num = torch.argwhere(scores > 0.8).shape[0]
font = cv2.FONT_HERSHEY_SIMPLEX
aa = cv2.LINE_AA
for i in range(num):
    xmin, ymin, xmax, ymax = bboxes[i].numpy().astype("int")
    label_name = coco_classes[labels.numpy()[i]]
    output_text = label_name + " " + str((scores.numpy()[i]*100).astype("int"))
    img_processed = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    img_processed = cv2.putText(img_processed, output_text, (xmin, ymin - 10), font, 0.5, (255, 0, 0), 1, aa)

cv2.imshow("Prediction",img_processed)
cv2.waitKey(0)