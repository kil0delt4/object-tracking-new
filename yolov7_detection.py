import torch
import cv2
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

# from utils.plots import plot_one_box


def detect(image_path, weights_path):
    # Load model
    model = attempt_load(weights_path, map_location=torch.device('cpu'))
    device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).float().div(255.0).unsqueeze(0).to(device)
    img_tensor, _ = letterbox(img_tensor, new_shape=640)

    # Perform inference
    with torch.no_grad():
        results = model(img_tensor)
        results = non_max_suppression(results, conf_thres=0.4, iou_thres=0.5)

    # Rescale bounding boxes to original image size
    img_size = img.shape[:2]
    results = results[0]
    if results is not None:
        results[:, :4] = scale_coords(img_tensor.shape[2:], results[:, :4], img_size).round()

        # Draw bounding boxes on image
        for x1, y1, x2, y2, conf, cls in results:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, classes[int(cls)], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('image', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --------------------------------------------

import torch
import cv2
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Move model to GPU if available
if torch.cuda.is_available():
    model.cuda()

# Define image preprocessing function
def preprocess(image_path):
    # Load image with opencv
    image = cv2.imread(image_path)
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize image
    image = cv2.resize(image, (640, 640))
    # Convert to PIL Image
    image = Image.fromarray(image)
    # Apply transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    # Add batch dimension
    image = image.unsqueeze(0)
    # Move to GPU if available
    if torch.cuda.is_available():
        image = image.cuda()
    return image

# Define batch inference function
def batch_inference(images):
    # Concatenate preprocessed images into a batch
    batch = torch.cat(images, dim=0)
    # Perform inference
    with torch.no_grad():
        detections = model(batch)
    # Postprocess detections
    postprocessed_detections = []
    for i in range(len(images)):
        postprocessed_detection = detections[i].cpu().numpy()
        postprocessed_detections.append(postprocessed_detection)
    return postprocessed_detections

# Define image detection function
def detect_objects(image_path):
    # Preprocess image
    image = preprocess(image_path)
    # Run batch inference
    detections = batch_inference([image])
    # Return detections
    return detections[0]

# Example usage
detections = detect_objects('image.jpg')
print(detections)