from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the model
model = YOLO("bestv2.pt")

# Load image
image_path = "image.jpg"
image = cv2.imread(image_path)

# Run inference
results = model(image)[0]

# Get the class names (from your model)
class_names = model.names  # Example: {0: 'car', 1: 'truck'}

# Iterate over detections
for box in results.boxes:
    # Get coordinates and class id
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    
    # Get label name
    label = f"{class_names[class_id]} {confidence:.2f}"

    # Choose color based on class
    color = (0, 255, 0) if class_names[class_id] == 'car' else (0, 0, 255)

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Draw label
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)

# Show image with detections
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Detection Result with Labels")
plt.show()

# Save output
cv2.imwrite("output_labeled.jpg", image)