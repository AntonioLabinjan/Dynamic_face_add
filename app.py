import os
import cv2
import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime

app = Flask(__name__)

# Paths to model weights and dataset
weights_path = "C:/Users/Korisnik/Desktop/Live_face_add/fine_tuned_classifier.pth"
dataset_path = "C:/Users/Korisnik/Desktop/Live_face_add/known_faces"

# Load the CLIP model, processor, and fine-tuned classifier
def load_clip_model(weights_path, dataset_path):
    """
    Loads the CLIP model and processor, initializes the classifier,
    and loads weights if the dataset contains known classes.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Determine number of classes from the dataset
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    num_classes = len(class_dirs)

    # Initialize the classifier based on the number of classes
    classifier = torch.nn.Linear(model.config.projection_dim, num_classes if num_classes > 0 else 1)
    
    # Load classifier weights only if they match the current classifier dimensions
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path)
        
        # Check if the saved weights match the classifier's shape
        if state_dict['weight'].shape == classifier.weight.shape and state_dict['bias'].shape == classifier.bias.shape:
            classifier.load_state_dict(state_dict)
            print("Loaded classifier weights successfully.")
        else:
            print("Warning: Saved weights do not match the current classifier dimensions. Initializing with new weights.")
    else:
        print("No weights found. Initializing classifier with random weights.")
    
    model.eval()
    classifier.eval()
    return model, processor, classifier


# Load the model, processor, and classifier
model, processor, classifier = load_clip_model(weights_path, dataset_path)
known_face_encodings = []
known_face_names = []

# Ensure the 'known_faces' folder exists for storing student images
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Function to add a face encoding from a live feed frame
def add_known_face_from_frame(image_frame, name):
    global known_face_encodings, known_face_names

    # Convert frame to RGB and process it
    image_rgb = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt")

    # Generate the image feature embedding
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    
    # Normalize and add to encodings
    embedding = outputs.cpu().numpy().flatten()
    known_face_encodings.append(embedding / np.linalg.norm(embedding))
    known_face_names.append(name)
    print(f"Added face for {name} from live capture")

# Route to capture live feed images and add a new student
@app.route('/add_student_live', methods=['GET', 'POST'])
def add_student_live():
    if request.method == 'POST':
        name = request.form['name']
        student_dir = os.path.join(dataset_path, name)
        os.makedirs(student_dir, exist_ok=True)
        
        # Start webcam capture
        cap = cv2.VideoCapture(0)
        frame_count = 0
        required_frames = 25  # Number of frames to capture

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Display live video feed
            cv2.imshow("Capture Face - Press 'q' to Quit", frame)

            # Capture every frame
            if frame_count < required_frames:
                # Save frame in student's directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                frame_path = os.path.join(student_dir, f"{name}_{timestamp}_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                
                # Process and save the embedding
                add_known_face_from_frame(frame, name)
                frame_count += 1
            
            # Stop capture if 'q' is pressed or required frames reached
            if cv2.waitKey(1) & 0xFF == ord('q') or frame_count >= required_frames:
                break

        cap.release()
        cv2.destroyAllWindows()

        return redirect(url_for('add_student_success', name=name))
    
    return render_template('add_student_live.html')

# Confirmation route after adding a student
@app.route('/add_student_success')
def add_student_success():
    name = request.args.get('name')
    return render_template('add_student_success.html', name=name)


@app.route('/')
def home():
    return redirect(url_for('hello'))


@app.route('/hello')
def hello():
    return redirect(url_for('add_student_live'))
# Run the app
if __name__ == '__main__':
    app.run(debug=True)
