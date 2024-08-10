from credentials import my_api_key,my_endpoint_id
import os
from landingai.pipeline.image_source import Webcam
from landingai.predict import Predictor

# Initialize the Predictor with your endpoint ID and API key
predictor = Predictor(  
    endpoint_id=my_endpoint_id,  # Replace with your specific endpoint ID
    api_key=my_api_key  # Replace with your API key
)

# Define the directory where you want to save the images
save_directory = "sample_images"

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Start capturing video from the webcam
with Webcam(fps=0.5) as webcam:
    for frame in webcam:
        frame.resize(width=512)  # Resize the frame to the desired width
        frame.run_predict(predictor=predictor)  # Run the prediction using the face recognition model
        
        # Overlay the predictions on the frame
        frame.overlay_predictions()
        
        # Check for the classes in the predictions
        if any(cls in frame.predictions for cls in ["left", "right", "front"]):
            # Save the image with predictions in the specified directory
            image_path = os.path.join(save_directory, "latest-webcam-image.png")
            frame.save_image(image_path, include_predictions=True)