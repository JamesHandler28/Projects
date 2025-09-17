import gradio as gr
from transformers import pipeline

# Initializing the image classification pipeline
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

def check_if_car(image_from_gradio):

    # Set settings
    confidence_threshold = 0.05
    car_keywords = ["car", "automobile", "sedan", "convertible", "suv", "jeep", "minivan", "coupe", "racer", "semi", "truck", "van"]

    # Classify the image
    results = image_classifier(image_from_gradio)

    is_car_found = False

    # Loop through each prediction from the model
    for result in results:
        # Split the label into a list of words ("sports car" -> ["sports", "car"])
        label_words = result['label'].split()
        score = result['score']
        
        # Loop through each of our keywords
        for keyword in car_keywords:
            # Check if the keyword is in the list of label words AND the score is high enough
            if keyword in label_words and score > confidence_threshold:
                is_car_found = True
                break # Exit the inner keyword-loop
        
        # If we found a car, we can stop looking at other predictions
        if is_car_found:
            break # Exit the outer prediction-loop

    # Print final answer
    if is_car_found:
        return ("✅ This is a car.")
    else:
        return ("❌ This is NOT a car.")

iface = gr.Interface(
    fn = check_if_car,
    inputs = gr.Image(type="pil", label="Upload an Image"),
    outputs = gr.Textbox(label="Result"),
    title = "🚗 Is it a car?",
    description = "My first machine learning app for GitHub",
    examples = [
        ["car1.jpg"],
        ["car2.jpg"],
        ["car11.jpg"],
        ["car13.jpg"]
    ]
)

if __name__ == "__main__":
    iface.launch()