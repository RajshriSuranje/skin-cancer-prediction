import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request


# Create Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("model.h5")  # Replace with the path to your saved model


# Define the route for the home page
@app.route("/")
def home():
    return render_template("index.html")


# Define the route for the prediction
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get the uploaded image from the request
        file = request.files["image"]

        # Save the image to a temporary location
        file.save(file.filename)  # type: ignore
        img_path = file.filename

        # Preprocess the image
        img = image.load_img(img_path, target_size=(150, 150))
        img = image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Make the prediction
        prediction = model.predict(img)  # type: ignore
        if prediction[0] < 0.5:
            result = "Non Melanoma"
        else:
            result = "Melanoma"

        # Render the prediction result template
        return render_template("result.html", result=result)

    return render_template("index.html")


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
