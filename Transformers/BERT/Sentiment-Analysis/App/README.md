# Containerize the Application

The application will handle sending/receiving HTTP requests to the TensorFlow Serving API and post the predictions in your local host browser.

### To build the docker image for the Flask app, run the following line in your local terminal:
docker build -t app .

### For Running the image as a container, run the following in your local terminal:
docker run -p 3000:3000 'imageID'
