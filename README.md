# RealTimeSignLanguageDetection
Real Time Sign Language Detection using Keras-CNN and OpenCV

collect-data.py - To collect images under train and test folders using OpenCV
train.py - To train the CNN model using Keras. The model is saved in two files: 'model-bw.json' and 'model-bw.h5'.

predict.py - For real time prediction using the trained model from before and OpenCV


The custom dataset used in the model consists of a total of 1000 images per label for training and 100 images per label for testing. This decision was made to ensure that the model had enough data to learn from and could generalize well to new, unseen images. The four gestures that were considered for the dataset are pain, stop, hurts, and iloveyou.
To capture images from the web camera and save them in the appropriate directory based on the user's gesture, OpenCV was used in the first part of the code. The code listens for key presses to save the images to the appropriate subdirectory based on the category of the hand gesture. For example, if the user presses the '0' key, the code saves the image to the "pain" subdirectory of the "train" folder.
The images captured are saved in the "data" directory in the appropriate subdirectory based on the mode - train or test. The Region of Interest (ROI) is extracted from the video frames using the coordinates of the rectangular region drawn on the video frames. The extracted ROI is then resized to 64x64 pixels and converted to grayscale using OpenCV.

In the repository, I have uploaded 50 images per label fir reference. More images can be stored under train and test folders by running 'collect-data.py' as per your requirement.
