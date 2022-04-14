Required Packages:

Sys
String
Pickle
Numpy
Pandas
OpenCV
Sklearn & Skimage
Tensorflow

How to Run the Code:

create_data.py creates a folder of images and a csv file from a image of ascii characters. 
It is hard coded to receive an image of ascii characters in a specific order and format. 
The provided image 'ascii.png' is included as an example. It is a screenshot of ascii characters seperated by a space using Arial 20pt. 
The following command will run the code with the included image: 
python create_data.py ascii.png

train_model.py will create a folder and a file that are the models used to predict the character in an image and translate the results to an ascii character. 
The code requires the user to input a csv file containing the path to character images and their labels and the number of epochs to train the model.
The following command will run the code with the dataset and train the model for 5000 epochs (on my PC it took around 5 minutes with a 3060ti): 
python train_model.py data.csv 5000

read_image.py will attempt to print the text from an image. 
The code requires the user to input folder contaning the keras model, file with a model that will translate the output, and an image to read. 
There are 2 test images provided. They are screenshots of partial Wikipedia entries for Google and The Vancouver Sun from a Word doc with Arial 20pt. 
The following command will run the code with the included files:
python read_image.py model labelsLB.dat test1.png

The included Word doc is where the screenshots were taken.

