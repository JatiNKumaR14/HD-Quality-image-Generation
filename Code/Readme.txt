ReadMe
Project have been divided into 3 parts
1. Classification (N-Layer CNN Classifier)
2. Image Generation (WGAN-GP_128x128)
3. Super Resolution (SR-DenseNet)

For execution of the code u need to follow this steps 

A) Training 
1. Download the Dataset(Dataset link is provided in Dataset.txt file)
2. Open the Classification Colab file
3. Set the directory path to the dataset 
4. Note the Accuracy 
5. Open Pipelined colab file of Image Generation and Super Resolution
6. Set the directory path to the dataset 
7. Execute the file and will get the output as a zip file of generated images of 5000 samples 
8. Extract that file and remove uncessary/bad images and replace it with original images 
9. Open the Classification file again
10. Train the Classifire with new images by changing the dataset directory
11. Note the accuracy


B) Testing
1. Open Test.py
2. Set the path of image that need to be tested
3. Observe the output
