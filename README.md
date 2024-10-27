# Hyperparameter optimization study for a PyTorch UNET with Optuna.

This project includes a hyperparameter optimization study of a PyTorch UNet for the segmentation of the DRIVE Digital Retinal Images for Vessel Extraction dataset using the hyperparameter optimization framework Optuna.

The UNet hyperparameters chosen to be optimized are:

   - learning rate

   - batch size

   - number of filters of convolutional layers


After the optimization is completed, the program will provide some statistics about the study and it will show the parameters of the best trial. It will also display the overall results and save them in a .csv file for future reference. Lastly, it will find and display the most important hyperparameters based on completed trials in the given study.



## About the dataset
The DRIVE: Digital Retinal Images for Vessel Extraction dataset dataset. The datasets is publilicly available and composed of 40 high res images for retinal vessel segmentation.

