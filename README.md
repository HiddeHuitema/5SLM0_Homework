# Improving robustness of CNN by highlighting edges
This repository contains the code that accompanies the final assignment of Hidde Huitema (1373005) for the course 5SLM0.
The username that was used to submit models to the CodaLab (https://codalab.lisn.upsaclay.fr/competitions/17868#learn_the_details) competition is HiddeHuitema.

In this project, it was attempted to increase the robustness of a U-net for semantic segmentation by augmenting the Data using a simple edge detector.


## Getting Started
To run this code, clone the repository to your own computer and make sure you have pytorch, numpy and weights&biases installed in your environment
The main train loop is implemented in the train.py file, you can run this code to train a new model. 
--------.ipynb contains code for evaluating the models and creating figures. If you want to see the results of the models I trained, please send me an email so I can send the pretrained models, since they can not be pushed to github due to their size. 


### File Descriptions

Here's a brief overview of the files you'll find in this repository:

- **run_container.sh:** Contains the script for running the container. In this file you have the option to enter your wandb keys if you have them and additional arguments if you have implemented them in the train.py file.

  
- **run_main:** Includes the code for building the Docker container. In this file, you only need to change the settings SBATCH (the time your job will run on the server) and ones you need to put your username at the specified location.
  

- **model.py:** Defines the neural network architecture.

  
- **train.py:** Contains the code for training the neural network.

### Authors

- Hidde Huitema
    h.r.huitema@student.tue.nl 
