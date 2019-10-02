

# Mitigating Overfitting with GANs

In this library you can find all the necessary code to reproduce the "Forecasting VIX peaks" experiment of :


## Installation

In order to install the necessary libraries : 

- Create a virtualenv with python 3.6, activate it, set your current directory to this repo and then run:
```
pip install -r requirements.txt
```

## Usage

 ``main.py``  is the script that runs the experiment. From the parser you can change the settings of the experiment.
 
In order to train the ResNet on the two different training sets (just real series vs enlarged training set) simply change the following option in the parser : 
```python
parser.add_argument('--improved_training', type=bool, default=False) # Enlarge the training set with synthetic series
``` 
Which will load the synthetic VIX series produced by a  WGAN-GP designed to work on time series data.

In both training settings, the ResNet weights will be saved in ``resnet_experiment\Saved models`` and the loss/accuracy plots in ``resnet_experiment\Losses``. 

## Checking Results

To evaluate a trained ResNet (i.e a set of weights) and plot the resulting confusion matrix, fill out the following parser options with the corresponding parameters: 
```python
parser.add_argument('--load', type=bool, default=False) #Load a saved model or start training from scratch  
parser.add_argument('--predict', type=bool, default=False)  # Predict classes on an out-of-sample dataset (change x_test_data accordingly) and plot the confusion matrix  
parser.add_argument('--epoch_to_load', type=int, default=0) # If you have a saved model,change the default to the epochs of the trained model   
parser.add_argument('--val_loss_to_load', type=float, default=0) # If you have a saved model,change the default to your val_loss  
parser.add_argument('--val_acc_to_load', type=float, default=0) # If you have a saved model,change the default to your val_acc
``` 

Enjoy!

![Alt Text](https://media.giphy.com/media/TKG7SiiN7lauE1aXHH/giphy.gif)

