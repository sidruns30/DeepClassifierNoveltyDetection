import numpy as np
import keras.backend as K
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import joblib
from light_curve import LightCurve # For using Naul's LightCurve class
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from survey_rnngmm_classifier import main as survey_autoencoder
from survey_rnngmm_classifier import preprocess, energy_loss
from keras.models import Model
from keras_util import parse_model_args, get_run_id

# For one-hot vector conversion
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

### For GMM training
from gmm import GMM
from autoencoder import extract_features
from estimation_net import EstimationNet
from keras.layers import Input, Lambda
from keras.optimizers import Adam
import keras_util as ku

### For novelty detection scores
from sklearn.metrics import precision_recall_fscore_support

def main(args=None):
  args = parse_model_args(args)
  K.set_floatx('float64')
  
  run = get_run_id(**vars(args))
  log_dir = os.path.join(os.getcwd(), 'keras_logs', args.sim_type, run) #Loading the model architecture 
  weights_path = os.path.join(log_dir, 'weights.h5') #Loading the model weights

  print ("log_dir", log_dir)
  print("Weight matrix read...")
  
  #Load the model
  #How do I access dict using index? I just want to load the model  
  #Why not use the main args?
  model = list(survey_autoencoder(vars(args)))[2]
  
  #LOADING GMM PARAMTERS
  # Where is gmm.mu updated?
  gmm_para = np.load(log_dir+'/gmm_parameters')
  gmm_mu = gmm_para[gmm_mu] #Size = embedding size * #classes
  
  #LOADING THE MODEL
  decode_model = Model(inputs=model.input, outputs=model.get_layer('time_dist').output)

  #What is X[valid], X[new]
  #TRAINING SAMPLES
  gmm_mu = np.float64(gmm_mu)
  #NO AUX INPUT FOR DECODER! DOES IT ASSUME EVEN INPUTS?
  decoding_train = decode_model.predict(gmm_mu)
  print(decoding_train.shape)
  
  phase = np.linspace(0,1,len(decoding_train[0]))
  for i in range(len(decoding_train)):    
    plt.plot(phase, decoding_train[i])  
 
if __name__ == '__main__':
    args = main()
  
  
  
  
  
  
  
  
  
  
