import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os 
import random
from pallet_n_classnames import pallet_ADE20K, pallet_cityscape, pallet_VOC, pallet_mine, pallet_vistas
from IPython.display import clear_output
from models import use_customdropout, num_of_classes
import matplotlib as mpl
from layers import Dropout
mpl.rcParams['figure.dpi'] = 300
import tensorflow as tf
if tf.__version__ == '1.15.0' or tf.__version__ == '1.13.1':
    from keras import backend as K
    from keras.callbacks import Callback
    from keras.layers import DepthwiseConv2D, SeparableConv2D, Conv2D
if tf.__version__ == '2.0.0' or tf.__version__ == '2.2.0' or tf.__version__ == '2.2.0-rc2':
    import tensorflow.keras.backend as K
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.layers import DepthwiseConv2D, SeparableConv2D, Conv2D
    
use_mydropout = use_customdropout()
if use_mydropout == True:
    from layers import Dropout
elif use_mydropout == False:
    if tf.__version__ == '1.15.0' or tf.__version__ == '1.13.1':
        from keras.layers import Dropout
    if tf.__version__ == '2.2.0' or tf.__version__ == '2.0.0' or tf.__version__ == '2.2.0-rc2' :
        from tensorflow.keras.layers import Dropout
   
    
num_class = num_of_classes()

class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.lr = []
        self.dropout = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.metric1 = self.model.metrics_names[0]
        self.metric2 = self.model.metrics_names[1]
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get(self.metric1))
        self.val_losses.append(logs.get('val_' + self.metric1))
        self.acc.append(logs.get(self.metric2))
        self.val_acc.append(logs.get('val_' + self.metric2))
        self.lr.append(float(K.get_value(self.model.optimizer.lr)))
        if len(self.model.layers) > 10:
            for layer in self.model.layers:
                if isinstance(layer, Dropout) == True:
                    self.dropout.append(K.get_value(layer.rate))
                    break
        else:
            for layer in self.model.layers[-2].layers:
                if isinstance(layer, Dropout) == True:
                    self.dropout.append(K.get_value(layer.rate))
                    break
        self.i += 1
        f, ax = plt.subplots(2, 2, figsize = (10,10), sharex=False)
        
        clear_output(wait=True)
        
        #ax1.set_yscale('log')
        ax[0,0].plot(self.x, self.losses, 'g--*', label=self.metric1)
        ax[0,0].plot(self.x, self.val_losses, 'r-.+', label='val_' + self.metric1)
        ax[0,0].legend()
        
        ax[0,1].plot(self.x, self.acc, 'g--*', label=self.metric2)
        ax[0,1].plot(self.x, self.val_acc, 'r-.+', label='val_' + self.metric2)
        ax[0,1].legend()
        
        ax[1,0].plot(self.x, self.lr, 'm-*', label='Learning Rate')
        ax[1,0].legend()
        
        ax[1,1].plot(self.x, self.dropout, 'c-*', label='Dropout')
        ax[1,1].legend()
        
        plt.show();
        
class PredictionCallback(Callback):
    '''
    Decalre Arguments Input Here
    '''
    def __init__(self, path_train, im_height, im_width):
            self.path_train = path_train
            self.im_height = im_height
            self.im_width = im_width
            self.j = 0
    def on_epoch_end(self, epoch, logs={}):
      random.seed(self.j)
      path_images = self.path_train + '/images/images/'
      random.seed(333)
      n = random.choice([x for x in os.listdir(path_images) 
                           if os.path.isfile(os.path.join(path_images, x))]).rstrip('.jpg')
      
      img = cv2.imread(self.path_train+'/images/images/'+ n + '.jpg')
      if img is None:
            img = cv2.imread(self.path_train+'/images/images/' + n + '.png')
            
      img = cv2.resize(img, (self.im_width, self.im_height))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      
      y_pred = self.model.predict(img[np.newaxis,:,:,:]/255)
      if y_pred.shape[-1] < 2:
          y_pred = sigmoid_activation(y_pred)
      else:
          y_pred = softmax_activation(y_pred)
      y_pred = (y_pred > 0.5).astype(np.uint8)
      y_pred = np.argmax(y_pred.squeeze(), 2)
      y_pred = tf_gray2rgb(y_pred, pallet_mine)
      
      gt = cv2.imread(self.path_train+'/masks/masks/'+ n + '.png', 0)
      if gt is None:
            gt = cv2.imread(self.path_train+'/masks/masks/' + n + '.jpg', 0)
      #gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
      gt = cv2.resize(gt, (self.im_width, self.im_height), interpolation=cv2.INTER_NEAREST)
      self.j += 1
      
      f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15,4), sharex=False)
      f.suptitle('Pred after {} Epoch(s)'.format(epoch+1))
      clear_output(wait=True)
      
      ax1.imshow(img)
      ax1.axis("off")
      
      ax2.imshow(gt)
      ax2.axis("off")
      
      ax3.imshow(y_pred)
      ax3.axis("off")
      
      plt.show();

class CustomLearningRateScheduler(Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, schedule, initial_lr, lr_decay, total_epochs, drop_epoch, power):
        #super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.initial_lr = initial_lr
        self.lr_decay = lr_decay
        self.total_epochs = total_epochs
        self.drop_epoch = drop_epoch
        self.power = power
        
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
            
        if self.schedule == 'step_decay':
            self.schedule = step_decay
        if self.schedule == 'polynomial_decay':
            self.schedule = polynomial_decay
            
        if epoch == 0:
            lr = self.initial_lr
        else:
            # Get the current learning rate from model's optimizer.
            lr = float(K.get_value(self.model.optimizer.lr))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr, self.lr_decay, self.drop_epoch, self.total_epochs, self.power)
        # Set the value back to the optimizer before this epoch starts
        K.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch {}: Learning rate is {}".format(epoch+1, scheduled_lr))

class CustomDropoutScheduler(Callback):
    def __init__(self, schedule, dropout_after_epoch):
        #super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.DAE = dropout_after_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if len(self.model.layers) > 10: # for handeling the multi gpu strategy
            for i in range(len(self.DAE)):
                if epoch == self.DAE[i]:
                    print('Updating Dropout Rate...')
                    for layer in self.model.layers:
                        if isinstance(layer, Dropout):
                            new_drop_out = self.schedule[i]
                            K.set_value(layer.rate, new_drop_out)
            for layer in self.model.layers:
                if isinstance(layer, Dropout) == True:
                    print('Epoch {}: Dropout Rate is {}'.format(epoch+1, np.round(K.get_value(layer.rate),4)))
                    break
        else:
            for i in range(len(self.DAE)):
                if epoch == self.DAE[i]:
                    print('Updating Dropout Rate...')
                    for layer in self.model.layers[-2].layers:
                        if isinstance(layer, Dropout):
                            new_drop_out = self.schedule[i]
                            K.set_value(layer.rate, new_drop_out)
            for layer in self.model.layers[-2].layers:
                if isinstance(layer, Dropout) == True:
                    print('Epoch {}: Dropout Rate* is {}'.format(epoch+1, np.round(K.get_value(layer.rate),4)))
                    break

class CustomKernelRegularizer(Callback):
    '''
    def __init__(self, schedule, dropout_after_epoch):
        #super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.DAE = dropout_after_epoch
    '''
    def on_epoch_begin(self, epoch, logs=None):
        if len(self.model.layers) > 10:
            for layer in self.model.layers:
                if isinstance(layer, Conv2D) == True:
                    print('Conv2D: kernel_regularizer = {}'.format(layer.kernel_regularizer.l2))
                    break
            for layer in self.model.layers:
                if isinstance(layer, SeparableConv2D) == True:
                    print('SeparableConv2D: kernel_regularizer = {}'.format(layer.depthwise_regularizer.l2))
                    break
        else:
            for layer in self.model.layers[-2].layers:
                if isinstance(layer, Conv2D) == True:
                    print('Conv2D: kernel_regularizer = {}'.format(layer.kernel_regularizer.l2))
                    break 
            for layer in self.model.layers[-2].layers:
                if isinstance(layer, SeparableConv2D) == True:
                    print('SeparableConv2D: kernel_regularizer = {}'.format(layer.depthwise_regularizer.l2))
                    break               
#%%   
def sigmoid_activation(pred):
    pred = tf.convert_to_tensor(pred)
    active_preds = tf.keras.activations.sigmoid(pred)
    if tf.executing_eagerly()==False:
        sess = tf.compat.v1.Session()
        active_preds = sess.run(active_preds)
    else:
        active_preds = active_preds.numpy()
        
    return active_preds

def softmax_activation(pred):
    pred = tf.convert_to_tensor(pred)
    active_preds = tf.keras.activations.softmax(pred, axis=-1)
    if tf.executing_eagerly()==False:
        sess = tf.compat.v1.Session()
        active_preds = sess.run(active_preds)
    else:
        active_preds = active_preds.numpy()
        
    return active_preds

def tf_gray2rgb(gray_processed, pallet):
    
    pallet = tf.convert_to_tensor(pallet)
    w, h = gray_processed.shape
    gray = gray_processed[:,:,np.newaxis]
    gray = tf.image.grayscale_to_rgb((tf.convert_to_tensor(gray)))
    gray = tf.cast(gray, 'int32')
    unq = np.unique(gray_processed)
    rgb = tf.zeros_like(gray, dtype=tf.float64)
    
    for i in range(len(unq)):
        clr = pallet[:,unq[i],:]
        clr = tf.expand_dims(clr, 0)
        rgb = tf.where(tf.not_equal(gray,unq[i]), rgb, tf.add(rgb,clr))
        
    if tf.executing_eagerly()==False:
        sess = tf.compat.v1.Session()
        rgb = sess.run(rgb)
    else:
        rgb = rgb.numpy()
    return rgb

def step_decay(epoch, initial_lr, lr_decay, drop_epoch, Epoch, power):
    initial_lrate = initial_lr
    drop = lr_decay
    epochs_drop = drop_epoch
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
def polynomial_decay(epoch, initial_lr, lr_decay, drop_epoch, Epoch, power):
    initial_lrate = initial_lr
    lrate = initial_lrate * math.pow((1-(epoch/Epoch)),power)
    return lrate