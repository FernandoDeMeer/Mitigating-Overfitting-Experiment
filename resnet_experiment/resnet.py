#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:51:39 2019

@author: fernandodemeer
"""


# ResNet
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import keras 
import numpy as np 
import time
import warnings
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

class Classifier_RESNET: 

	def __init__(self, output_directory, input_shape, nb_classes, verbose=2):
		self.output_directory = output_directory
		self.model = self.build_model(input_shape, nb_classes)
		if(verbose==2):
			self.model.summary()
		self.verbose = verbose
		self.model.save_weights(self.output_directory+'model_init.hdf5')

	def build_model(self, input_shape, nb_classes):
		n_feature_maps = 64

		input_layer = keras.layers.Input(input_shape)
		
		# BLOCK 1 

		conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		# expand channels for the sum 
		shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
		shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

		output_block_1 = keras.layers.add([shortcut_y, conv_z])
		output_block_1 = keras.layers.Activation('relu')(output_block_1)

		# BLOCK 2 

		conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		# expand channels for the sum 
		shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1, padding='same')(output_block_1)
		shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

		output_block_2 = keras.layers.add([shortcut_y, conv_z])
		output_block_2 = keras.layers.Activation('relu')(output_block_2)

		# BLOCK 3 

		conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_2)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		# no need to expand channels because they are equal 
		shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

		output_block_3 = keras.layers.add([shortcut_y, conv_z])
		output_block_3 = keras.layers.Activation('relu')(output_block_3)

		# FINAL 
		
		gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

		file_path = self.output_directory + 'weights,epoch={epoch:02d} validation_loss={val_loss:.2f} validation_acc={val_acc:.2f}.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_acc',
			save_best_only=True,period= 250)

		self.callbacks = [reduce_lr,model_checkpoint]


		return model
	
	def fit(self, x_train, y_train, x_val, y_val,y_true,n_epochs):
		# x_val and y_val are only used to monitor the test loss and NOT for training  
		batch_size = 64

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		start_time = time.time() 

		model_history = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=n_epochs,
			verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks,class_weight={0: 1.,1: 10.})

		# summarize history for accuracy
		plt.plot(model_history.history['acc'])
		plt.plot(model_history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig('resnet_vega/Losses/Training_Accuracy')
		plt.show()
		# summarize history for loss
		plt.figure()
		plt.plot(model_history.history['loss'])
		plt.plot(model_history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig('resnet_vega/Losses/Training_Loss')
		plt.show()

	def predict(self,x_val):

		y_pred = self.model.predict(x_val)

		# convert the predicted from binary to integer
		y_class_pred = np.argmax(y_pred , axis=1)

		keras.backend.clear_session()

		return y_pred,y_class_pred



	def plot_confusion_matrix(self,y_true, y_class_pred, classes):
		def get_confusion_matrix_plots(y_true, y_class_pred, classes,
								  normalize=False,
								  title=None,
								  cmap=plt.cm.Blues):
			"""
			This function prints and plots the confusion matrix.
			Normalization can be applied by setting `normalize=True`.
			"""
			if not title:
				if normalize:
					title = 'Normalized confusion matrix'
				else:
					title = 'Confusion matrix, without normalization'

			# Compute confusion matrix
			cm = confusion_matrix(y_true, y_class_pred)
			# Only use the labels that appear in the data
			classes=np.array(classes)
			classes = classes[unique_labels(y_true, y_class_pred)]
			if normalize:
				cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
				print("Normalized confusion matrix")
			else:
				print('Confusion matrix, without normalization')

			print(cm)

			fig, ax = plt.subplots()
			im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
			ax.figure.colorbar(im, ax=ax)
			# We want to show all ticks...
			ax.set(xticks=np.arange(cm.shape[1]),
				   yticks=np.arange(cm.shape[0]),
				   # ... and label them with the respective list entries
				   xticklabels=classes, yticklabels=classes,
				   title=title,
				   ylabel='True label',
				   xlabel='Predicted label')

			# Rotate the tick labels and set their alignment.
			plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
					 rotation_mode="anchor")

			# Loop over data dimensions and create text annotations.
			fmt = '.2f' if normalize else 'd'
			thresh = cm.max() / 2.
			for i in range(cm.shape[0]):
				for j in range(cm.shape[1]):
					ax.text(j, i, format(cm[i, j], fmt),
							ha="center", va="center",
							color="white" if cm[i, j] > thresh else "black")
			fig.tight_layout()
			return ax

		np.set_printoptions(precision=2)

		# Plot non-normalized confusion matrix
		get_confusion_matrix_plots(y_true, y_class_pred, classes=classes,title='Confusion matrix, without normalization')

		# Plot normalized confusion matrix
		get_confusion_matrix_plots(y_true, y_class_pred, classes=classes, normalize=True,title='Normalized confusion matrix')

		plt.show()

	def load(self,epoch,val_loss,val_acc):

		model = keras.models.load_model(self.output_directory + 'weights,epoch={} validation_loss={} validation_acc={}.hdf5'.format(epoch,val_loss,val_acc))

		self.model = model
