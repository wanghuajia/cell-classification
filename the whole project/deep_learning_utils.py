from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta
from keras.optimizers import Adam
from keras.optimizers import Adamax
from keras.optimizers import Nadam

import numpy as np


import os
from pathlib import Path
import sys
import glob
import metrics

def get_label_from_filename(filename):
	return filename.split("/")[0]

def get_class_from_prediction_vector(vector,num_classes):
	if(num_classes==2):
		classes=['cancer_plasma', 'lymphocyte']
		return classes[np.argmax(vector)]
	
	classes=['cancer_cell', 'lymphocyte', 'plasma_cell']
	return classes[np.argmax(vector)]


def evaluate_model(trained_model,test_data_dir,phase,num_classes,img_width,img_height,output_dir):
	num_samples=len(glob.glob(test_data_dir+"/*/*.jpg"))

	datagen = ImageDataGenerator(rescale=1. / 255)

	generator = datagen.flow_from_directory( 
		test_data_dir,
		target_size=(img_width, img_height),
		batch_size=num_samples,
		shuffle=False,
		class_mode='categorical')


	evaluate = trained_model.evaluate_generator(generator,1)
	print(phase+" metrics:")
	print('Loss: ' + str(evaluate[0]))
	print('Accuracy: ' + str(evaluate[1]))

	predictions_test = trained_model.predict_generator(generator,1)

	file = open(output_dir+phase+'_results.txt','w')
	#format:
	#<filename><label><prediction><output_vector>
	for filename,y in zip(generator.filenames,predictions_test):
		file.write(filename+ " " + get_label_from_filename(filename)+ " " +get_class_from_prediction_vector(y,num_classes)+ " ")
		for yi in y:
			file.write(str(yi)+ " ")
		file.write("\n")
	file.close()



def define_learning_rate_and_optimizer(optimizer, learning_rate):
	if(learning_rate>0):
		if(optimizer=='Adam'):
			return Adam(lr=learning_rate),learning_rate
		elif(optimizer=='Adamax'):
			return Adamax(lr=learning_rate),learning_rate
		elif(optimizer=='Adadelta'):
			return Adadelta(lr=learning_rate),learning_rate
		elif(optimizer=='Adagrad'):
			return Adagrad(lr=learning_rate),learning_rate
		elif(optimizer=='Nadam'):
			return Nadam(lr=learning_rate),learning_rate
		elif(optimizer=='RMSprop'):
			return RMSprop(lr=learning_rate),learning_rate
		elif(optimizer=='SGD'):
			return SGD(lr=learning_rate),learning_rate

	#learning_rate is the default value
	if(optimizer=='Adam'):
		return Adam(),1e-3
	elif(optimizer=='Adamax'):
		return Adamax(),2e-3
	elif(optimizer=='Adadelta'):
		return Adadelta(),1
	elif(optimizer=='Adagrad'):
		return Adagrad(),1e-2
	elif(optimizer=='Nadam'):
		return Nadam(),2e-3
	elif(optimizer=='RMSprop'):
		return RMSprop(),1e-3
	elif(optimizer=='SGD'):
		return SGD(),1e-2

def define_model(img_width, img_height, defined_optimizer, model_name='simple_cnn', num_classes=2):
	print("Training "+ model_name + " with above dataset configuration...")
	print("Keras and TensorFlow")
	if K.image_data_format() == 'channels_first':
		input_shape = (3, img_width, img_height)
	else:
		input_shape = (img_width, img_height, 3)

	model = Sequential()
	if(model_name=='simple_cnn'):
		model.add(Conv2D(50, (3, 3), input_shape=input_shape, activation='relu', kernel_initializer='he_normal', name="CONV1"))
		model.add(MaxPooling2D(pool_size=(2, 2), name="MP1"))
		model.add(Conv2D(100, (3, 3), activation='relu', kernel_initializer='he_normal',name="CONV2"))
		model.add(MaxPooling2D(pool_size=(2, 2),name="MP2"))
		model.add(Flatten())
		model.add(Dense(500, activation='relu', kernel_initializer='he_normal', name="FC1"))
		model.add(Dropout(0.5))
		model.add(Dense(num_classes, activation='softmax',  name="FC2_output"))

	elif(model_name=='encoder_net'):
		model.add(Flatten( input_shape=input_shape, name="Input"))
		model.add(Dense(4096, activation='relu', kernel_initializer='he_normal',name="FC1"))
		model.add(Dropout(0.5))
		model.add(Dense(256, activation='relu',kernel_initializer='he_normal', name="FC2"))
		model.add(Dropout(0.5))
		model.add(Dense(num_classes, activation='softmax',name="FC3_output"))


	model.compile(loss='categorical_crossentropy', optimizer=defined_optimizer, metrics=['accuracy', metrics.precision, metrics.recall, metrics.fscore])

	#print("Chosen SGD with lr="+str(learning_rate)+", epochs="+str(epochs)+", batch_size="+str(batch_size))
	return model



def get_images_counts_in_fold(fold='01',num_classes=2):
	if(num_classes==2):
		filelist_cancer_plasma_train  = glob.glob("./data_fold_"+fold+"/train/cancer_plasma/*.jpg")
		filelist_lymphocyte_train  = glob.glob("./data_fold_"+fold+"/train/lymphocyte/*.jpg")

		filelist_cancer_plasma_validation  = glob.glob("./data_fold_" +fold+"/validation/cancer_plasma/*.jpg")
		filelist_lymphocyte_validation  = glob.glob("./data_fold_"+fold+"/validation/lymphocyte/*.jpg")

		filelist_cancer_plasma_test  = glob.glob("./data_fold_" +fold+"/test/cancer_plasma/*.jpg")
		filelist_lymphocyte_test  = glob.glob("./data_fold_"+fold+"/test/lymphocyte/*.jpg")

		Total_train = len(filelist_cancer_plasma_train) +len(filelist_lymphocyte_train) 
		Total_validation = len(filelist_cancer_plasma_validation) +len(filelist_lymphocyte_validation) 
		Total_test = len(filelist_cancer_plasma_test) +len(filelist_lymphocyte_test) 

		print("Dataset Summary Fold "+fold)
		print("Class\t#Training\t\t#Validation#\t#Test")
		print("Cancer_cell\t"+ str(len(filelist_cancer_plasma_train)) + "\t\t" +
			"\t"+ str(len(filelist_cancer_plasma_validation))  +
			"\t"+ str(len(filelist_cancer_plasma_test)))
		print("Lymphocyte\t"+ str(len(filelist_lymphocyte_train)) + "\t\t" +
			"\t"+ str(len(filelist_lymphocyte_validation))  +
			"\t"+ str(len(filelist_lymphocyte_test)))


		return Total_train,Total_validation,Total_test

	else:
		filelist_cancer_cell_train  = glob.glob("./data_fold_"+fold+"/train/cancer_cell/*.jpg")
		filelist_plasma_cell_train  = glob.glob("./data_fold_"+fold+"/train/plasma_cell/*.jpg")
		filelist_lymphocyte_train  = glob.glob("./data_fold_"+fold+"/train/lymphocyte/*.jpg")

		filelist_cancer_cell_validation  = glob.glob("./data_fold_" +fold+"/validation/cancer_cell/*.jpg")
		filelist_plasma_cell_validation  = glob.glob("./data_fold_" +fold+"/validation/plasma_cell/*.jpg")
		filelist_lymphocyte_validation  = glob.glob("./data_fold_"+fold+"/validation/lymphocyte/*.jpg")

		filelist_cancer_cell_test  = glob.glob("./data_fold_" +fold+"/test/cancer_cell/*.jpg")
		filelist_plasma_cell_test  = glob.glob("./data_fold_" +fold+"/test/plasma_cell/*.jpg")
		filelist_lymphocyte_test  = glob.glob("./data_fold_"+fold+"/test/lymphocyte/*.jpg")

		Total_train = len(filelist_cancer_cell_train) +len(filelist_lymphocyte_train)  + len(filelist_plasma_cell_train) 
		Total_validation = len(filelist_cancer_cell_test) +len(filelist_lymphocyte_test) + len(filelist_plasma_cell_test)
		Total_test = len(filelist_cancer_cell_test) +len(filelist_lymphocyte_test) + len(filelist_plasma_cell_test)

		print("Dataset Summary Fold "+fold)
		print("Class\t#Training\t\t#Validation#\t#Test")
		print("Cancer_cell\t"+ str(len(filelist_cancer_cell_train)) + "\t\t" +
			"\t"+ str(len(filelist_cancer_cell_validation))  +
			"\t"+ str(len(filelist_cancer_cell_test)))
		print("Lymphocyte\t"+ str(len(filelist_lymphocyte_train)) + "\t\t" +
			"\t"+ str(len(filelist_lymphocyte_validation))  +
			"\t"+ str(len(filelist_lymphocyte_test)))
		print("Plasma_cell\t"+ str(len(filelist_plasma_cell_train)) + "\t\t" +
			"\t"+ str(len(filelist_plasma_cell_validation))  +
			"\t"+ str(len(filelist_plasma_cell_test))  )


		return Total_train,Total_validation,Total_test


