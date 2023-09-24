# miniproject
Steel surface defect detection algorithm
import numpy as np
import pandas as pd
import os
train_dir = os.listdir(r"C:\Users\shriv\OneDrive\Desktop\train")
test_dir = os.listdir(r"C:\Users\shriv\OneDrive\Desktop\test")
valid_dir = os.listdir(r"C:\Users\shriv\OneDrive\Desktop\valid")
train_dir
test_dir
valid_dir
import os
# Open a file
path = r"C:\Users\shriv\OneDrive\Desktop\train"
dirs = os.listdir( path )
# This would print all the files and directories
for file in dirs:
    print(file)
for i in train_dir:
    tr = C:\Users\shrir"v\OneDrive\Desktop\train\{}".format(i)
print("Training Inclusion data:",len(os.listdir(tr)))
for i in test_dir:
    te = r"C:\Users\shriv\OneDrive\Desktop\test\{}".format(i)
print("Training Inclusion data:",len(os.listdir(te)))
for i in valid_dir:
    vld = r"C:\Users\shriv\OneDrive\Desktop\valid\{}".format(i)
print("Training Inclusion data:",len(os.listdir(vld)))
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
rescale=1. / 255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
tr_dr = r"C:\Users\shriv\OneDrive\Desktop\train"
# Flow training images in batches of 10 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
tr_dr,
target_size=(200, 200),
batch_size=10,
class_mode='categorical')
# Flow validation images in batches of 10 using test_datagen generator
vl_dr = r"C:\Users\shriv\OneDrive\Desktop\valid"
validation_generator = test_datagen.flow_from_directory(
vl_dr,
target_size=(200, 200),
batch_size=10,
class_mode='categorical')
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.98 ):
            print("\nReached 98% accuracy so cancelling training!")
            self.model.stop_training = True
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (2,2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.summary()
#Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print('Compiled!')
#train the model
callbacks = myCallback()
history = model.fit(train_generator,
        batch_size = 32,
        epochs=20,
        validation_data=validation_generator,
        callbacks=[callbacks],
        verbose=1, shuffle=True)
import matplotlib.pyplot as plt 
plt.figure(1)  
# summarize history for accuracy  
plt.subplot(211)  
plt.plot(history.history['accuracy'])  
plt.plot(history.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
   
 # summarize history for loss  
   
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()
# First, we are going to load the file names and their respective target labels into numpy array! 
from sklearn.datasets import load_files
import numpy as np

test_dir = r'C:\Users\shriv\OneDrive\Desktop\test'

def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files,targets,target_labels
    
x_test, y_test,target_labels = load_dataset(test_dir)
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination

# Define the Bayesian Network
model = BayesianNetwork([('Weather', 'GoPicnic'), ('Temperature', 'GoPicnic')])

# Define Conditional Probability Distributions (CPDs)
cpd_weather = TabularCPD(variable='Weather', variable_card=2, values=[[0.7], [0.3]])
cpd_temperature = TabularCPD(variable='Temperature', variable_card=2, values=[[0.6], [0.4]])
cpd_gopicnic = TabularCPD(variable='GoPicnic', variable_card=2,
                          values=[[0.9, 0.7, 0.6, 0.1], [0.1, 0.3, 0.4, 0.9]],
                          evidence=['Weather', 'Temperature'],
                          evidence_card=[2, 2])

# Add CPDs to the model
model.add_cpds(cpd_weather, cpd_temperature, cpd_gopicnic)

# Check if the model is valid
assert model.check_model()

# Print CPDs
print("CPD for Weather:")
print(cpd_weather)
print("\nCPD for Temperature:")
print(cpd_temperature)
print("\nCPD for GoPicnic:")
print(cpd_gopicnic)

# Create a directed graph (DiGraph)
dag = nx.DiGraph()

# Add nodes and edges to the DiGraph
dag.add_nodes_from(model.nodes())
dag.add_edges_from(model.edges())

# Position nodes using a spring layout
pos = nx.spring_layout(dag)

# Define node colors
node_colors = {'Weather': 'lightblue', 'Temperature': 'lightblue', 'GoPicnic': 'lightgreen'}

# Add nodes and edges to the DiGraph with different colors
for node in model.nodes():
    dag.add_node(node, color=node_colors[node])
dag.add_edges_from(model.edges())

# Position nodes using a spring layout
pos = nx.spring_layout(dag)

# Get node colors from the dictionary
colors = [node_colors[node] for node in dag.nodes()]

# Draw the Bayesian Network as a directed graph with different node colors
nx.draw(dag, pos, with_labels=True, node_color=colors, node_size=1200, font_size=12,
        font_color='black', arrowsize=20)

# Initialize the inference object
inference = VariableElimination(model)

# Show the plot
plt.show()

no_of_classes = len(np.unique(y_test))
no_of_classes
from keras.utils import np_utils

y_test = np_utils.to_categorical(y_test,no_of_classes)
# We just have the file names in the x set. Let's load the images and convert them
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import array_to_img
from tensorflow.keras.utils import load_img
def convert_image_to_array(files):
    images_as_array=[]
    for file in files:
# Convert to Numpy Array
        images_as_array.append(img_to_array(load_img(file)))
    return images_as_array
x_test = np.array(convert_image_to_array(x_test))
print('Test set shape : ',x_test.shape)
x_test = x_test.astype('float32')/255
# Let's visualize test prediction.

y_pred = model.predict(x_test)

# plot a raandom sample of test images, their predicted labels, and ground truth
#Green results in correct prediction 
#Red results in wrong prediction
fig = plt.figure(figsize=(16, 9))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = np.argmax(y_pred[idx])
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(target_labels[pred_idx], target_labels[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))
