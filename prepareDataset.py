#prepareDataset.py
'''This code arranges the dataset for experiments in training and test sets for deep learning models in 70:30 ratio.
The split is random and stratified i.e. training and test set are randomly selected but each label class is 
distrbuted in both sets in same proportions (70:30). Images should be in npz format and annotations in csv format.
Input: path to datasets
Output: Training and Test sets (images_train, images_test, labels_train, labels_test)

Usage: python3 prepareDataset.py
'''

#import libraries
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_labels(imfile,label_dir):
    """This function loads labels from annotation files.
    
    Parameters: 
    imfile: image file whose annotations are to be retrieved
    label_dir: path to directory containinf annotation.csv files
    Returns:
    labels of a patients angiogram images dataset as a pandas series.
    """ 
    label_file = imfile.replace("npz","csv") #get corresponding annotation file  
    label_data = pd.read_csv(label_dir+label_file, sep=",", index_col=0) 
    label_data.set_index(np.arange(0,len(label_data['1'])),inplace=True) #setting index from 0 to 125 instead of 1 to 126
    l2 =label_data['1'] #labels for angiogram images, labels[0]...labels[125]
    return l2


def get_images(imfile,image_dir):
    """This function loads images from npz files.
    
    Parameters: 
    imfile: image file to be loaded
    image_dir: path to directory containing image.npz files
    Returns:
    images of a patient loaded as a 4 dimensional array [n,1,128,128] where n is no of images for a patient. 
    """
    image_data = np.load(image_dir+imfile)
    im = image_data['arr_0'] #array of angiogram images, images[0]...images[125]
    return im

def load_data(image_dir,label_dir):
    """This function reads image files from npz files and keyframe(1)/non keyframe(0) labels 
    from annotations (csv files). Data & labels for all patients are then concatenated to form complete set.
    
    Parameters:
    image_dir,label_dir = psth to directories containing annotations and images
    Returns:
    All the 128x128 pixel angiogram images in an array and their corresponding labels
    
    """
    images=np.empty(shape=(1,1,128,128),dtype='uint8') #initializing an empty array for images
    labels = pd.Series([],dtype=int) #initializing an empty pandas series for labels

    ifiles = os.listdir(image_dir)
    for image_file in ifiles:   
        images=np.append(images,get_images(image_file,image_dir),axis=0)   
        labels =pd.concat([labels,get_labels(image_file,label_dir)],ignore_index=True)

    images = images[1:]  #removing empty array
    return images, labels
    
def main():
    """Load image dataset and their labels and then prepare training and test sets for deep learning models.
    """
    start_time = time.time()
    
    lab_dir = "Angio_Toy_Dataset/csv/" #path to annoattaion files (csv format)
    im_dir = "Angio_Toy_Dataset/npz/" #path to image files (npz format)
    images,labels=load_data(im_dir,lab_dir)
    
    #split the dataset in training and test in 70:30 ratio.
    #Using stratified split i.e.,each class is split in trainiing and test set in same ratio 70:30 
    X_train, X_test, y_train, y_test = train_test_split(images,labels,train_size=0.7,test_size=0.3, stratify=labels, random_state=28)
    print("training set: "+str(len(y_train))+" images and test set: "+str(len(y_test))+" images")
    print("The program took %s seconds" % (time.time() - start_time))

    return (X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    images_train, images_test, labels_train, labels_test = main()
    