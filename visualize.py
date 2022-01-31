#visuaize.py
'''This code reads image and annotation data from Angio toy dataset, and plots the given number of keyframes and non keyframes for a patient.
No of images (n) and patient id (p) can be specified as per requirement. 
Better visualization of saved plot for n = 1 to 10. Following that pdf file may become bulky.

Input: p = patient_id whose data is to be visualized,n = No of images to be plotted per category
Output: angio.pdf
Usage: python3 visualize.py
'''

#import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import time


def read_data(patient_id):
    
    """This function reads image (.npz) and annotation (.csv) files, retrieves indexes for keyframe and nonkeyframe images
    and loads images as a numpy array.
    
    Input files:
    .npz files = Each file contains 128x128 pixel angiogram keyframe and non-keyframe images for a patient. 
    .csv filed =  Annotation file that contains keyframe (1) and non keyframe (0) labels for angiogram images in
    corresponding.npz files denoting if the image is ideal for future analysis or not.
    
    Note: npz file size ((126, 1, 128, 128)) and csv file size ((127, 1)) don't match, so assuming first row in csv file is a header
    
    Parameters:
    patient_id = id of patient whose data is to be read
    
    Returns:
    key_index = index for keyframe labels
    nonkey_index = index for non keyframe labels
    images = array of angiogram images of 128x128 pixels for the patient
    
    """
    #read image files
    try:
        imageFile = "Angio_Toy_Dataset/npz/npz-"+str(patient_id)+".npz" 
        imageData = np.load(imageFile)
        images = imageData['arr_0'] #array of angiogram images 

        #read keyframe (1) and non keyframe (0) labels from annotations (csv files)
        labelFile = "Angio_Toy_Dataset/csv/csv-"+str(patient_id)+".csv" 
        labels = pd.read_csv(labelFile,sep=",", index_col=0) 

        #Note: npz file size (images['arr_0'].shape= (126, 1, 128, 128)) and csv file size (labels.shape= (127, 1)) don't match, 
        #so assuming first row in csv file is a header

        #get index for keyframes and non-keyframes
        key_index = labels[labels['1']==1].index # index for keyframe labels
        nonkey_index = labels[labels['1']==0].index #index for non keyframe labels

        #images.files
        #print(images['arr_0'].shape)

        return key_index,nonkey_index,images
    
    except FileNotFoundError:
        sys.exit('\nSorry, data for patient number ' +str(patient_id)+ ' not found.\n')
        
    
def visualize_images(output,p,n=1):
    
    """This function plots images and their annotations for a patient. 
    Given patient id (p) and no. of images (n), it saves the plot with annotations in a (output) file.
    Number of images  to be visualized can be changed using parameter 'n'. 
    
    Parameters:
    output = output file in which plot is to be saved.
    n = No of images to be plotted per category of keyframes and non keyframes. 
        Its value should be less than number of keyframe and non keyframes.
        Default value of n is 1, so one image from each keyframe and non keyframe category is plotted.
    p = patient_id whose data is to be visualized. In toy dataset p ranges from 1 to 5.
    
    Output:
    output : Plot is saved as a file where  Keyframe (left) and non keyframe  (right) images are labelled kf_imageid and 
        nkf_imageid respectively    
    """
  
    kf,nkf,images=read_data(patient_id=p)
    
    fig, ax = plt.subplots(nrows=n, ncols=2, sharey=True, constrained_layout=True, figsize=(4,2*n))
    index=0
    for i,j in list(zip(kf,nkf))[:n]:
        title1= "kf_img"+str(i)
        title2= "nkf_img"+str(j)
        
        if n==1:
            ax[index].imshow(images[i-1,0],cmap="gray",interpolation='nearest')
            ax[index].set_title(title1,fontsize=10)
            ax[1].imshow(images[j-1,0],cmap="gray",interpolation='nearest')
            ax[1].set_title(title2,fontsize=10)
        else :
            ax[index][0].imshow(images[i-1,0],cmap="gray",interpolation='nearest')
            ax[index][0].set_title(title1,fontsize=10)
            ax[index][1].imshow(images[j-1,0],cmap="gray",interpolation='nearest')
            ax[index][1].set_title(title2,fontsize=10)
        index+=1

    plt.suptitle ("Sample images of keyframes (left) and non-keyframes (right)")
    plt.savefig(output, bbox_inches='tight')    
    #plt.show()     

def main():
    start_time = time.time()

    visualize_images(output='angio.pdf',p=1,n=4) #plot n=4 keyframe/non keyframe images for patient number 1(p=1)
    #visualize_images(p=2,n=3) #plot 3 keyframe/non keyframe images for patient number 2
    print("The program took %s seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()