## If you do not have the following packages, install them.
## pip install ripser persim
## pip install gudhi
## pip pip install opencv-python

## Import packages...
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import persim
import cv2

###############################################
###############################################
## Pickle tools...
###############################################
###############################################

def save_list_by_pickle(input_list, target_path):
  with open(target_path, 'wb') as f:
    pickle.dump(input_list, f)

###############################################
###############################################
## Imaging tools...
###############################################
###############################################

## Given a path of a png file, simply read it...
def read_from_PNG(file_path):
  # Read the PNG file
  img = cv2.imread(file_path)
  return img

## Given a 2D np.array, a dpi, and a color map, plot it...
def plot_img(img, dpi=100, cmap='gray'):
  _, axes = plt.subplots(dpi=dpi)
  plt.imshow(img, cmap=cmap)
  plt.show()

## Basic tool for find indices in a numpy array.
def find(condition):
  res = np.nonzero(condition)
  return res

## Thresholding operation (smaller, the typical one) on an image.
def biImg_by_threshold_leq(img, threshold):
  output_img = np.copy(img)
  idxs_0 = find(img <= threshold)
  idxs_1 = find(img > threshold)
  output_img[idxs_0] = 0
  output_img[idxs_1] = 1
  return output_img

###############################################
###############################################
## Core algorithms...
###############################################
###############################################

## Given a 2d np.array, we output the 1d-arrray for Gudhi.
def img_to_1d_array(img):
  result = []
  img_shape = np.shape(img)
  for i in range(img_shape[0]):
    result = list(img[i,:]) + result
  return np.array(result)

## Given a 2d np.array of values, compute its 0th and 1st persistence diagrams.
def persistence_of_img(img, do_plot=False, save_with_pickle=True, save_path="tmp.pd"):
  img_shape = np.shape(img)
  nx = img_shape[1]
  ny = img_shape[0]
  filt_values = img_to_1d_array(img)
  cubical_complex = gd.CubicalComplex(dimensions = [nx ,ny], top_dimensional_cells = filt_values)
  cubical_complex.compute_persistence()
  persistence_0 = cubical_complex.persistence_intervals_in_dimension(0)
  persistence_1 = cubical_complex.persistence_intervals_in_dimension(1)
  if (do_plot == True):
    plt.figure()
    persim.plot_diagrams(list([persistence_0, persistence_1]), title="Persistence Diagram")
    plt.show()
  ##
  if (save_with_pickle==True):
    save_list_by_pickle(input_list=[persistence_0, persistence_1], target_path=save_path)
  ##
  return [persistence_0, persistence_1]

## Given a nd np.array of values (2D images with channels), compute its 0th and 1st persistence diagrams.
def persistence_of_img_with_channels(img, save_with_pickle=True, save_path="tmp.pd"):
  img_shape = np.shape(img)
  nx = img_shape[1]
  ny = img_shape[0]
  number_of_channels = img_shape[2]
  PB_in_channels = []
  for i in range(number_of_channels):
      filt_values = img_to_1d_array(img[:,:,i])
      cubical_complex = gd.CubicalComplex(dimensions = [nx ,ny], top_dimensional_cells = filt_values)
      cubical_complex.compute_persistence()
      persistence_0 = cubical_complex.persistence_intervals_in_dimension(0)
      persistence_1 = cubical_complex.persistence_intervals_in_dimension(1)
      PB_in_channels.append([persistence_0, persistence_1])
  ##
  if (save_with_pickle==True):
    save_list_by_pickle(input_list=PB_in_channels, target_path=save_path)
  ##
  return PB_in_channels

## Given a collection of 0th and 1st persistence diagrams generated from image, plot their barcodes through Gudhi.
def plot_PBs(input_PD_collection):
    plt.figure()
    gd.plot_persistence_barcode([input_PD_collection[0], input_PD_collection[1]])
    plt.show()
