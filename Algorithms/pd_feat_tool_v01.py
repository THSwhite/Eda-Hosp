import numpy as np
import matplotlib.pyplot as plt

###############################################
###############################################
## Statistics for intervals...
###############################################
###############################################

def mid_of_intervals(input_PD):
  result_list = []
  ##
  for i in range(len(input_PD)):
    the_mid = (input_PD[i][0] + input_PD[i][1])/2
    result_list.append(the_mid)
  ##
  return result_list

def lifespan_of_intervals(input_PD):
  result_list = []
  ##
  for i in range(len(input_PD)):
    the_mid = input_PD[i][1] - input_PD[i][0]
    result_list.append(the_mid)
  ##
  return result_list

def deaths_of_intervals(input_PD):
  result_list = []
  ##
  for i in range(len(input_PD)):
    the_d = input_PD[i][1]
    result_list.append(the_d)
  ##
  return result_list

def births_of_intervals(input_PD):
  result_list = []
  ##
  for i in range(len(input_PD)):
    the_b = input_PD[i][0]
    result_list.append(the_b)
  ##
  return result_list

###############################################
###############################################
## Histogram generation...
###############################################
###############################################

def gen_histogram(input_list,
                 histogram_bins=np.linspace(0,100,10),
                 plot_the_histogram=False):
  ## Step 1: Clean data
  arr = np.array(input_list, dtype=float)
  arr = arr[np.isfinite(arr)]  # drop NaN/Inf
  ## Step 2: Generate the histogram
  histogram = np.histogram(arr, bins=histogram_bins)
  ## Step 3 (optional): Plot the histogram
  if plot_the_histogram == True:
    plt.hist(arr, bins=histogram_bins)
    plt.title("Histogram")
    plt.show()
  ## Step 4: Return the result
  return histogram