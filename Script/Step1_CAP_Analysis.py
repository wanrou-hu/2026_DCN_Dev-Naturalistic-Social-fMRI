# -*- coding: utf-8 -*-
"""
Co-activation pattern (CAP) analysis of naturalistic movie-viewing fMRI.

This script identifies recurring whole-brain CAP states from ROI-level BOLD
signals pooled across children and adults, and quantifies subject-level
brain-state dynamics.

Analysis workflow
-----------------
1. Load ROI time series (TR x ROI x subject) and apply temporal z-score
   normalization independently to each ROI within each participant.
2. Pool all subject time points and evaluate k = 2-9 using k-means++
   (1,000 initializations per k). Summarize clustering inertia and silhouette
   scores to inspect the choice of the number of CAP states.
3. Fit the four-state solution reported in the paper, assign a CAP label to
   each TR, and calculate subject-level fractional occurrence and
   transition probabilities between different CAP states.
"""

# %reset
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from collections import Counter

n_rois = 432
path_output = '...' ## output directory

# =================================================================================
# Step 1: Load the time series data
# =================================================================================
time_series_data = np.load('ROIsignals_input.npy')
print(time_series_data.shape) # (TR, ROI, Subjects)


# =================================================================================
# Step 2: Within-participant temporal z-score normalization
# =================================================================================
# Normalize each ROI time series across TRs independently for each
# participant (axis = 0; population SD, ddof = 0).
roi_mean = np.mean(time_series_data, axis=0, keepdims=True)
roi_std = np.std(time_series_data, axis=0, keepdims=True)

z_scored_data = (time_series_data - roi_mean) / roi_std

# Concatenate all subject-level frames into a 2D matrix for clustering.
# Each row represents a whole-brain activation pattern at one TR.
z_scored_data_reshaped = z_scored_data.transpose(0, 2, 1)
z_scored_data_reshaped = z_scored_data_reshaped.reshape(-1, n_rois)


# =================================================================================
# Step 3: searching optimal k for co-activation prototypes
# =================================================================================
n_timepoints, n_rois, n_subjects = time_series_data.shape

z_scored_data_reshaped = z_scored_data.transpose(0, 2, 1)  # (TR, Subjects, ROI)
z_scored_data_reshaped = z_scored_data_reshaped.reshape(-1, 432) # (TR * Subjects, ROI)
print(z_scored_data_reshaped.shape)


## initiate clustering parameters
n_clusters_range = range(2, 10)  # Cluster range from 2 to 9
best_kmeans = None
best_inertia = float('inf')

inertia_per_k = []  # Store the lowest inertia for each K
silhouette_scores = []  # Store silhouette scores for each K

for n_clusters in n_clusters_range:
    lowest_inertia = float('inf')
    best_kmeans_for_k = None
    for _ in range(1000):  # Repeat clustering 1000 times
        kmeans = KMeans(n_clusters = n_clusters, random_state = None, n_init = 1)
        kmeans.fit(z_scored_data_reshaped)
        
        if kmeans.inertia_ < lowest_inertia:
            lowest_inertia = kmeans.inertia_
            best_kmeans_for_k = kmeans
            
    inertia_per_k.append(lowest_inertia)
    silhouette_avg = silhouette_score(z_scored_data_reshaped, best_kmeans_for_k.labels_)
    silhouette_scores.append(silhouette_avg)
    
np.save(path_output + 'SelectK_Inertia.npy', inertia_per_k)


## Optial clusters was determined as k with the maximize curvature
first_diff = np.diff(inertia_per_k) # the first-order difference
second_diff = np.diff(first_diff) # the second-order difference (i.e., curvature)

optimal_k_index = np.argmax(second_diff) + 2
optimal_k = n_clusters_range[optimal_k_index]


####################################################################################
# Plot inertia vs. number of clusters
plt.figure(figsize = (10, 6))
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

 
plt.rcParams['xtick.major.size'] = 10   
plt.rcParams['xtick.major.width'] = 1.5  
plt.rcParams['ytick.major.size'] = 10   
plt.rcParams['ytick.major.width'] = 1.5

plt.plot(n_clusters_range, inertia_per_k, marker = 'o')
plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], colors = 'r', linestyles = '--', 
           label = f'Optimal k = {optimal_k}')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Lowest Within-Cluster Sum of Squares (Inertia)')
plt.title('Elbow Method with Curvature Maximization')
plt.grid(True)
plt.show()

####################################################################################
# Plot silhouette score vs. number of clusters
plt.figure(figsize = (10, 6))
plt.plot(n_clusters_range, silhouette_scores, marker = 'o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Each K')
plt.grid(True)
plt.show()


# =================================================================================
# Step 4: Re-run clustering with the optimal number of clusters
# =================================================================================
print(f'The optimal number of clusters based on elbow method is: {optimal_k}')

final_kmeans = KMeans(n_clusters = optimal_k, random_state = 42, n_init = 'auto')
final_kmeans.fit(z_scored_data_reshaped)

caps = final_kmeans.cluster_centers_  # clustering centroid
np.save(path_output + f"CAPmaps_Centroid_k_{optimal_k}.npy", caps)


# =================================================================================
# Step 5: fitting group-level prototype templates back into subjects
# =================================================================================
sub_labels = []
for i in range(n_subjects):
    temp = final_kmeans.predict(z_scored_data[:,:,i])
    sub_labels.append(temp)
    
np.save(path_output + f"BackFitting_Labels_k_{optimal_k}.npy", sub_labels)


# =================================================================================
# Step 6: statistics calculation
# =================================================================================
unique_states = np.arange(optimal_k)

def dynamic_analysis(sequence,unique_states): 
    total_len = len(sequence)
    state_counts = Counter(sequence)
    
    ####################################################################################
    # occurrence rates (i.e., the proportion of time spent in each of the four CAP states)
    state_proportion = np.array([state_counts[state] / total_len for state in unique_states])
    
    
    ####################################################################################
    # transition probability (i.e., the likelihood of moving from one CAP state to another across consecutive time points)
    num_states = len(unique_states)
    transition_matrix = np.zeros((num_states, num_states))
    
    for i in range(0,total_len-1):
        if sequence[i] != sequence[i+1]:
            transition_matrix[sequence[i],sequence[i+1]] += 1

    # Normalize each row by the total number of outgoing state changes
    # Rows without outgoing transitions are assigned zero probabilities
    trans_prob =  (np.diag(1 /  (np.sum(transition_matrix,axis=1)))).dot(transition_matrix)
    trans_prob[np.isnan(trans_prob)] = 0

    return state_proportion, trans_prob


# ####################################################################################
sub_state_proportion, sub_trans_prob = [], []

for i in range(n_subjects):
    temp_state_proportion, temp_trans = dynamic_analysis(sub_labels[i], unique_states)

    sub_state_proportion.append(temp_state_proportion)
    sub_trans_prob.append(temp_trans)


np.save(path_output + 'Occurrence.npy', sub_state_proportion)
np.save(path_output + 'Transition.npy', sub_trans_prob)