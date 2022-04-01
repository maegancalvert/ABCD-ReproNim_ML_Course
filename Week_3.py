from nilearn import datasets as nidataset
from nilearn import plotting
from nilearn.image import get_data, image
from nilearn.decomposition import CanICA
from nilearn.plotting import plot_prob_atlas
from nilearn.regions import Parcellations
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiLabelsMasker
from nilearn.regions import RegionExtractor
from matplotlib import patches, ticker
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from sklearn import datasets as skdataset
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import r2_score
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import os

### Week 3 Problem Set ###
### First they measured the overlap between the subspaces (using e which is spanned by the maps) and then measured the one to one ###
### reprocibility of maps (using t - the normalized trace of the reordered cross-correlation matrix). ###
### They concluded that overall their results suggest reproducibility - although results can be affected ###
### by the number of components and volumes. ###


rest_dataset = nidataset.fetch_development_fmri(n_subjects=10, data_dir = 'dev_data', age_group = 'child')
func_filenames, confounds = rest_dataset.func, rest_dataset.confounds

#All the scans we downloaded
for filename in func_filenames:
    print(filename)

#Obtaining the path and the visualization for the first of these nifti files
print('First functional nifti image (4D) is at the path: %s' %rest_dataset.func[0])  # 4D data

first_rsn = image.index_img(rest_dataset.func[0], 0)
print(first_rsn.shape)
# plotting.plot_stat_map(first_rsn)

canica = CanICA(n_components=20,
                memory="nilearn_cache", memory_level=2,
                verbose=10,
                mask_strategy='whole-brain-template',
                random_state=0)
print('Fitting canICA to subjects')
canica.fit(func_filenames)
#
# # Retrieve the independent components in brain space. Directly
# # accessible through attribute `components_img_`.
#
canica_components_img = canica.components_img_
# plot_prob_atlas(canica_components_img)

# #components_img is a Nifti Image object, and can be saved to a file with
# #the following line:
# #canica_components_img.to_filename('canica_resting_state.nii.gz')
#
# ## Problem 2: Clustering methods to learn a brain parcellation from fMRI ###
#
#
dataset = nidataset.fetch_development_fmri(n_subjects=1)
#
# # print basic information on the dataset
# print('First subject functional nifti image (4D) is at path: %s' %dataset.func[0])
#
startw = time.time()

ward = Parcellations(method='ward', n_parcels=1000,
                     standardize=False, smoothing_fwhm=2.,
                     memory='nilearn_cache', memory_level=1,
                     verbose=1)
# Call fit on functional dataset: single subject (less samples).
ward.fit(dataset.func)
print("Ward agglomeration 1000 clusters: %.2fs" % (time.time() - startw))
#
startk = time.time()
kmeans = Parcellations(method='kmeans', n_parcels=50,
                       standardize=True, smoothing_fwhm=10.,
                       memory='nilearn_cache', memory_level=1,
                       verbose=1)
# Call fit on functional dataset: single subject (less samples)
kmeans.fit(dataset.func)
print("KMeans clusters: %.2fs" % (time.time() - startk))

ward_labels_img = ward.labels_img_
kmeans_labels_img = kmeans.labels_img_
#
# first_plot = plotting.plot_roi(ward_labels_img, title="Ward parcellation", display_mode='xz')
# display = plotting.plot_roi(kmeans_labels_img, title="KMeans parcellation", display_mode='xz')
# plotting.show()
# display.close()


### Problem 3: Manifold Learning methods ###
# n_points = 1000
# X, color = skdataset.make_s_curve(n_points, random_state=0)
# n_neighbors = 10
# n_components = 2

# Create figure
# fig = plt.figure(figsize=(90, 48))

# Add 3d scatter plot
# ax = fig.add_subplot(251, projection="3d")
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
# ax.view_init(4, -72)
# ax.grid('True')

### Manifold learning is an attempt to help reduction frameworks to be sensitive to ###
### non-linear structure in the data. It is unsupervised as it learns the high-dimensional ###
### structure of the data from the data itself. Can use isometric mapping reveals nonlinear ###
### degrees of freedom. Isomapping has been used in Parkinson's and Alzheimer's research distinguishing ###
### disease from healthy. Can use Locally Linear Embedding which reduces features and preserves the geometric ###
### features of the original dataset. Can use t-Distributed Stochastic Neighbor Embedding. ###
### t-SNE is mapped to low dimensions while preserving the structure of the data. Manifold learning###
### can be useful in determining brain morphology and diffusion imaging. ###

### ISO map ###

# t0 = time.time()
# trans_iso = (manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components).fit_transform(X))
# t1 = time.time()
# print("%s: %.2g sec" % ("ISO", t1 - t0))
# print(trans_iso.shape)
#
# ax1 = fig.add_subplot(252, projection="3d")
# ax1.scatter(trans_iso[:, 0], trans_iso[:, 1], c=color, cmap=plt.cm.Spectral)
# ax1.grid('True')
#
# ### Locally Linear Embedding ###
# t2 = time.time()
# trans_lle = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components)
# lle = trans_lle.fit_transform(X)
# t3 = time.time()
# print("%s: %.2g sec" % ("LLE", t3 - t2))
# print(lle.shape)

# fig = plt.figure(figsize=(45, 24))
# ax2 = fig.add_subplot(253, projection="3d")
# ax2.scatter(lle[:, 0], lle[:, 1], c=color, cmap=plt.cm.Spectral)
# ax2.grid('True')
#
# # ### t-distributed stochastic neighbor embedding ###
#
# t4 = time.time()
# tsne = manifold.TSNE(n_components=n_components, init="pca", random_state=0)
# trans_tsne = tsne.fit_transform(X)
# t5 = time.time()
# print("t-SNE: %.2g sec" % (t5 - t4))
# print(trans_tsne.shape)

# fig = plt.figure(figsize=(45, 24))
# ax3 = fig.add_subplot(254, projection="3d")
# ax3.scatter(trans_tsne[:, 0], trans_tsne[:, 1], c=color, cmap=plt.cm.Spectral)
# ax3.grid('True')
# plt.show()
# plt.close(fig)

### Graphical Models ###

extractor = RegionExtractor(canica_components_img,
                            threshold=0.5,
                            thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions',
                            standardize=True,
                            min_region_size=1350)

print(f'Fitting the Extactor...')

# Fit the extractor
extractor.fit()

# Extracted regions are stored in regions_img_
regions_extracted_img = extractor.regions_img_

# Total number of regions extracted
n_regions_extracted = regions_extracted_img.shape[-1]

print(f'{n_regions_extracted} Regions extracted...')

# Store sub (key), and adjacency matrix (val)
mat_dict = {}

# ConnectivityMeasure
connectome_measure = ConnectivityMeasure(kind='correlation')
func_filenames, confounds = rest_dataset.func, rest_dataset.confounds

# Loop through each subject
for filename, confound in zip(func_filenames, confounds):
    # Get cleaner version of subject name
    sub = filename.split(os.path.sep)[-1]
    sub = sub.split('_')[0]

    print(f'Creating correlation matrix for: {sub}')

    # Extract timeseries signals from our extractor
    timeseries_each_subject = extractor.transform(filename, confounds=confound)

    # Create our Connectivity Measure from this extracted time series
    mat = connectome_measure.fit_transform([timeseries_each_subject])
    mat = mat.squeeze(axis=0)

    # Add to our dict
    mat_dict[sub] = mat

print(mat)
### Plot one of the correlation matrices ###
sub1_corr = plt.imshow(mat_dict['sub-pixar001'], cmap='coolwarm')
# plt.show()

print(mat_dict.keys())
num_subs = len(mat_dict.keys())
X, y = mat.shape
num_feats = int(((X*y)-X)/2)
mat_full = np.zeros((num_subs, num_feats))

### Generate an empty matrix to store this data ###
for ind, (sub, mat) in enumerate(mat_dict.items()):
    mat_full[ind, :] = mat[np.triu_indices(mat.shape[0], k=1)]

print(mat.shape)
print(mat_full.shape)

fig, ax = plt.subplots(figsize = (8,4))
ax.imshow(mat_full, aspect = 'auto', cmap = 'coolwarm')
# plt.show()

# Lets use our raw connectivity as our data, and age is our target
X = mat_full

phen_data = rest_dataset['phenotypic']
Y = np.array([row[1] for row in phen_data])
print(f'X shape: {X.shape} Y shape: {Y.shape}')

# Create the test train split
X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=0.33, random_state=42)

# Create and fit the model
model = Ridge()
model.fit(X_tr, y_tr)

# Predict on our training data and plot (actual vs pred) and print the score, mse
pred = model.predict(X_tr)
plt.scatter(y_tr, pred.squeeze())
print(f'Model Score: {r2_score(y_tr, pred)}')
plt.show()

# Predict on our training data and plot (actual vs pred) and print the score, mse
test_pred = model.predict(X_te)
plt.scatter(y_te, test_pred.squeeze())
print(f'Model Score: {r2_score(y_te, test_pred)}')
plt.show()

### Why might we be struggling to generalize to new data? ###
# we didn't tune the hyperparameters #
# there might be a non-linear relationship between age and functional connectivity #
# There may be a relationship between resting state functional connectivity and  #
# age but it is not captured within our data for many possible reasons (artifact, small sample size, batch effects, etc)#

graphs_dict = {}

for key, val in mat_dict.items():
    # Thresholding
    val=np.abs(val)
    val[val > .2] = 1
    val[val < 1] = 0
    g = nx.from_numpy_matrix(val)
    graphs_dict[key] = g
pos = nx.kamada_kawai_layout(g)
nx.draw(g, pos=pos)

# ### Store these in a dict ###
gMetric_dict = {'y': Y,
               'clust_co': [],
               'efficiency': [],
               'degree': []}

for sub, g in graphs_dict.items():
    gMetric_dict['clust_co'].append(nx.average_clustering(g))
    gMetric_dict['efficiency'].append(nx.global_efficiency(g))
    gMetric_dict['degree'].append(np.mean(nx.degree(g)))

fig, ax = plt.subplots(1, 3, figsize = (12, 4))

ax[0].scatter(gMetric_dict['efficiency'], gMetric_dict['clust_co'])
ax[1].scatter(gMetric_dict['efficiency'], gMetric_dict['degree'])
ax[2].scatter(gMetric_dict['degree'], gMetric_dict['clust_co'])
plt.show()

### Efficiency is how efficient nodes exchange information ###
### We would expect high efficiency among nodes that are close to each other ###
### or to be part of the primary cortex ###
### We would expect nodes of one network to have higher clustering coefficients ###
### with another node of the same network than a node of another network ###
### 