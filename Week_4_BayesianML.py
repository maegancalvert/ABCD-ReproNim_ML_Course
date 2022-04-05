import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.stats import norm
from scipy.stats import norm, multivariate_normal
import seaborn as sns


# a = 40
# b = 100
#
# m = (40+100)/2
# print(m)
#
# a, b = 40, 100
# dist = np.random.randint(low=a, high=b, size=200)
# print(dist)
#
# mean = np.mean(dist)
# std = np.std([dist])
# std1 = np.sqrt(np.sum(np.subtract(dist, mean)**2)/len(dist))
# print(mean, std, std1)
#
# samples = []
# for num in range(1000):
#     x = (a+(b-a) * random.random())
#     samples.append(x)
# samples_arr = np.asarray(samples)
# mean2 = sum(samples)/len(samples)
# std2 = np.sqrt(np.sum(np.subtract(samples_arr, mean2)**2)/len(samples))
#
# plt.hist(samples_arr)
# plt.show()

### The mean and standard deviation center around mean = 70 and std = 17 ###
### At 200 subjects the mean is 69.53 and std 16.46 ###
### At 1000 subjects the mean is  and the std  ###
### At 5000 subjects the mean is 69.75 and std 17.21 ###

### Problem 2 ###
# mu = 0
# std = 1
#
# # Create the distribution
# dist = norm(loc = mu, scale=std)
#
# # Range over which we access the pdf/cdf
# x = np.linspace(-5, 5, 500)
#
# # Obtain the pdf over this range of x
# pdf = dist.pdf(x)
# print(pdf)
# # Obtain the cdf over this range of x
# cdf = dist.cdf(x)
# print(cdf)
# # Plot this
# fig = plt.figure(figsize = (10, 8))
#
# prob1 = dist.pdf(0.003)
# prob2 = dist.cdf(-2)
# prob3 = 1-dist.cdf(1.25)
#
# # Answers to the problems
# print(f'Prob 1: {prob1:.3f} Prob 2: {prob2:.3f} Prob 3: {prob3:.3f}')
#
# #Plot this
# fig = plt.figure(figsize = (10, 8))
# plt.plot(x, pdf, color = 'green', alpha = .4, linewidth = 5, label = 'PDF')
# plt.plot(x, cdf, color = 'green', alpha = .4, linewidth = 2, linestyle = '--', label = 'CDF')
# plt.vlines(x = .003, ymin = 0, ymax = 1, linewidth = 3, linestyle = '--', color = 'k', label = 'Prob 1')
# plt.vlines(x = -2, ymin = 0, ymax = 1, linewidth = 3, linestyle = '--', color = 'r', label = 'Prob 2')
# plt.vlines(x = 1.25, ymin = 0, ymax = 1, linewidth = 3, linestyle = '--', color = 'r', label = 'Prob 3')
# plt.legend()
# plt.show()
#
# ### Problem 3 ###
# def genMesh(lb, ub, n):
#     # Two linspace vectors
#     x = np.linspace(lb, ub, n)
#     y = np.linspace(lb, ub, n)
#
#     # These will be our X and Y coordinates for plotting
#     X, Y = np.meshgrid(x, y)
#
#     # This is the matrix over which we generate our PDF
#     pos = np.zeros((n, n, 2))
#     pos[:, :, 0] = X;
#     pos[:, :, 1] = Y
#
#     return X, Y, pos
#
#
# def plot_multivariate_normal(X, Y, Z, lb=-5, ub=5, cmap='coolwarm'):
#     # Instantiate plot
#     fig = plt.figure(figsize=(20, 12))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Surface Plot
#     ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.25, cmap=cmap)
#
#     # Contour plots for density, X and Y
#     ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z), alpha=.4, cmap=cmap)
#     ax.contourf(X, Y, Z, zdir='x', offset=np.min(X), cmap='binary', levels=1)
#     ax.contourf(X, Y, Z, zdir='y', offset=np.max(Y), cmap='binary', levels=1)
#
#     # Fix labels
#     ax.grid(False)
#     ax.set_xlabel('Distr(X)');
#     ax.set_xlim(lb, ub)
#     ax.set_ylabel('Distr(Y)');
#     ax.set_ylim(lb, ub)
#     ax.set_zlim(np.min(Z), np.max(Z))
#
#     # ax.set_title('Multivariate Gaussian')
#     ax.view_init(25, -25)
#     plt.show()
#
# ### Create two independent Gaussian RVs ###
# # Generate our X, Y, pos
# X, Y, pos = genMesh(-5, 5, 1000)
#
# # Create our array of mu
# mus = np.array([0.0, 1.0])
#
# # Create our covariance matrix for INDEPENDENT gaussian rvs
# cov = np.array([[1.00, 0], [0, 1.00]])
#
# # Create our multivariate normal
# rv = multivariate_normal(mus, cov)
# Z = rv.pdf(pos)
#
# # Plot this
# plot_multivariate_normal(X, Y, Z)
#
# ### Create two correlated Gaussian RVs ###
# # Generate our X, Y, pos
# X, Y, pos = genMesh(-5, 5, 1000)
#
# # Create our array of mu
# mus = np.array([0.0, 1.0])
#
# # Create our covariance matrix for CORRELATED gaussian rvs
# cov = np.array([[1.00, 0.50], [0.50, 1.00]])
#
# # Create our multivariate normal
# rv = multivariate_normal(mus, cov)
# Z = rv.pdf(pos)
#
# # Plot this
# plot_multivariate_normal(X, Y, Z)

### Problem 4 ###
def plot_hypothesis(x, p0, p1, pdf_py_H0, pdf_py_H1, cdf_py_H0, cdf_py_H1, intersect_x, intersect_y):

    fig, ax = plt.subplots(figsize = (10, 8))

    # Plot PDFs of H0/H1
    ax.plot(x, pdf_py_H0, color = 'purple', alpha = .4, linewidth = 5, label = 'PDF H0')
    ax.plot(x, pdf_py_H1, color = 'green', alpha = .4, linewidth = 5, label = 'PDF H1')

    # Plot CDFs of H0/H1
    ax.plot(x, cdf_py_H0, color = 'purple', alpha = .4, linewidth = 2, linestyle = '--', label = 'CDF H0')
    ax.plot(x, cdf_py_H1, color = 'green', alpha = .4, linewidth = 2, linestyle = '--', label = 'CDF H1')

    # Plot intercept
    ax.vlines(x = intersect_x, ymin = 0, ymax = 1, linewidth = 3, linestyle = '--', color = 'k')
    ax.scatter(intersect_x, intersect_y, s = 75, color = 'blue', )
    ax.legend()
    ax.set_ylim(0, 1)

    plt.show()

# ### Fill in the appropriate parts to correctly return sensitivity and specificity ###
def get_sens_spec(x, p0, p1, mu1, mu2, std1, std2, plot=True):
    # Create distributions of Y given H0/H1
    py_H0 = norm(mu1, std1)
    py_H1 = norm(mu2, std2)

    # Create the pdfs
    pdf_py_H0 = py_H0.pdf(x) * p0
    pdf_py_H1 = py_H1.pdf(x) * p1

    # Create the cdfs
    cdf_py_H0 = py_H0.cdf(x) * p0
    cdf_py_H1 = py_H1.cdf(x) * p1

    # This fancy number will find us the intersectin of our two PDFs
    # credit: https://stackoverflow.com/questions/28766692/intersection-of-two-graphs-in-python-find-the-x-value
    intersect_ind = np.argwhere(np.diff(np.sign(pdf_py_H0 - pdf_py_H1))).flatten()
    print(intersect_ind.shape)

    #Get the intersections by ind
    intersect_x = x[intersect_ind][0]
    intersect_y = py_H0.pdf(intersect_x) * p0

    if plot:
        plot_hypothesis(x, p0, p1, pdf_py_H0, pdf_py_H1, cdf_py_H0, cdf_py_H1, intersect_x, intersect_y)

    # Return sensitivity, specificity
    TP = 1 - (py_H1.cdf(intersect_x))
    TN = py_H0.cdf(intersect_x)
    FP = 1 - (py_H0.cdf(intersect_x))
    FN = py_H1.cdf(intersect_x)
    sens = TP / (TP + FN)
    spec = TN / (TN + FP)

    print(f'Sens: {sens:.3f} Spec: {spec:.3f}\n')

    return spec, sens

### H0 and H1 are equally likely ###
# Range of x
x = np.linspace(-5, 5, 500)

# The priors of H0/H1
p0, p1 = .5, .5

# Mean, std for H0/H1
mu1, mu2 = 0, 2
std1, std2 = 1, 1

sens1, spec1 = get_sens_spec(x, p0, p1, mu1, mu2, std1, std2)

### H0 is more likely ###
x = np.linspace(-5, 5, 500)

# The priors of H0/H1
p0, p1 = .3, .7

# Mean, std for H0/H1
mu1, mu2 = 0, 2
std1, std2 = 1, 1

sens2, spec2 = get_sens_spec(x, p0, p1, mu1, mu2, std1, std2)

### H1 is more likely ###
# Range of x
x = np.linspace(-5, 5, 500)

# The priors of H0/H1
p0, p1 = .7, .3

# Mean, std for H0/H1
mu1, mu2 = 0, 2
std1, std2 = 1, 1

sens3, spec3 = get_sens_spec(x, p0, p1, mu1, mu2, std1, std2)

### H0 and H1 are equally likely and means are more distant ###
# Range of x
x = np.linspace(-5, 5, 500)

# The priors of H0/H1
p0, p1 = 0.5, 0.5

# Mean, std for H0/H1
mu1, mu2 = 0, 3.5
std1, std2 = 1, 1

sens4, spec4 = get_sens_spec(x, p0, p1, mu1, mu2, std1, std2)

### ROC curve ###
fig, ax = plt.subplots(figsize = (6, 4))

# Setup lines
ax.plot(np.arange(0, 1, .01), np.arange(0, 1, .01), linewidth = 2, linestyle = '--', color = 'k')
ax.vlines(x = 1, ymin = 0, ymax = 1, linewidth = 2, linestyle = '--', color = 'k')
ax.hlines(y = 1, xmin = 0, xmax = 1, linewidth = 2, linestyle = '--', color = 'k')

# Plot
ax.scatter(1-spec1, sens1, s = 150, color = 'red', label = 'Prob 1')
ax.scatter(1-spec2, sens2, s = 150, color = 'blue', label = 'Prob 2')
ax.scatter(1-spec3, sens3, s = 150, color = 'orange', label = 'Prob 3')
ax.scatter(1-spec4, sens4, s = 150, color = 'green', label = 'Prob 4')

# Changes
ax.set_title('ROC Curve')
ax.set_xlabel('1 - Specificity')
ax.set_ylabel('Sensitivity')
ax.legend(loc = 'lower right')
plt.show()

### The false positive rates are low in general ###
### When the probability of H0 is the same as H1, but the means are further apart ###
### there is higher sensivity and lower false positive rate ###
### When the probability of H0 is more likely the false positive rate is higher ###


### LMMSE Estimator ###
# Our range of x
n = np.arange(-4, 6, .01)

# First distribution (prior) and pdf
normal1 = norm(0, 1)
f_y = normal1.pdf(n)

# Second distribution and pdf
Y = 2
normal2 = norm(Y, 1)
f_xy = normal2.pdf(n)

# Plot these
fig = plt.figure(figsize = (20, 8))

plt.plot(n, f_y, label = 'f_X(x)', linewidth = 4, color = 'purple')
plt.plot(n, f_xy, label = 'f_Xy(x)', linewidth = 4, color = 'green')
plt.stem([Y], [normal2.pdf(Y)], linefmt = 'red')
plt.legend()

# Reinstantiate plot
fig = plt.figure(figsize=(20, 8))

plt.plot(n, f_y, label='f_X(x)', linewidth=4, color='purple')
plt.plot(n, f_xy, label='f_Xy(x)', linewidth=4, color='green')
plt.stem([Y], [normal2.pdf(Y)])
plt.legend()

# Number of samples we will add
N = 100
X = Y + np.random.normal(size=N)
X[0] = np.exp(1)

# Our estimator
Yest = np.zeros(X.shape)

# Loop through our number of samples
for i in range(N):
    C_XY = 1 + np.eye(i + 1)
    C_xy = np.ones((i + 1, 1))
    a = np.linalg.inv(C_XY) @ C_xy

    z = a.T @ X[:i + 1]
    Yest[i] = z[0]

    plt.stem([Yest[i]], [normal2.pdf(Yest[i])], linefmt='red')

plt.show()

