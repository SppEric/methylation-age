# Install libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import pickle

# Define function to perform KDE on data given age and beta vectors
def smooth_scatter(age_v, beta_v, filename, marker_size=0):
    # Normalize age vector
    ageT_v = (age_v - np.min(age_v)) / (np.max(age_v) - np.min(age_v))

    # Make KDE
    data = np.vstack([ageT_v, beta_v])
    kernel = gaussian_kde(data)
    x1 = np.linspace(-0.1, 1.1, 500)
    x2 = np.linspace(-0.1, 1.1, 500)
    X1, X2 = np.meshgrid(x1, x2)
    Z = kernel(np.vstack([X1.ravel(), X2.ravel()])).reshape(X1.shape)

    # Normalize the density estimates
    z_m = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

    # Define color breaks and color map
    breaks_v = np.concatenate(([-0.1, 0.00001, 0.01], np.arange(0.1, 1.1, 0.1))) # Add parameters?
    color_map = plt.cm.colors.ListedColormap(['white'] + [plt.cm.Greens(i) for i in np.linspace(0.2, 1, len(breaks_v) - 2)])

    # Create the plot
    fig = plt.figure(figsize=(4, 3))
    plt.imshow(z_m, extent=(-0.1, 1.1, -0.1, 1.1), cmap=color_map, origin='lower', aspect='auto', vmin=0, vmax=1)
    plt.plot(ageT_v, beta_v, 'k.', markersize=marker_size)
    plt.axis('off')

    # Save results
    output = open(f"{filename}.pkl", 'wb')
    pickle.dump(fig, output)
    #plt.savefig(f"{filename}.jpg")

    # Perform cleanup
    plt.close()
    output.close()

# # Load data
# age_v = np.random.uniform(0, 100, 1000)  # Temporary data
# beta_v = np.random.uniform(0, 1, 1000)   # Temporary data

