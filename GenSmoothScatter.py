# Install libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import pickle
from multiprocessing import Pool, cpu_count

# Define function to perform KDE on data given age and beta vectors
def smooth_scatter(age_v, beta_v, filename, marker_size=1, bandwidth=0.1, outlier_bandwidth=2, threshold=4):
    # NOTE: Generated images are 400 x 300
    # Normalize age vector
    ageT_v = (age_v - np.min(age_v)) / (np.max(age_v) - np.min(age_v))

    # Make KDE
    data = np.vstack([ageT_v, beta_v])
    kernel = gaussian_kde(data, bw_method=bandwidth)
    x1 = np.linspace(-0.1, 1.1, 1000)
    x2 = np.linspace(-0.1, 1.1, 1000)
    X1, X2 = np.meshgrid(x1, x2)
    Z = kernel(np.vstack([X1.ravel(), X2.ravel()])).reshape(X1.shape)

    # Normalize the density estimates
    z_m = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

    # Define color breaks and color map
    breaks_v = np.concatenate(([-1, 0.0001, 1], np.arange(0.1, 10.1, 0.1))) # Add parameters?
    color_map = plt.cm.colors.ListedColormap(['white'] + [plt.cm.Greens(i) for i in np.linspace(0.2, 1, len(breaks_v) - 2)])

    # Create the plot
    fig = plt.figure(figsize=(4, 3))
    plt.imshow(Z, extent=(-0.1, 1.1, -0.1, 1.1), cmap=plt.cm.Greens, origin='lower', aspect='auto', vmin=0, vmax=1)
    plt.plot(ageT_v, beta_v, 'k.', markersize=marker_size)
    plt.axis('off')
    
    # # Debugging
    # plt.show()

    # Save results
    output = open(f"{filename}.pkl", 'wb') # NOTE: We are pickling the pyplot graphs, not the generated jpg images!
    pickle.dump(fig, output)

    # Perform cleanup
    plt.close()
    output.close()

# # Load data
# age_v = np.random.uniform(0, 100, 1000)  # Temporary data
# beta_v = np.random.uniform(0, 1, 1000)   # Temporary data

def compute_density_2(args):
    """Helper function to compute KDE for a batch."""
    data, bandwidth, x1_values, x2_values = args
    kernel = gaussian_kde(data, bw_method=bandwidth)  # Initialize kernel inside subprocess
    points = np.vstack([x1_values, x2_values])
    return kernel(points)

def smooth_scatter_2(age_v, beta_v, filename, marker_size=0, bandwidth=0.1):
    # Normalize age vector
    ageT_v = (age_v - np.min(age_v)) / (np.max(age_v) - np.min(age_v))
    
    # Prepare data for KDE
    data = np.vstack([ageT_v, beta_v]).astype(np.float32)  # Use float32 for efficiency
    
    # Create grid
    x1 = np.linspace(-0.1, 1.1, 1000)
    x2 = np.linspace(-0.1, 1.1, 1000)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Prepare for multiprocessing
    num_cores = cpu_count()
    x1_batches = np.array_split(X1.ravel(), num_cores)
    x2_batches = np.array_split(X2.ravel(), num_cores)
    args = [(data, bandwidth, x1_batches[i], x2_batches[i]) for i in range(num_cores)]
    
    with Pool(processes=num_cores) as pool:
        results = pool.map(compute_density_2, args)
    
    # Combine results
    Z = np.concatenate(results).reshape(X1.shape)
    
    # Normalize the density estimates
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    
    # Define color breaks and color map
    breaks_v = np.concatenate(([-1, 0.0001, 1], np.arange(0.1, 10.1, 0.1)))
    color_map = plt.cm.colors.ListedColormap(['white'] + [plt.cm.Greens(i) for i in np.linspace(0.2, 1, len(breaks_v) - 2)])
    
    # Create the plot
    fig = plt.figure(figsize=(4, 3))
    plt.imshow(Z, extent=(-0.1, 1.1, -0.1, 1.1), cmap=plt.cm.Greens, origin='lower', aspect='auto', vmin=0, vmax=1)
    plt.plot(ageT_v, beta_v, 'k.', markersize=marker_size)
    plt.axis('off')
    
    # Save results as .pkl
    with open(f"{filename}.pkl", 'wb') as output:
        pickle.dump(fig, output)
    
    # Perform cleanup
    plt.close(fig)