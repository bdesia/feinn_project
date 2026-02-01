import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class SolutionComparator:
    """
    Class to compare two solutions from different methods on the same finite element mesh.
    
    Attributes:
        mesh: Mesh object with coordinates, elements, nnod.
        y_true (ndarray): 1D array with ground truth solution values at nodes.
        y_pred (ndarray): 1D array with predicted solution values at nodes.
        local_errors (dict): Local errors for plotting (difference, absolute error, percentage error).
    """
    
    def __init__(self, mesh, y_true, y_pred):
        """
        Initialize with mesh and solutions. Validates sizes.
        """
        self.mesh = mesh
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        
        # Validate sizes
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError("Reference and comparison solutions must have the same size.")
        if self.y_true.shape[0] != self.mesh.nnod:
            raise ValueError("Solution sizes do not match number of nodes in mesh.")
        
        # Compute local errors for plotting
        self.local_errors = {
            'difference': self.y_pred - self.y_true,
            'absolute_error': np.abs(self.y_pred - self.y_true),
            'percentage_error': np.where(self.y_true != 0, 
                                         100 * np.abs((self.y_pred - self.y_true) / self.y_true), 
                                         0)  # Avoid division by zero
        }
    
    def compute_r2(self):
        """Compute R² score."""
        return r2_score(self.y_true, self.y_pred)
    
    def compute_mae(self):
        """Compute Mean Absolute Error (MAE)."""
        return mean_absolute_error(self.y_true, self.y_pred)
    
    def compute_mape(self):
        """Compute Mean Absolute Percentage Error (MAPE), ignoring zeros in reference solution."""
        mask = self.y_true != 0
        if not np.any(mask):
            raise ValueError("All reference solution values are zero; MAPE cannot be computed.")
        return 100 * np.mean(np.abs((self.y_pred[mask] - self.y_true[mask]) / self.y_true[mask]))
    
    def compute_rmse(self):
        """Compute Root Mean Square Error (RMSE)."""
        return np.sqrt(np.mean((self.y_pred - self.y_true)**2))
    
    def _create_grid_and_interpolate(self, values):
        """Helper to create grid and interpolate values."""
        points = self.mesh.coordinates
        x = points[:, 0]
        y = points[:, 1]
        
        grid_x, grid_y = np.mgrid[
            x.min():x.max():100j,
            y.min():y.max():100j
        ]
        
        grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
        
        return grid_x, grid_y, grid_z
    
    def _contour_plot(self, ax, values, label, cmap, levels):
        """Helper to plot contour on given ax."""
        grid_x, grid_y, grid_z = self._create_grid_and_interpolate(values)
        
        contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap)
        fig = ax.get_figure()
        fig.colorbar(contour, ax=ax, label=label)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(label)
        ax.set_aspect('equal')
    
    def plot(self, metric='absolute_error', title=None, cmap='viridis', levels=50):
        """
        Plot 2D contour of point-wise metric on the mesh.
        
        Args:
            metric (str): 'difference', 'absolute_error', or 'percentage_error'.
            title (str, optional): Plot title.
            cmap (str): Matplotlib colormap (default: 'viridis').
            levels (int): Number of contour levels (default: 50).
        """
        if metric not in self.local_errors:
            raise ValueError(f"Metric '{metric}' not available. Options: {list(self.local_errors.keys())}")
        
        values = self.local_errors[metric]
        label = metric.replace('_', ' ').title()
        
        fig, ax = plt.subplots()
        self._contour_plot(ax, values, label, cmap, levels)
        
        if title:
            ax.set_title(title)
        
        plt.show()
    
    def plot_comparison(self, metric='absolute_error', title=None, cmap='viridis', levels=50):
        """
        Plot 3x1 subplots: y_true, y_pred, and the selected metric.
        
        Args:
            metric (str): 'difference', 'absolute_error', or 'percentage_error'.
            title (str, optional): Main figure title.
            cmap (str): Matplotlib colormap (default: 'viridis').
            levels (int): Number of contour levels (default: 50).
        """
        if metric not in self.local_errors:
            raise ValueError(f"Metric '{metric}' not available. Options: {list(self.local_errors.keys())}")
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot y_true
        self._contour_plot(axs[0], self.y_true, 'Ground Truth (y_true)', cmap, levels)
        
        # Plot y_pred
        self._contour_plot(axs[1], self.y_pred, 'Prediction (y_pred)', cmap, levels)
        
        # Plot metric
        values = self.local_errors[metric]
        label = metric.replace('_', ' ').title()
        self._contour_plot(axs[2], values, label, cmap, levels)
        
        if title:
            fig.suptitle(title)
        
        plt.tight_layout()
        plt.show()