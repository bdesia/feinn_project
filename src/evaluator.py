import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

class SolutionComparator:
    """
    Generalized comparator for N-component solutions (e.g., displacements, stresses)
    on a 2D finite element mesh.
    Expects interleaved vectors: [c1_0, c2_0, ..., cN_0, c1_1, c2_1, ...].
    """
    
    def __init__(self, mesh, u_true, u_pred, labels=['ux', 'uy']):
        """
        Initialize with mesh, full solution vectors, and component labels.
        """
        self.mesh = mesh
        self.labels = labels
        self.n_comp = len(labels)
        
        u_t = self._to_numpy(u_true).squeeze()
        u_p = self._to_numpy(u_pred).squeeze()

        # Validate input dimensions
        if u_t.shape != u_p.shape:
            raise ValueError("True and Pred vectors must have the same size.")
        if u_t.shape[0] != self.mesh.nnod * self.n_comp:
            raise ValueError(f"Vector size ({u_t.shape[0]}) doesn't match nnod * {self.n_comp} components.")

        self.tol = 1e-6
        self.comp_data = {}

        # Dynamically slice and store each component
        for i, label in enumerate(self.labels):
            y_t = u_t[i::self.n_comp]
            y_p = u_p[i::self.n_comp]
            
            # Precompute local error arrays for each component
            mask = np.abs(y_t) > self.tol
            perc_err = np.zeros_like(y_t)
            perc_err[mask] = 100 * np.abs((y_p[mask] - y_t[mask]) / y_t[mask])
            
            self.comp_data[label] = {
                'true': y_t,
                'pred': y_p,
                'errors': {
                    'difference': y_p - y_t,
                    'absolute_error': np.abs(y_p - y_t),
                    'percentage_error': perc_err
                }
            }

    @staticmethod
    def _to_numpy(data):
        """Converts potential PyTorch tensors to numpy arrays."""
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return np.array(data)

    def _create_trimesh(self):
        """
        Creates a Matplotlib Triangulation object. 
        Converts 1-based element connectivity to 0-based indexing.
        """
        pos = self.mesh.coordinates
        triangles = []
        
        if 'quad' in self.mesh.elements:
            for e in self.mesh.elements['quad']:
                v = np.array(e) - 1
                # Split quad into two triangles
                triangles.extend([[v[0], v[1], v[2]], [v[2], v[3], v[0]]])
                
        if 'tri' in self.mesh.elements:
            for e in self.mesh.elements['tri']:
                triangles.append([e[0]-1, e[1]-1, e[2]-1])
                
        return Triangulation(pos[:, 0], pos[:, 1], triangles)

    def _contour_plot(self, ax, val, title, cmap, levels, vmin, vmax, cbar, row_label=None):
        """Internal helper to plot a single 2D contour."""
        tri = self._create_trimesh()
        cnt = ax.tricontourf(tri, val, cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
        
        if cbar: 
            plt.colorbar(cnt, ax=ax)
            
        if title: 
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            
        # Display row label on the leftmost column, hide y-ticks for others
        if row_label:
            ax.set_ylabel(row_label, fontsize=16, fontweight='bold', labelpad=15)
        else:
            ax.set_yticks([]) 
            
        ax.set_xlabel('X')
        ax.set_aspect('equal')
        return cnt

    def plot_comparison(self, metric='difference', cmap='viridis', levels=50):
        """
        Plots a clean grid layout: rows for components, columns for GT, Pred, and Metric.
        """
        fig, axs = plt.subplots(self.n_comp, 3, figsize=(15, 4.5 * self.n_comp), constrained_layout=True)
        
        # Ensure axs is a 2D array even if there's only one component
        if self.n_comp == 1: 
            axs = np.expand_dims(axs, axis=0)

        metric_title = metric.replace('_', ' ').title()
        col_titles = ['Ground Truth', 'Prediction', metric_title]

        for i, label in enumerate(self.labels):
            d = self.comp_data[label]
            y_t, y_p = d['true'], d['pred']
            err_val = d['errors'][metric]

            # Enforce common color scale for True vs Pred in the current row
            vmin, vmax = min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())

            # Only show column headers on the very first row
            t0 = col_titles[0] if i == 0 else None
            t1 = col_titles[1] if i == 0 else None
            t2 = col_titles[2] if i == 0 else None
            
            # Left column: Ground Truth (includes the Y-axis row label)
            self._contour_plot(axs[i,0], y_t, t0, cmap, levels, vmin, vmax, False, row_label=label)
            
            # Middle column: Prediction
            c2 = self._contour_plot(axs[i,1], y_p, t1, cmap, levels, vmin, vmax, False)
            
            # Shared colorbar for GT and Pred
            fig.colorbar(c2, ax=axs[i, 0:2], aspect=30, pad=0.02)

            # Right column: Metric Plot
            m_cmap = 'coolwarm' if metric == 'difference' else 'plasma'
            m_max = np.abs(err_val).max()
            m_vmin, m_vmax = (-m_max, m_max) if metric == 'difference' else (0, m_max)
            
            self._contour_plot(axs[i,2], err_val, t2, m_cmap, levels, m_vmin, m_vmax, True)

        plt.show()

    def plot_error_distribution(self, bins=50):
        """
        Plots histograms of the absolute error distribution for all components.
        """
        fig, axs = plt.subplots(1, self.n_comp, figsize=(7 * self.n_comp, 5))
        if self.n_comp == 1: axs = [axs]
        
        for i, label in enumerate(self.labels):
            error = self.comp_data[label]['errors']['absolute_error']
            mae = np.mean(error)
            
            axs[i].hist(error, bins=bins, color='coral', edgecolor='black')
            axs[i].set_yscale('log')
            axs[i].set_xlabel(f'Absolute Error ({label})')
            axs[i].set_ylabel('Node Count (Log Scale)')
            axs[i].set_title(f'Error Distribution ({label})')
            axs[i].axvline(mae, color='blue', linestyle='dashed', label=f'MAE: {mae:.2e}')
            axs[i].legend()
            
        plt.tight_layout()
        plt.show()

    def report(self):
        """
        Computes R2, Relative L2, RMSE, MAE, and MAPE metrics for all components.
        Returns a Pandas DataFrame.
        """
        rows = []
        for label in self.labels:
            yt, yp = self.comp_data[label]['true'], self.comp_data[label]['pred']
            
            r2 = r2_score(yt, yp)
            mae = mean_absolute_error(yt, yp)
            rmse = np.sqrt(mean_squared_error(yt, yp))
            l2_rel = np.linalg.norm(yp - yt) / (np.linalg.norm(yt) + 1e-10)
            
            mask = np.abs(yt) > self.tol
            mape = np.mean(np.abs((yp[mask] - yt[mask]) / yt[mask])) * 100 if np.any(mask) else 0.0
            
            rows.append({
                'Component': label, 
                'R2': r2, 
                'L2_Rel': l2_rel, 
                'RMSE': rmse, 
                'MAE': mae, 
                'MAPE%': mape
            })
            
        return pd.DataFrame(rows).set_index('Component')

    def export_to_vtk(self, filename="results.vtu"):
        """
        Exports solution fields to VTK/VTU format for ParaView.
        Groups up to 3 components into 3D vectors for native Warp functionality.
        """
        try:
            import meshio
        except ImportError:
            raise ImportError("Please install meshio to export to VTK: pip install meshio")
            
        pts = np.column_stack((self.mesh.coordinates, np.zeros(self.mesh.nnod)))
        
        # Prepare connectivity arrays
        cls = []
        if 'quad' in self.mesh.elements:
            cls.append(("quad", np.array(self.mesh.elements['quad']) - 1))
        if 'tri' in self.mesh.elements:
            cls.append(("triangle", np.array(self.mesh.elements['tri']) - 1))
            
        def build_vector(data_key, sub_key=None):
            """Helper to pack components into an (N, 3) vector array."""
            vec = np.zeros((self.mesh.nnod, 3))
            # Safely map up to 3 components to X, Y, Z channels
            for i, label in enumerate(self.labels[:3]):
                if sub_key:
                    vec[:, i] = self.comp_data[label][data_key][sub_key]
                else:
                    vec[:, i] = self.comp_data[label][data_key]
            return vec

        # Define the point data fields
        pdata = {
            "Ground_Truth": build_vector('true'),
            "Prediction": build_vector('pred'),
            "Absolute_Error": build_vector('errors', 'absolute_error'),
            "Difference": build_vector('errors', 'difference')
        }
            
        meshio.Mesh(pts, cls, point_data=pdata).write(filename)
        print(f"Exported Vector fields successfully to {filename}. Open it in ParaView!")