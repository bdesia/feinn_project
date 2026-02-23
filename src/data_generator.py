import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import pdist, squareform
import os
from typing import List, Tuple

class StrainHistoryGenerator:
    """
    Generator of strain histories for synthetic RVEs data using Gaussian Processes.
    
    Features:
    - Generates smooth loading paths using Gaussian Processes (RBF kernel)
    - Ensures ε₁(0) = ε₂(0) = 0 at the begginigning of the history
    - Converts principal strains to Cartesian components (exx, eyy, γxy) if needed
    - Saves generated histories as CSV files
    """
    
    def __init__(self, 
                 n_steps: int = 50,
                 eps_max: float = 0.03,
                 theta_max: float = np.pi / 6,
                 seed: int | None = None):
        
        self.n_steps = n_steps
        self.eps_max = eps_max
        self.theta_max = theta_max
        self.rng = np.random.default_rng(seed)
        self.t = np.arange(n_steps)
    
    def rbf_kernel(self, X: np.ndarray, lengthscale: float) -> np.ndarray:
        """
        Radial basis function kernel 
        """
        sqdist = squareform(pdist(X[:, None], 'sqeuclidean'))
        return np.exp(-0.5 * sqdist / lengthscale**2)
    
    def generate_single_history(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        l_min = self.n_steps / 10
        l_max = self.n_steps / 2
        lengthscale = self.rng.uniform(l_min, l_max)

        # Random amplitude parameters (for control of prior scale)
        scale_eps = self.rng.uniform(0.5 * self.eps_max, 0.8 * self.eps_max)
        scale_theta = self.rng.uniform(0.05 * self.theta_max, 0.3 * self.theta_max)
        
        theta0 = self.rng.uniform(-np.pi, np.pi)
        
        # Unitary RBF kernel
        K_eps = self.rbf_kernel(self.t, lengthscale)
        K_theta = self.rbf_kernel(self.t, lengthscale)
        # Scaled GP samples
        gp_eps1_raw = self.rng.multivariate_normal(np.zeros(self.n_steps), scale_eps**2 * K_eps)
        gp_eps2_raw = self.rng.multivariate_normal(np.zeros(self.n_steps), scale_eps**2 * K_eps)
        gp_theta_var = self.rng.multivariate_normal(np.zeros(self.n_steps), scale_theta**2 * K_theta)
        
        # Force starting point at zero
        e1 = gp_eps1_raw - gp_eps1_raw[0]
        e2 = gp_eps2_raw - gp_eps2_raw[0]
        theta_var = gp_theta_var
        theta = theta0 + np.cumsum(theta_var)
        
        # Final re-scaling to enforce limits
        max_abs_eps = max(np.max(np.abs(e1)), np.max(np.abs(e2)))
        if max_abs_eps > self.eps_max:
            factor_eps = self.eps_max / max_abs_eps
            e1 *= factor_eps
            e2 *= factor_eps
        
        theta_range = np.ptp(theta)  # peak-to-peak
        max_allowed_range = 2 * self.theta_max
        if theta_range > max_allowed_range:
            factor_theta = max_allowed_range / theta_range
            theta = theta0 + (theta - theta0) * factor_theta
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        
        return e1, e2, theta
    
    def generate_multiple_histories(self, n_histories: int = 1000) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return [self.generate_single_history() for _ in range(n_histories)]
    
    def to_cartesian(self, e1: np.ndarray, e2: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert principal strains (e1, e2) and angle theta to Cartesian components (exx, eyy, gamma_xy)
        """
        eps_avg = (e1 + e2) / 2
        eps_dev = (e1 - e2) / 2
        
        exx = eps_avg + eps_dev * np.cos(2 * theta)
        eyy = eps_avg - eps_dev * np.cos(2 * theta)
        gamma_xy = 2 * eps_dev * np.sin(2 * theta)
        
        return exx, eyy, gamma_xy
    
    def save_histories(self, 
                    histories: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], 
                    prefix: str = "strain_history",
                    directory: str = "histories",
                    include_cartesian: bool = True):
        
        os.makedirs(directory, exist_ok=True)
        steps = np.arange(self.n_steps)
        
        for i, (e1, e2, theta) in enumerate(histories):
            if include_cartesian:
                exx, eyy, gamma_xy = self.to_cartesian(e1, e2, theta)
                data = np.column_stack((steps, e1, e2, theta, exx, eyy, gamma_xy))
                header = 'step,e1,e2,theta,exx,eyy,gxy'
            else:
                data = np.column_stack((steps, e1, e2, theta))
                header = 'step,e1,e2,theta'
            
            filename = f"{prefix}_{i+1}.csv"
            filepath = os.path.join(directory, filename)
            
            np.savetxt(
                filepath,
                data,
                delimiter=',',
                header=header,
                comments='',
                fmt='%.12e'
            )
        
        cols = header
        print(f"Saved {len(histories)} histories → columns: {cols} ")

    def plot_distribution(self, 
                    histories: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
                    n_histories: int = 10000,
                    bins: int = 100,                # Number of bins for the classic 2D histogram
                    save_path: str | None = None):
        """
        Plots the joint distribution of principal strains using a classic rectangular 2D histogram (hist2d)
        with marginal histograms and logarithmic color scale.
        """
        if histories is None:
            print(f"Generating {n_histories} histories for the distribution plot...")
            histories = self.generate_multiple_histories(n_histories=n_histories)
        
        # Concatenate all principal strain values from the generated histories
        all_e1 = np.concatenate([h[0] for h in histories])
        all_e2 = np.concatenate([h[1] for h in histories])

        # Create figure and grid layout for main plot + marginal histograms
        fig = plt.figure(figsize=(9, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                            wspace=0.05, hspace=0.05)

        # Main panel: classic rectangular 2D histogram with logarithmic color scale
        ax_main = fig.add_subplot(gs[1, 0])
        hist_obj = ax_main.hist2d(all_e1, all_e2, bins=bins, 
                                cmap='Blues',
                                norm=mcolors.LogNorm(vmin=1),  # Logarithmic scale, avoid log(0)
                                linewidths=0, edgecolor='face')  # Clean look without cell borders
        
        # Add colorbar for the 2D histogram
        fig.colorbar(hist_obj[3], ax=ax_main, label='Counts (log scale)')

        # Set labels and limits for the main plot
        ax_main.set_xlabel(r'$\epsilon_1$')
        ax_main.set_ylabel(r'$\epsilon_2$')
        ax_main.set_aspect('equal', adjustable='box')
        ax_main.grid(True, alpha=0.3)

        # Upper marginal histogram (distribution of ε₁)
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_top.hist(all_e1, bins=bins, color='navy', alpha=0.8, log=True)
        ax_top.set_yscale('log')
        ax_top.set_ylabel('Counts')
        plt.setp(ax_top.get_xticklabels(), visible=False)  # Hide x-tick labels (shared axis)

        # Right marginal histogram (distribution of ε₂)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_right.hist(all_e2, bins=bins, orientation='horizontal', color='navy', alpha=0.8, log=True)
        ax_right.set_xscale('log')
        ax_right.set_xlabel('Counts')
        plt.setp(ax_right.get_yticklabels(), visible=False)  # Hide y-tick labels (shared axis)

        # Overall title
        fig.suptitle('Joint distribution of principal strains', y=0.95, fontsize=14)

        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved in {save_path}")

        plt.show()

    def plot_example(self, e1: np.ndarray | None = None, e2: np.ndarray | None = None, theta: np.ndarray | None = None):
        if e1 is None:
            e1, e2, theta = self.generate_single_history()
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        axs[0].plot(e1, label=r'$\epsilon_1$', color='tab:blue')
        axs[0].plot(e2, label=r'$\epsilon_2$', color='tab:orange')
        axs[0].plot(0, e1[0], 'o', color='black', markersize=5)
        axs[0].set_xlabel('Loading step')
        axs[0].set_ylabel('Principal strain')
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_title('(a) Principal strains vs loading step')
        
        axs[1].plot(e1, e2, color='tab:green')
        axs[1].plot(e1[0], e2[0], 'o', color='black', markersize=5)
        axs[1].set_xlabel(r'$\epsilon_1$')
        axs[1].set_ylabel(r'$\epsilon_2$')
        axs[1].grid(True)
        axs[1].set_aspect('equal', adjustable='box')
        axs[1].set_title('(b) Loading path in principal strain space')
        
        plt.tight_layout()
        plt.show()