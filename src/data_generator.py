import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import pdist, squareform
import os
from typing import List, Tuple
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
                 deviatoric_strength: float = 0.75, 
                 seed: int | None = None):
        
        self.n_steps = n_steps
        self.eps_max = eps_max
        self.theta_max = theta_max
        self.deviatoric_strength = deviatoric_strength
        self.rng = np.random.default_rng(seed)
        self.t = np.arange(n_steps)
    
    def rbf_kernel(self, X: np.ndarray, lengthscale: float) -> np.ndarray:
        """
        Radial basis function kernel 
        """
        sqdist = squareform(pdist(X[:, None], 'sqeuclidean'))
        return np.exp(-0.5 * sqdist / lengthscale**2)
    
    def generate_single_history(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera una sola historia de deformaciones con distribución mucho más plana
        y mejor cobertura hasta los bordes del cuadrado [-eps_max, eps_max].
        """
        # Longitud de escala del kernel RBF (suavidad de la trayectoria)
        l_min = self.n_steps / 10
        l_max = self.n_steps / 2
        lengthscale = self.rng.uniform(l_min, l_max)

        # Random amplitude parameters (for control of prior scale)
        scale_eps = self.rng.uniform(0.75 * self.eps_max, 1.5 * self.eps_max)
        scale_theta = self.rng.uniform(0.05 * self.theta_max, 0.30 * self.theta_max)

        if self.rng.random() < self.deviatoric_strength:
            reduction = self.rng.uniform(0.1, 0.5) 
            if self.rng.random() < 0.5:
                scale_e1 = scale_eps * 1
                scale_e2 = scale_eps * reduction
            else:
                scale_e1 = scale_eps * reduction
                scale_e2 = scale_eps * 1
        else:
            scale_e1 = scale_eps * 1
            scale_e2 = scale_eps * 1

        theta0 = self.rng.uniform(-np.pi, np.pi)
        
        # Unitary RBF kernel
        K_eps = self.rbf_kernel(self.t, lengthscale)
        K_theta = self.rbf_kernel(self.t, lengthscale)
        
         # Scaled GP samples
        gp_eps1_raw = self.rng.multivariate_normal(np.zeros(self.n_steps), scale_e1**2 * K_eps)
        gp_eps2_raw = self.rng.multivariate_normal(np.zeros(self.n_steps), scale_e2**2 * K_eps)
        gp_theta_var = self.rng.multivariate_normal(np.zeros(self.n_steps), scale_theta**2 * K_theta)
        
        # Force starting point at zero
        e1 = gp_eps1_raw - gp_eps1_raw[0]
        e2 = gp_eps2_raw - gp_eps2_raw[0]
        theta_var = gp_theta_var
        theta = theta0 + np.cumsum(theta_var)
        
        # Final re-scaling to enforce limits
        max_abs_eps = max(np.max(np.abs(e1)), np.max(np.abs(e2)))
        
        if max_abs_eps > 1e-12:
            u = self.rng.uniform(0.3, 1)
            target_max = self.eps_max * np.sqrt(u)
            factor_eps = target_max / max_abs_eps
            e1 *= factor_eps
            e2 *= factor_eps
        
        theta_range = np.ptp(theta)  # peak-to-peak
        max_allowed_range = 2 * self.theta_max
        if theta_range > max_allowed_range:
            factor_theta = max_allowed_range / theta_range
            theta = theta0 + (theta - theta0) * factor_theta

        max_abs_theta = np.max(np.abs(theta))
        if max_abs_theta > 1e-12:
            if max_abs_theta > np.pi:
                factor_theta = np.pi / max_abs_theta
                theta *= factor_theta        

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
        with marginal histograms and logarithmic scale.
        """

        if histories is None:
            print(f"Generating {n_histories} histories...")
            histories = self.generate_multiple_histories(n_histories=n_histories)
        
        # Concatenate all principal strains
        all_e1 = np.concatenate([h[0] for h in histories])
        all_e2 = np.concatenate([h[1] for h in histories])

        # Creamos solo la figura y el eje principal
        fig, ax_main = plt.subplots(figsize=(10.9, 8.7))

        # Main 2D histogram (central panel)
        H, xedges, yedges, im = ax_main.hist2d(all_e1, all_e2, bins=bins,
                                               cmap='Blues',
                                               norm=mcolors.LogNorm(vmin=1),
                                               linewidths=0, edgecolor='face')

        # === AUTOMATIC CONTOUR LEVELS ===
        max_count = H.max()
        if max_count < 10:
            levels = [5, 10] if max_count >= 5 else []
        else:
            max_exp = int(np.floor(np.log10(max_count)))
            levels = [10 ** k for k in range(1, max_exp + 1)]
            if len(levels) > 6:
                levels = levels[-6:]

        base_colors = ['#ff4d00', '#ffab00', '#d4e157', '#00e676', '#00d4ff', '#2979ff']
        colors = [base_colors[i % len(base_colors)] for i in range(len(levels))]

        # Draw contours
        if levels:
            xcenters = (xedges[:-1] + xedges[1:]) / 2
            ycenters = (yedges[:-1] + yedges[1:]) / 2
            X, Y = np.meshgrid(xcenters, ycenters)
            
            ax_main.contour(X, Y, H.T, levels=levels,
                            colors=colors, linewidths=1.1, alpha=0.94)

        # Main plot settings
        ax_main.set_xlabel(r'$\epsilon_1$')
        ax_main.set_ylabel(r'$\epsilon_2$')
        ax_main.set_aspect('equal', adjustable='box')
        ax_main.grid(True, alpha=0.25)

        divider = make_axes_locatable(ax_main)
        
        ax_top = divider.append_axes("top", size="25%", pad=0.1, sharex=ax_main)
        ax_right = divider.append_axes("right", size="25%", pad=0.1, sharey=ax_main)
        cax = divider.append_axes("right", size="5%", pad=0.4) # Colorbar separado por pad=0.4

        # Top marginal histogram (ε₁)
        ax_top.hist(all_e1, bins=bins, color='navy', alpha=0.88, log=True)
        ax_top.set_yscale('log')
        ax_top.set_ylabel('Counts')
        plt.setp(ax_top.get_xticklabels(), visible=False)

        # Right marginal histogram (ε₂)
        ax_right.hist(all_e2, bins=bins, orientation='horizontal',
                      color='navy', alpha=0.88, log=True)
        ax_right.set_xscale('log')
        ax_right.set_xlabel('Counts')
        plt.setp(ax_right.get_yticklabels(), visible=False)

        # External colorbar
        cbar = fig.colorbar(im, cax=cax, label='Counts (log scale)')
        cbar.ax.tick_params(labelsize=10)

        # Colored marks on colorbar
        for level, color in zip(levels, colors):
            cbar.ax.axhline(level, xmin=0, xmax=0.38,
                            color=color, linewidth=3.5, alpha=0.95)

        fig.suptitle('Joint distribution of principal strains', y=0.96, fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=350, bbox_inches='tight')
            print(f"Plot saved in {save_path}")

        plt.show()

    def plot_example(self, e1: np.ndarray | None = None, e2: np.ndarray | None = None, theta: np.ndarray | None = None):
        if e1 is None:
            e1, e2, theta = self.generate_single_history()
        
        fig, axs = plt.subplots(1, 2, 
                               figsize=(10, 5),                    
                               gridspec_kw={'width_ratios': [1, 1]},
                               constrained_layout=True)
        
        lim = self.eps_max * 1.05
        
        # Figure 1: Principal strains vs loading step
        axs[0].plot(e1, label=r'$\epsilon_1$', color='tab:blue', linewidth=2.2)
        axs[0].plot(e2, label=r'$\epsilon_2$', color='tab:orange', linewidth=2.2)
        axs[0].plot(0, e1[0], 'o', color='black', markersize=7)
        
        axs[0].set_xlim(0, self.n_steps-1)
        axs[0].set_ylim(-lim, lim)
        
        axs[0].set_xlabel('Loading step', fontsize=11)
        axs[0].set_ylabel('Principal strain', fontsize=11)
        axs[0].legend(fontsize=11, loc='upper right')
        axs[0].grid(True, alpha=0.35)
        axs[0].set_title('(a) Principal strains vs loading step', fontsize=11, pad=10)
        
        # Figure 2: Loading path in principal strain space
        axs[1].plot(e1, e2, color='tab:green', linewidth=2.2)
        axs[1].plot(e1[0], e2[0], 'o', color='black', markersize=7)
        
        axs[1].set_xlim(-lim, lim)
        axs[1].set_ylim(-lim, lim)
        
        axs[1].set_xlabel(r'$\epsilon_1$', fontsize=11)
        axs[1].set_ylabel(r'$\epsilon_2$', fontsize=11)
        axs[1].grid(True, alpha=0.35)
        axs[1].set_aspect('equal', adjustable='box')
        axs[1].set_title('(b) Loading path in principal strain space', fontsize=11, pad=10)
        
        fig.suptitle('Example strain history', fontsize=14, y=1.07)
        plt.show()

    def plot_theta(self, n_cases: int = 6):
        # Generate multiple histories to plot their theta and loading paths
        histories = [self.generate_single_history() for _ in range(n_cases)]
        
        fig, axs = plt.subplots(1, 2, 
                               figsize=(10, 5),                    
                               gridspec_kw={'width_ratios': [1, 1]},
                               constrained_layout=True)
        
        lim = self.eps_max * 1.08
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_cases))
        
        # Figure 1: thetas vs loading step
        for i, (_, _, theta) in enumerate(histories):
            theta_unwrapped = np.unwrap(theta)
            axs[0].plot(theta_unwrapped/np.pi, color=colors[i], linewidth=2.1, alpha=0.85)
        
        axs[0].set_xlim(0, self.n_steps-1)
        axs[0].set_ylim(-1.05, 1.05)
        axs[0].set_xlabel('Loading step', fontsize=11)
        axs[0].set_ylabel(r'$\theta/\pi$', fontsize=11)
        axs[0].set_title('(a) Principal orientation θ vs loading step', fontsize=11, pad=10)
        axs[0].grid(True, alpha=0.35)
        
        # Figure 2: thetas in principal strain space
        for i, (e1, e2, _) in enumerate(histories):
            axs[1].plot(e1, e2, color=colors[i], linewidth=2.1, alpha=0.9)
            axs[1].plot(e1[0], e2[0], 'o', color=colors[i], markersize=6.5) 
        
        axs[1].set_xlim(-lim, lim)
        axs[1].set_ylim(-lim, lim)
        axs[1].set_xlabel(r'$\epsilon_1$', fontsize=11)
        axs[1].set_ylabel(r'$\epsilon_2$', fontsize=11)
        axs[1].set_title('(b) Loading paths in principal strain space', fontsize=11, pad=10)
        axs[1].grid(True, alpha=0.35)
        axs[1].set_aspect('equal', adjustable='box')
        
        fig.suptitle(f'Example Strain Histories ({n_cases} cases)', fontsize=14, y=1.07)
        plt.show()

    def plot_strain_ratio_distribution(self, 
                                    histories: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
                                    n_histories: int = 10000,
                                    bins: int = 80,
                                    min_abs_strain: float = 1e-12,
                                    save_path: str | None = None):
        """
        Plots the SIGNED strain ratio ρ = ε₂ / ε₁ (ε₁ ≥ ε₂ algebraically).
        Positive = hydrostatic tendency, Negative = deviatoric tendency.
        """
        if histories is None:
            print(f"Generating {n_histories} histories...")
            histories = self.generate_multiple_histories(n_histories=n_histories)
    
        all_e1 = np.concatenate([h[0] for h in histories])
        all_e2 = np.concatenate([h[1] for h in histories])
        
        # Sort: ε_major ≥ ε_minor
        e_major = np.maximum(all_e1, all_e2)
        e_minor = np.minimum(all_e1, all_e2)
        
        # Ratio signed
        mask = np.abs(e_minor) > 1e-12
        ratios = e_minor[mask] / e_major[mask]
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6.5))
        ax.hist(ratios, bins=bins, range=(-5, 5), color='#1f77b4', alpha=0.85, 
                edgecolor='black', linewidth=0.3)
        ax.axvline(1, color='red', linestyle='--', linewidth=2, label='Pure hydrostatic')
        ax.legend()
        
        ax.set_xlabel(r'$\rho = \epsilon_{min} / \epsilon_{max}$  (signed ratio)', fontsize=13)
        ax.set_ylabel('Counts', fontsize=12)
        ax.set_title('Signed Principal Strain Ratio Distribution', fontsize=14, pad=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=350, bbox_inches='tight')
        plt.show()
    
    def plot_j2_distribution(self, 
                            histories: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
                            n_histories: int = 10000,
                            bins: int = 100,
                            save_path: str | None = None):
        """
        Plots the distribution of the second deviatoric strain invariant J₂.
        
        For incompressible materials (volume constancy, ε₃ = -(ε₁ + ε₂)):
            J₂ = ε₁² + ε₂² + ε₁·ε₂   (always ≥ 0)
        
        Higher J₂ = stronger deviatoric (distortional) deformation.
        """
        if histories is None:
            print(f"Generating {n_histories} histories...")
            histories = self.generate_multiple_histories(n_histories=n_histories)
        
        # Concatenate all principal strains (pairs preserved)
        all_e1 = np.concatenate([h[0] for h in histories])
        all_e2 = np.concatenate([h[1] for h in histories])
        
        # Second deviatoric invariant J₂
        J2 = all_e1**2 + all_e2**2 + all_e1 * all_e2
        
        # Remove any tiny negative numerical artifacts
        J2 = J2[J2 >= 0]
        
        if len(J2) == 0:
            print("Warning: No valid J₂ values.")
            return
        
        # Figure
        fig, ax = plt.subplots(figsize=(10.6, 6.9))
        
        ax.hist(J2, bins=bins, color='#2ca02c', alpha=0.88,
                edgecolor='black', linewidth=0.35)
        
        ax.set_xlabel(r'Second Deviatoric Invariant $J_2 = \epsilon_1^2 + \epsilon_2^2 + \epsilon_1\epsilon_2$', fontsize=13)
        ax.set_ylabel('Counts', fontsize=12)
        ax.set_title('Distribution of Second Deviatoric Invariant $J_2$', fontsize=14, pad=18)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=350, bbox_inches='tight')
            print(f"J₂ distribution plot saved in {save_path}")
        
        plt.show()