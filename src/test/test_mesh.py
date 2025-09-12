import numpy as np
from mesh_utils import UniformQuadMesh2D
import matplotlib.pyplot as plt

def test_mesh_plot():
    """
    Test the plot method of UniformQuadMesh2D for Q4, Q8, and Q9 meshes.
    """
    # Test parameters
    lx, ly = 2.0, 2.0
    nx, ny = 2, 2
    element_types = ['Q4', 'Q8', 'Q9']
    
    for etype in element_types:
        print(f"Testing {etype} mesh plot")
        # Create and compute mesh
        mesh = UniformQuadMesh2D(lx, ly, nx, ny, element_type=etype)
        mesh.compute()
        
        # Plot mesh with boundary node groups and all elements
        fig, ax = plt.subplots()
        mesh.plot(
            node_groups_to_plot=['all'],
            element_groups_to_plot=['all'],
            show_node_ids=False,
            show_element_ids=False,
            ax=ax
        )
        plt.title(f"{etype} Mesh")
        plt.show()

if __name__ == "__main__":
    test_mesh_plot()
    print("All plots generated.")