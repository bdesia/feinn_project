import numpy as np
import matplotlib.pyplot as plt

class Mesh2D:
    """
    A 2D finite element mesh representation.
    
    Attributes:
        nodes (ndarray): (nnod x 2) array with [x, y] coordinates of nodes.
        elements (dict): (etype: list with element connectivity)
        nnod (int): Number of nodes.
        nelem (int): Number of elements.
        node_groups (dict): Dictionary mapping group names to sets of node indices (1-based).
        element_groups (dict): Dictionary mapping group names to sets of element indices (1-based).
    
    """
    
    def __init__(self):
        """
        Initialize the mesh with coordinates and connectivity.
        """
        self.coordinates = []
        self.elements = {}
        self.nnod = 0
        self.nelem = 0
        self.node_groups = {}                   # Initialize empty node groups
        self.element_groups = {}                # Initialize empty element groups
    
    @staticmethod
    def _check_elem_type(elem_type: str, stop=True):
        supported_types = ['quad','line']
        if elem_type not in supported_types:
            if stop:
                raise ValueError(f"Unsupported element type: {elem_type}. Supported types: {supported_types}")
            else:
                return False
        return True
    
    @staticmethod
    def get_edge_order(elem_type: str, npe: int):
        if elem_type == 'quad':
            if npe == 4:
                return [0, 1, 2, 3, 0]
            elif npe == 8:
                return [0, 4, 1, 5, 2, 6, 3, 7, 0]
            elif npe == 9:
                return [0, 4, 1, 5, 2, 6, 3, 7, 0]
            else:
                raise ValueError("Unsupported number of nodes per element for quad.")

    def add_node(self, x: float, y: float):
        self.coordinates.append(np.array([x, y]))

    def add_element(self, elem_type: str, connectivity: np.ndarray):
        self._check_elem_type(elem_type)
        self.elements[elem_type].append(connectivity)

    def get_node_coords(self, node_id: int):
        """
        Return coordinates of a specific node (1-based index).
        
        Args:
            node_id (int): Node index (1-based).
            
        Returns:
            ndarray: [x, y] coordinates of the node.
        """
        return self.coordinates[node_id - 1]

    def get_element_nodes(self, elem_type: str, elem_id: int):
        """
        Return nodes of a specific element (1-based index).
        
        Args:
            etype (str):    Element type.
            elem_id (int):  Element index (1-based).
            
        Returns:
            ndarray: Array of node indices (1-based) for the element.
        """
        self._check_elem_type(elem_type)
        if elem_id > 0:
            return self.elements[elem_type][elem_id - 1]
        else:
            raise ValueError("Element id must be between 1 and nelem.")

    def add_node_group(self, group_name: str, node_indices: list):
        """
        Add a group of nodes under a specified name.
        
        Args:
            group_name (str): Name of the node group.
            node_indices (list): Set of node indices (1-based).
        """
        self.node_groups[group_name] = set(node_indices)
        if not all(1 <= idx <= self.nnod for idx in node_indices):
            raise ValueError("Node indices must be between 1 and nnod.")
    
    def add_element_group(self, group_name: str, element_indices: list):
        """
        Add a group of elements under a specified name.
        
        Only for quad elements.
        
        Args:
            group_name (str): Name of the element group.
            element_indices (list): Set of element indices (1-based).
        """
        self.element_groups[group_name] = set(element_indices)
        if not all(1 <= idx <= self.nelem for idx in element_indices):
            raise ValueError("Element indices must be between 1 and nelem.")

    def find_nodes(self, value: float, axis: int, tol = 1e-10):
        """
        Find node IDs where the specified coordinate (x or y) is approximately equal to a given value.

        Args:
            value (float): The coordinate value to search for.
            axis (int): The axis to consider (1 for x-coordinate, 2 for y-coordinate).
            tol (float): Tolerance for matching the coordinate value. Defaults to 1e-10.

        Returns:
        list[int]: List of node IDs (starting from 1) that satisfy the condition.      
        """
        mask = np.abs(self.coordinates[:, axis - 1] - value) < tol
        node_ids = np.where(mask)[0] + 1  
        return node_ids.tolist()

    def plot(self, show_nodes=True, node_groups_to_plot=None, element_groups_to_plot=None, show_node_ids=False, show_element_ids=False, ax=None):
        """
        Plot the 2D mesh, with options to highlight node/element groups.
        
        Args:
            node_groups_to_plot (list or None): List of node group names to highlight. None for no groups.
            element_groups_to_plot (list or None): List of element group names to highlight. None for no groups.
            show_node_ids (bool): If True, label nodes with their 1-based indices. Defaults to False.
            show_element_ids (bool): If True, label elements with their 1-based indices. Defaults to False.
            ax (matplotlib.axes.Axes or None): Axes to plot on. If None, creates a new figure.
        
        Returns:
            matplotlib.axes.Axes: The axes object used for plotting.
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        # Plot all nodes
        if show_nodes:
            ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], c='black', s=20, label='Nodes')
        
        # Plot elements
        etype_old = None
        for etype, connect_list in self.elements.items():
            if etype in ['quad','tri']:
                # convert to numpy array (1-based → 0-based)
                connect = np.array(connect_list) - 1   # shape: (nelem, npe)
                
                npe = connect.shape[1]
                edge_order = self.get_edge_order(etype, npe)
                
                # Draw each element
                for i, nodes in enumerate(connect):
                    x = self.coordinates[nodes[edge_order], 0]
                    y = self.coordinates[nodes[edge_order], 1]
                    
                    label = f'{etype} elements' if etype != etype_old else None
                    ax.plot(x, y, 'b-', alpha=0.5, linewidth=1.2, label=label)
                    etype_old = etype
                    
                    if show_element_ids:
                        centroid = np.mean(self.coordinates[nodes], axis=0)
                        ax.text(centroid[0], centroid[1], str(i + 1),
                                color='blue', fontsize=8, ha='center', va='center', zorder=4)
        
        # Plot node groups with different colors
        if node_groups_to_plot:
            colors = plt.cm.tab10(np.linspace(0, 1, len(node_groups_to_plot)))
            for group, color in zip(node_groups_to_plot, colors):
                if group in self.node_groups:
                    node_indices = np.array(list(self.node_groups[group])) - 1  # Convert to 0-based
                    ax.scatter(self.coordinates[node_indices, 0], self.coordinates[node_indices, 1], 
                              c=[color], s=50, label=group)
        
        # Plot element groups with different edge colors
        if element_groups_to_plot:
            colors = plt.cm.Set1(np.linspace(0, 1, len(element_groups_to_plot)))
            etype = 'quad'
            for group_name, color in zip(element_groups_to_plot, colors):
                if group_name not in self.element_groups:
                    continue
                
                elem_ids = list(self.element_groups[group_name])
                first = True
                
                for elem_id in elem_ids:
                    nodes_1based = self.get_element_nodes(elem_type=etype, elem_id=elem_id)
                    nodes = np.array(nodes_1based) - 1
                    
                    npe = len(nodes)
                    edge_order = self.get_edge_order(etype, npe)
                    
                    x = self.coordinates[nodes[edge_order], 0]
                    y = self.coordinates[nodes[edge_order], 1]
                    
                    label = group_name if first else None
                    
                    # Fill group elements with color
                    ax.fill(x, y,
                            facecolor=color,
                            edgecolor='black',
                            linewidth=1.0,
                            alpha=0.4,          # transparency
                            label=label)
                    
                    first = False
                    
        # Plot node IDs if requested
        if show_node_ids:
            for i in range(self.nnod):
                ax.text(self.coordinates[i, 0], self.coordinates[i, 1], str(i + 1), color='black', fontsize=8)
        
        # Set plot properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('2D Finite Element Mesh')
        # ax.grid(True)
        ax.legend()
        ax.set_aspect('equal')  # Ensure equal scaling for x and y axes
        return

    @classmethod
    def from_salome_med(cls, filepath: str, verbose: bool = True):
        """
        Load a 2D mesh exported from SALOME-MECA (SMESH module) in .med format.
        
        Args:
            filepath (str): Path to the .med file
        
        Returns:
            Mesh2D: Populated mesh instance according to the MED file
        """
        import meshio
        from collections import defaultdict
        
        # Read the MED file
        m = meshio.read(filepath)

        # Basic validation: must be 2D
        if m.points.shape[1] != 2:
            raise ValueError("Expected 2D mesh — points must have shape (nnod, 2)")
        
        # Create new empty Mesh2D instance
        instance = cls()
        
        # Load nodes
        instance.coordinates = np.asarray(m.points, dtype=float)
        instance.nnod = len(instance.coordinates)
        
        # Find and load 2D cell block
        for cell_block in m.cells:
            ctypeX = cell_block.type
            ctype = next((ctypeX[:i] for i, c in enumerate(ctypeX) if c.isdigit()), ctypeX)     # Extract base type
            if cls._check_elem_type(ctype, stop=False):
                connectivity = cell_block.data  # (nelem, nodes_per_elem)

                # Convert to 1-based indices
                instance.elements[ctype] = [row.astype(np.int32) for row in (connectivity + 1)]

        # Compute total number of elements
        instance.nelem = sum(len(connectivity) for connectivity in instance.elements.values())

        # Node groups: only existing ones
        if hasattr(m, 'point_tags') and m.point_tags:
            groups = defaultdict(set)
            tags_array = m.point_data.get('point_tags', np.array([]))
            
            for tag, names in m.point_tags.items():
                if names:
                    nodes = np.flatnonzero(tags_array == tag) + 1
                    for name in map(str.strip, names):
                        groups[name] |= set(nodes)
            
            instance.node_groups.update(groups)
        
        # Element groups: only existing ones
        groups = defaultdict(set)

        if 'cell_tags' in m.cell_data_dict:
            cell_tags_content = m.cell_data_dict['cell_tags']
            
            for cell_type, tags_array in cell_tags_content.items():
                if not isinstance(tags_array, np.ndarray) or len(tags_array) == 0:
                    continue
                
                unique_tags = np.unique(tags_array[tags_array != 0])  # ignora 0
                
                for tag_id in unique_tags:
                    mask = tags_array == tag_id
                    elems_1based = np.flatnonzero(mask) + 1
                    
                    # Get name associated to tag_id
                    names = []
                    
                    if hasattr(m, 'cell_tags') and tag_id in m.cell_tags:
                        value = m.cell_tags[tag_id]
                        
                        if isinstance(value, list):
                            names = [str(v).strip() for v in value if v and str(v).strip()]
                        
                        elif isinstance(value, str):
                            names = [value.strip()]
                        
                        elif isinstance(value, dict) and 'name' in value:
                            names = [str(value['name']).strip()]

                    # Add element to each group
                    for group_name in names:
                        groups[group_name].update(elems_1based)

        instance.element_groups.update(groups)
        
        # Summary print
        if verbose:
            print(f"Loaded SALOME .med mesh: {instance.nnod} nodes, {instance.nelem} elements")
            if instance.node_groups:
                print(f"Node groups ({len(instance.node_groups)}): {list(instance.node_groups.keys())}")
            if instance.element_groups:
                print(f"Element groups ({len(instance.element_groups)}): {list(instance.element_groups.keys())}")
        
        return instance

class UniformQuadMesh2D(Mesh2D):
    """
    A subclass of Mesh2D for generating uniform quadrilateral meshes in 2D rectangular domains.
    
    Attributes (inherited from Mesh2D, set after compute()):
        nodes (ndarray): (nnod x 2) array with [x, y] coordinates of nodes.
        elements (dict): (etype: list with element connectivity)
        nnod (int): Number of nodes.
        nelem (int): Number of elements.
        node_groups (dict): Dictionary mapping group names to sets of node indices (1-based).
        element_groups (dict): Dictionary mapping group names to sets of element indices (1-based).
    
    Additional Attributes:
        lx (float): Domain length in x-direction.
        ly (float): Length in y-direction.
        nx (int): Number of elements in x-direction.
        ny (int): Number of elements in y-direction.
        elem_type (str): Quadrilateral element type ('Q4', 'Q8', 'Q9').
    """
    
    def __init__(self, lx: float, ly: float, nx: int, ny: int, elem_type='Q4'):
        super().__init__()
        """
        Initialize the uniform quadrilateral mesh parameters. Call compute() to generate the mesh.
        
        Args:
            lx (float): Length in x-direction.
            ly (float): Length in y-direction.
            nx (int): Number of elements in x-direction.
            ny (int): Number of elements in y-direction.
            elem_type (str, optional): Element type ('Q4', 'Q8', 'Q9'). Defaults to 'Q4'.
        
        Raises:
            ValueError: If elem_type is invalid or inputs are non-positive.
        """
        if elem_type.upper() not in ['Q4', 'Q8', 'Q9']:
            raise ValueError("elem_type must be 'Q4', 'Q8', or 'Q9'.")
        if lx <= 0 or ly <= 0:
            raise ValueError("lx and ly must be positive.")
        if nx <= 0 or ny <= 0:
            raise ValueError("nx and ny must be positive integers.")
        
        self.lx = float(lx)
        self.ly = float(ly)
        self.nx = int(nx)
        self.ny = int(ny)
        self.elem_type = elem_type.upper()
        
    def compute(self):
        """
        Generate nodes and element connectivity for a uniform quadrilateral mesh.
        Sets the inherited Mesh2D attributes and adds boundary node groups.
        
        Node numbering is row-major (left to right, bottom to top).
        - Q4: top-right, top-left, bottom-left, bottom-right
        - Q8/Q9: top-right, top-left, bottom-left, bottom-right, mid-top, mid-left, mid-bottom, mid-right, [center for Q9]
        
        Returns:
            None
        """

        # -----------------------------------------------
        # NODES GENERATION
        # -----------------------------------------------

        npe = 4 if self.elem_type == 'Q4' else 8 if self.elem_type == 'Q8' else 9

        step1 = self.lx / self.nx
        step2 = self.ly / self.ny
        a = 1.0
        if npe == 8 or npe == 9:
            step2 = step2 / 2
            a = 0.5

        # x1 = np.arange(0, self.lx + a * step1, a * step1)
        x2 = np.arange(0, self.ly + step2, step2)

        coordinates = []
        k = 1
        for j in x2:
            for i in np.arange(0, self.lx + (step1 * a), step1 * a):
                coordinates.append([i, j])
                k += 1
            if npe == 8:
                a = 1.5 - a  # Alternar entre 0.5 y 1.0 para zigzag

        self.coordinates = np.array(coordinates)
        self.nnod = len(self.coordinates)

        # -----------------------------------------------
        # ELEMENT GENERATION
        # -----------------------------------------------

        aux = 1
        m = 1
        n = 1
        l = 1

        if npe == 4:
            b = 1
            c = 2
        elif npe == 8:
            b = 3
            c = 4
            p = 1
        elif npe == 9:
            b = 4
            c = 4
            p = 2

        self.nelem = self.nx * self.ny

        quad_list = []
        for h in range(self.nelem):
            connectivity = np.zeros(npe, dtype=int)

            connectivity[0] = b * self.nx + c + m
            connectivity[1] = b * self.nx + m + (c // 2)
            connectivity[2] = m
            connectivity[3] = m + (c // 2)

            if npe != 4:
                connectivity[4] = b * self.nx + c + m - 1
                connectivity[5] = 2 * self.nx + aux + l
                connectivity[6] = m + (c // 2) - 1
                connectivity[7] = 2 * self.nx + aux + l + p
                if npe == 9:
                    connectivity[8] = 2 * self.nx + m + 2

            # Actualización de contadores
            if n == self.nx:
                m = aux + b * self.nx + (c // 2)
                aux = aux + b * self.nx + (c // 2)
                n = 1
                l = 1
            else:
                if npe == 4:
                    m += 1
                else:
                    m += 2
                    if npe == 8:
                        l += 1
                    elif npe == 9:
                        l += 2
                n += 1

            quad_list.append(connectivity)
        
        self.elements['quad'] = quad_list


        # -----------------------------------------------
        # GROUP GENERATION
        # -----------------------------------------------
        
        self.node_groups['all']    = list(range(1, self.nnod + 1))
        self.node_groups['bottom'] = self.find_nodes(0.0, axis=2)
        self.node_groups['top']    = self.find_nodes(self.ly, axis=2)
        self.node_groups['left']   = self.find_nodes(0.0, axis=1)
        self.node_groups['right']  = self.find_nodes(self.lx, axis=1)

        # --- Grupos de nodos ---
        self.element_groups['all'] = list(range(1, self.nelem + 1))         # All elements
        self.element_groups['bottom'] = list(range(1, self.nx + 1))        
        self.element_groups['top'] = list(range(self.nx * (self.ny-1) + 1, self.nx * self.ny + 1))
        self.element_groups['left'] = list(range(1, (self.ny - 1) * self.nx + 2, self.nx))
        self.element_groups['right'] = list(range(self.nx, self.ny * self.nx + 1, self.nx))