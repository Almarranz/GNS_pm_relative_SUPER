import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
import sys
# field_one = 10
# field_two = 4

# chip_one = 0
# chip_two = 0

# gns2 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/pruebas/F{field_two}/{field_two}_H_chips_opti.ecsv', format = 'ascii.ecsv')


# def grid_stars(table, x_col, y_col, mag_col, mag_min, mag_max, grid_size, isolation_radius):
def grid_stars(table, x_col, y_col, mag_col, mag_min, mag_max, grid_size, isolation_radius):
    """
    Selects isolated stars from an Astropy Table within specified magnitude limits and spatial isolation criteria.

    Parameters:
    -----------
    table : astropy.table.Table
        Input table containing star data.
    x_col : str
        Name of the column representing the x-coordinate.
    y_col : str
        Name of the column representing the y-coordinate.
    mag_col : str
        Name of the column representing the magnitude.
    mag_min : float
        Minimum magnitude for filtering stars.
    mag_max : float
        Maximum magnitude for filtering stars.
    grid_size : int, optional
        Number of divisions along each axis for the grid (default is 50).
    isolation_radius : float, optional
        Radius in arcseconds to consider a star isolated (default is 0.5).

    Returns:
    --------
    astropy.table.Table
        Table containing the selected isolated stars.
    """
    # Step 1: Define the Grid Boundaries
    
    
    x_min = np.nanmin(table[x_col])
    x_max = np.nanmax(table[x_col])
    y_min = np.nanmin(table[y_col])
    y_max = np.nanmax(table[y_col])

    
    # Step 2: Create the Grid
    x_edges = np.linspace(x_min, x_max, grid_size + 1)
    # y_edges = np.linspace(y_min, y_max, grid_size + 1)
    y_edges = np.linspace(y_min, y_max, int(grid_size/2) + 1)

    # Step 3: Filter by Magnitude
    filtered_stars = table[(table[mag_col] >= mag_min) & (table[mag_col] <= mag_max)]

    # Step 4: Assign Stars to Grid Cells
    x_indices = np.digitize(filtered_stars[x_col], x_edges) - 1
    y_indices = np.digitize(filtered_stars[y_col], y_edges) - 1

    # Combine x and y indices to create unique cell identifiers
    cell_indices = x_indices * grid_size + y_indices

    # Step 5: Select Representative Stars
    selected_stars = []

    for cell in np.unique(cell_indices):
        stars_in_cell = filtered_stars[cell_indices == cell]
        
        stars_in_cell.sort(mag_col)

        if len(stars_in_cell) == 0:
            continue
        
        # Convert to SkyCoord for angular separation calculations
        coords = SkyCoord(l=stars_in_cell['l'], b=stars_in_cell['b'], frame = 'galactic')

        # Check for isolation within the specified radius
        for i, coord in enumerate(coords):
            separations = coord.separation(coords)
            neighbors_within_radius = np.sum(separations < isolation_radius * u.arcsec) - 1
            if neighbors_within_radius == 0:
                selected_stars.append(stars_in_cell[i])
                break  # Select only one star per cell

    # Convert the list to an Astropy Table
    selected_stars_table = vstack(selected_stars) if selected_stars else Table(names=table.colnames)

    return selected_stars_table, x_edges, y_edges

# gns2_g, xed, yed = grid_stars(gns2, 'x', 'y', 'H', 12, 15, 100, 0.5)

# # %
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1,1)
# # ax.scatter(gns2['x'][::100], gns2['y'][::100], alpha = 0.1, s = 1)
# ax.scatter(gns2_g['x'], gns2_g['y'], s = 1)






















































