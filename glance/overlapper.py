import numpy as np
from scipy.stats import gaussian_kde

def compute_kde(ra_samples, dec_samples, ra_grid, dec_grid, bw='silverman'):
    """
    Computes a 2D KDE over RA-Dec space and evaluates it on a given grid.
    
    Parameters:
        ra_samples (array-like): Samples of right ascension.
        dec_samples (array-like): Samples of declination.
        ra_grid (array-like): Grid values for RA.
        dec_grid (array-like): Grid values for Dec.
        bw (float): Bandwidth for KDE.
    
    Returns:
        pdf (2D np.ndarray): KDE evaluated on the RA-Dec grid.
        RA (2D np.ndarray): Meshgrid of RA values.
        DEC (2D np.ndarray): Meshgrid of Dec values.
    """
    data = np.vstack([ra_samples, dec_samples])
    kde = gaussian_kde(data, bw_method=bw)

    RA, DEC = np.meshgrid(ra_grid, dec_grid)
    grid_points = np.vstack([RA.ravel(), DEC.ravel()])
    pdf = kde(grid_points).reshape(RA.shape)

    return pdf


def find_threshold(pdf, level=0.9):
    """
    Finds the KDE threshold corresponding to a desired confidence level.
    
    Parameters:
        pdf (2D np.ndarray): KDE evaluated on a grid.
        level (float): Desired confidence level (default 0.9 for 90%).
    
    Returns:
        threshold (float): Density value corresponding to the confidence level.
    """
    sorted_pdf = np.sort(pdf.ravel())[::-1]
    cumsum_pdf = np.cumsum(sorted_pdf)
    cumsum_pdf /= cumsum_pdf[-1]  # Normalize to 1
    threshold = sorted_pdf[np.searchsorted(cumsum_pdf, level)]
    return threshold


def ra_dec_overlap(ra1, dec1, ra2, dec2, ra_grid, dec_grid, level=0.9):
    """
    Computes overlapping region in RA-Dec space between two 90% KDE contours.
    
    Parameters:
        ra1, dec1 (array-like): First set of RA-Dec samples.
        ra2, dec2 (array-like): Second set of RA-Dec samples.
        ra_grid, dec_grid (array-like): Grids for KDE evaluation.
    
    Returns:
        overlap_points (2D np.ndarray): Array of [RA, Dec] points in the overlap region.
    """
    pdf1, RA, DEC = compute_kde(ra1, dec1, ra_grid, dec_grid)
    pdf2, _, _ = compute_kde(ra2, dec2, ra_grid, dec_grid)

    threshold1 = find_threshold(pdf1, level=level)
    threshold2 = find_threshold(pdf2, level=level)

    region1 = pdf1 >= threshold1
    region2 = pdf2 >= threshold2

    overlap_region = region1 & region2
    overlap_points = np.column_stack([RA[overlap_region], DEC[overlap_region]])

    return overlap_points


def ra_dec_to_x_y_z (ra, dec):
    theta = dec + np.pi/2
    phi = ra 
    x, y, z = np.sin(theta)* np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
    return x, y, z

def ra_dec_overlap_legacy(ra_1, dec_1, ra_2, dec_2, points = 100):

    xmin, xmax = -1, 1
    ymin, ymax = -1, 1
    zmin, zmax = -1, 1

    xgrid, ygrid, zgrid = np.linspace(xmin, xmax, points), np.linspace(ymin, ymax, points), np.linspace(zmin, zmax, points)

    X1, Y1 = np.meshgrid(xgrid, ygrid)
    grid_points_xy = np.vstack([X1.ravel(), Y1.ravel()])

    Y2, Z2 = np.meshgrid(ygrid, zgrid)
    grid_points_yz = np.vstack([Y2.ravel(), Z2.ravel()])

    Z3, X3 = np.meshgrid(zgrid, xgrid)
    grid_points_zx = np.vstack([Z3.ravel(), X3.ravel()])

    x1, y1, z1 = ra_dec_to_x_y_z (ra_1, dec_1)
    x2, y2, z2 = ra_dec_to_x_y_z (ra_2, dec_2)
    
    data_xy1 = np.vstack([x1, y1])
    data_xy2 = np.vstack([x2, y2])

    kde_xy1 = gaussian_kde(data_xy1, bw_method=0.3)
    kde_xy2 = gaussian_kde(data_xy2, bw_method=0.3)

    # Evaluate KDEs
    pdf_xy1 = kde_xy1(grid_points_xy).reshape(X1.shape)
    pdf_xy2 = kde_xy2(grid_points_xy).reshape(X1.shape)

    # Compute thresholds for 90% C.I.
    threshold_xy1 = find_threshold(pdf_xy1, level=0.9)
    threshold_xy2 = find_threshold(pdf_xy2, level=0.9)

    # Mask grids for 90% regions
    region_xy1 = pdf_xy1 >= threshold_xy1
    region_xy2 = pdf_xy2 >= threshold_xy2

    # Check for overlap
    overlap_xy = np.any(region_xy1 & region_xy2)
    
    data_yz1 = np.vstack([y1, z1])
    data_yz2 = np.vstack([y2, z2])

    kde_yz1 = gaussian_kde(data_yz1, bw_method=0.3)
    kde_yz2 = gaussian_kde(data_yz2, bw_method=0.3)

    # Evaluate KDEs
    pdf_yz1 = kde_yz1(grid_points_yz).reshape(Y2.shape)
    pdf_yz2 = kde_yz2(grid_points_yz).reshape(Y2.shape)

    # Compute thresholds for 90% C.I.
    threshold_yz1 = find_threshold(pdf_yz1, level=0.9)
    threshold_yz2 = find_threshold(pdf_yz2, level=0.9)

    # Mask grids for 90% regions
    region_yz1 = pdf_yz1 >= threshold_yz1
    region_yz2 = pdf_yz2 >= threshold_yz2

    # Check for overlap
    overlap_yz = np.any(region_yz1 & region_yz2)
    
    data_zx1 = np.vstack([z1, x1])
    data_zx2 = np.vstack([z2, x2])

    kde_zx1 = gaussian_kde(data_zx1, bw_method=0.3)
    kde_zx2 = gaussian_kde(data_zx2, bw_method=0.3)

    # Evaluate KDEs
    pdf_zx1 = kde_zx1(grid_points_zx).reshape(Z3.shape)
    pdf_zx2 = kde_zx2(grid_points_zx).reshape(Z3.shape)

    # Compute thresholds for 90% C.I.
    threshold_zx1 = find_threshold(pdf_zx1, level=0.9)
    threshold_zx2 = find_threshold(pdf_zx2, level=0.9)

    # Mask grids for 90% regions
    region_zx1 = pdf_zx1 >= threshold_zx1
    region_zx2 = pdf_zx2 >= threshold_zx2

    # Check for overlap
    overlap_zx = np.any(region_zx1 & region_zx2)
    
    return (overlap_xy & overlap_yz & overlap_zx)#, region_xy1, region_xy2, region_yz1, region_yz2, region_zx1, region_zx2
