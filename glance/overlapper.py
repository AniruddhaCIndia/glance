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
