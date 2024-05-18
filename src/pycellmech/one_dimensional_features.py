"""
File: <one_dimensional_features.py>

Author: Janan ARSLAN (janan.arslan@icm-institute.org)
Institut du Cerveau - Data Analysis Core (DAC) Platform
Created: 22 FEB 2024
Last Modified: 08 APR 2024
Modification Author: Janan ARSLAN

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import atan2, cos, sin, pi, sqrt



# ==================================================
# === ONE-DIMENSIONAL FEATURE CALCULATIONS =========
# ==================================================


def calculate_tangent_vectors(contour):
    """
    Calculate normalized tangent vectors for each point in a contour.

    Args:
        contour: A 2D array of points (x, y) defining the contour.

    Returns:
        np.array: An array of normalized tangent vectors for each point in the contour.
    """

    x = contour[:, 0]
    y = contour[:, 1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    tangent_vectors = np.array([dx, dy]).T
    norms = np.sqrt(tangent_vectors[:, 0]**2 + tangent_vectors[:, 1]**2)
    tangent_vectors = tangent_vectors / norms[:, np.newaxis]
    return tangent_vectors



def calculate_perpendicular_points(tangent_vectors, contour):
    """
    Calculate points perpendicular to each point on a contour. 

    Args:
        tangent_vectors: Array of tangent vectors for each point in the contour.
        contour: A 2D array of points defining the contour.

    Returns:
        list: A list of points perpendicular to each point on the contour.
    """


    perpendicular_points = []
    for i, tangent_vector in enumerate(tangent_vectors):
        perpendicular_vector = np.array([-tangent_vector[1], tangent_vector[0]])
        pass

    return perpendicular_points


def calculate_clf(contour, tangent_vectors):
    """
    Calculate chord length function (CLF) for each point on a contour.

    Args:
        contour: A 2D array of points defining the contour.
        tangent_vectors: Array of tangent vectors for each point in the contour.

    Returns:
        list: A list of CLF values for each point in the contour.
    """

    clf_values = []
    for i, tangent in enumerate(tangent_vectors):
        perp = np.array([-tangent[1], tangent[0]])
        prev_index = (i - 1) % len(contour)
        next_index = (i + 1) % len(contour)
        prev_point = contour[prev_index]
        next_point = contour[next_index]
        chord_length = np.linalg.norm(next_point - prev_point)
        clf_values.append(chord_length)
    return clf_values
    


def calculate_tangent_angle(y0, y1, x0, x1):
    """
    Calculate the angle of the tangent at a point.

    Args:
        y0, y1: y-coordinates of two points.
        x0, x1: x-coordinates of two points.

    Returns:
        float: The tangent angle at the given point.
    """

    if (x1 - x0) == 0:
        return 0
    else:
        return np.arctan2((y1 - y0), (x1 - x0))


def calculate_curvature(dx, dy, d2x, d2y):
    """
    Calculate the curvature at a point using derivatives.

    Args:
        dx, dy: First derivatives in x and y directions.
        d2x, d2y: Second derivatives in x and y directions.

    Returns:
        float: Curvature at the point.
    """

    denominator = (dx**2 + dy**2)**1.5
    if denominator != 0:
        return (dx * d2y - dy * d2x) / denominator
    else:
        return 0



def calculate_triangle_area(p0, p1, c):
    """
    Calculate the area of a triangle formed by three points.

    Args:
        p0, p1, c: Points forming the triangle.

    Returns:
        float: Area of the triangle.
    """

    return 0.5 * abs((p0[0] - c[0]) * (p1[1] - c[1]) - (p1[0] - c[0]) * (p0[1] - c[1]))



def calculate_tar(contour):
    """
    Calculate Triangle Area Representation (TAR) for a contour.

    Args:
        contour: A 2D array of points defining the contour.

    Returns:
        list: TAR values for the contour.
    """

    tar_values = []
    for i in range(2, len(contour)):
        p0, p1, p2 = contour[i-2], contour[i-1], contour[i]
        area = calculate_triangle_area(p0, p1, p2)
        tar_values.append(area)
    return tar_values


# ==================================================
# === ONE-DIMENSIONAL PLOT FUNCTIONS ===============
# ==================================================


def plot_original_contour(ax, binary_image):
    ax.imshow(binary_image, cmap='gray')
    ax.set_title('Original Contour')
    ax.axis('off')

def plot_centroid(ax, binary_image, cx, cy):
    ax.imshow(binary_image, cmap='gray')
    ax.scatter([cx], [cy], color='red')
    ax.set_title('Centroid')
    ax.axis('off')

def plot_complex_coordinates(ax, binary_image, cx, cy, complex_coordinates):
    ax.imshow(binary_image, cmap='gray', origin='upper')
    x_coords = [cx + np.real(cc) for cc in complex_coordinates]
    y_coords = [cy + np.imag(cc) for cc in complex_coordinates]
    magnitudes = [np.abs(cc) for cc in complex_coordinates]
    ax.scatter(x_coords, y_coords, c=magnitudes, cmap='jet')
    ax.set_title('Complex Coordinates')
    ax.axis('off')


def plot_cdf(ax, binary_image, cdf, cx, cy, contour):
    contour_array = contour.squeeze()
    x = contour_array[:, 0]
    y = contour_array[:, 1]
    sorted_cdf = np.sort(cdf)
    cumulative = np.cumsum(sorted_cdf)
    normalized_cumulative = cumulative / cumulative[-1]
    ax.imshow(binary_image, cmap='gray')
    ax.scatter([cx], [cy], color='red', zorder=5)  

    for x_val, y_val in zip(x, y):
        ax.plot([cx, x_val], [cy, y_val], color='yellow', linewidth=0.5)

    ax.set_title('CDF')
    ax.axis('off')


def plot_area_function(ax, binary_image, cx, cy, contour):
    contour_array = contour.squeeze()
    ax.imshow(binary_image, cmap='gray')
    ax.plot(contour_array[:, 0], contour_array[:, 1], color='blue', linewidth=0.5)
    ax.scatter([cx], [cy], color='red', zorder=5)

    for i in range(len(contour_array)):
        p0 = contour_array[i]
        p1 = contour_array[(i + 1) % len(contour_array)]
        ax.plot([cx, p0[0], p1[0], cx], [cy, p0[1], p1[1], cy], color='blue', linewidth=0.5)
    ax.set_title('Area Function')
    ax.axis('off')



def plot_clf(ax, binary_image, contour, tangent_vectors):
    contour_array = contour.squeeze()
    ax.imshow(binary_image, cmap='gray')
    ax.set_title('CLF')

    for point, tangent in zip(contour_array, tangent_vectors):
        perp_dir = np.array([-tangent[1], tangent[0]])

        perp_distances = (contour_array - point) @ perp_dir

        farthest_point = contour_array[np.argmax(np.abs(perp_distances))]

        ax.plot([point[0], farthest_point[0]], [point[1], farthest_point[1]], color='green', linewidth=0.5)

    ax.axis('off')

    

def plot_tar(ax, binary_image, contour, tar_values):
    contour_array = contour.squeeze()
    for i in range(2, len(contour_array)):
        p0, p1, p2 = contour_array[i-2], contour_array[i-1], contour_array[i]
        triangle = np.array([p0, p1, p2])
        ax.fill(triangle[:, 0], triangle[:, 1], color="green")
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_aspect('equal', 'box')

    ax.axis('off')
    ax.set_title('TAR')


# ==================================================
# === CALL ALL ONE-DIMENSIONAL FEATURES ============
# ==================================================


def get_one_dimensional_features(contour, cx, cy):

    # Initialize lists
    complex_coordinates = []
    cdf = []
    curvature_list = []
    tangent_angles = []
    area_function = []
    clf_values = []
    cog = []

    # Simplify the contour array
    contour_array = contour.squeeze()
    x = contour_array[:, 0]
    y = contour_array[:, 1]

    # Calculate gradients
    dx = np.gradient(x)
    dy = np.gradient(y)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    # Curvature, CDF, and COG Calculations
    for i, point in enumerate(contour_array):
        xi, yi = point
        complex_coordinate = (xi - cx) + 1j * (yi - cy)
        complex_coordinates.append(complex_coordinate)
        cdf.append(np.abs(complex_coordinate))
        curvature_list.append(calculate_curvature(dx[i], dy[i], d2x[i], d2y[i]))
        if i < len(contour_array) - 1:
            area_function.append(calculate_triangle_area(contour_array[i], contour_array[i+1], (cx, cy)))

    # Tangent angles Calculations
    for i in range(len(contour_array)):
        P = 1
        i_prev = (i - P) % len(contour_array)
        tangent_angle = calculate_tangent_angle(y[i_prev], y[i], x[i_prev], x[i])
        tangent_angles.append(tangent_angle)

    for i in range(len(contour) - 1):
        p0 = contour[i][0]
        p1 = contour[(i + 1) % len(contour)][0]
        area = calculate_triangle_area(p0, p1, (cx, cy))
        area_function.append(area)


    # Tangent Vector Calculations
    tangent_vectors = calculate_tangent_vectors(contour_array)

    # CLF Calculations
    clf_values = calculate_clf(contour_array, tangent_vectors)

    # TAR Calculations
    tar_values = calculate_tar(contour_array)

    one_d_features = {
            "centroid": (cx, cy),
            "complex_coordinates": complex_coordinates,
            "cdf": cdf,
            "tangent_angles": tangent_angles,
            "curvature": curvature_list,
            "area_function": area_function,
            "tangent_vectors": tangent_vectors,
            "tar_values": tar_values,
            "tar": tar_values,
            "clf": clf_values
        }

    return one_d_features
