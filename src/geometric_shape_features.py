'''
======================================================================
 Title:                   PYCELLMECH - GEOMETRIC SHAPES
 Creating Author:         Janan Arslan
 Creation Date:           22 FEB 2024
 Latest Modification:     25 MAR 2024
 Modification Author:     Janan Arslan
 E-mail:                  janan.arslan@icm-institute.org
 Version:                 1.0
======================================================================


Geometric Shape Features extracted as part of the pycellmech. 

'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import atan2, cos, sin, sqrt, pi
import cv2

### --- CALCULATIONS --- ###

# Calculate curvature at a point
def calculate_curvature(dx, dy, d2x, d2y):
    denominator = (dx**2 + dy**2)**1.5
    if denominator != 0:
        return (dx * d2y - dy * d2x) / denominator
    else:
        return 0

# Calculate the area of a triangle given by three points
def calculate_triangle_area(p0, p1, c):
    return 0.5 * abs((p0[0] - c[0]) * (p1[1] - c[1]) - (p1[0] - c[0]) * (p0[1] - c[1]))


# Calculate pq used for AMI
def calculate_pq(contour):
    if contour.ndim > 2:
        contour = contour.reshape(-1, 2)

    p = np.sum(contour[:, 0]**2)
    q = np.sum(contour[:, 0] * contour[:, 1])
    return p, q


# Axis of Minimum Inertia
def calculate_AMI(p, q, r, phi):
    return 0.5 * (p + r) - 0.5 * (p - r) * np.cos(2 * phi) - q * np.sin(2 * phi)


# Find angle phi that minimizes inertia
def calculate_phi(p, q, r):
    phi = 0.5 * np.arctan2(q, p - r)
    return phi if phi >= -np.pi / 2 else phi + np.pi

# Average Bending Energy
def calculate_ABE(curvature_list):
    abe_value = np.mean(np.square(curvature_list))
    return abe_value

# Eccentricity
def calculate_eccentricity(contour):
    M = cv2.moments(contour)
    mu20 = M['mu20'] / M['m00']
    mu02 = M['mu02'] / M['m00']
    mu11 = M['mu11'] / M['m00']
    
    # Compute the eigenvalues
    common = sqrt((mu20 - mu02)**2 + 4*mu11**2)
    lambda1 = 0.5*(mu20 + mu02 + common)
    lambda2 = 0.5*(mu20 + mu02 - common)
    
    eccentricity = sqrt(1 - (lambda2 / lambda1))
    return eccentricity


def calculate_covariance_matrix(contour):

    if contour.ndim > 2:
        contour = contour.reshape(-1, 2)

    mean_x, mean_y = np.mean(contour, axis=0)

    contour_normalized = contour - [mean_x, mean_y]

    covariance_matrix = np.cov(contour_normalized.T)

    return covariance_matrix

# Minimum Bounding Rectangle and Elongation
def calculate_mbr_and_elongation(contour):
    rect = cv2.minAreaRect(contour)
    (cx, cy), (width, height), angle = rect
    
    if width < height:
        width, height = height, width

    # Eccentricity E and Elongation Elo
    E = height / width
    Elo = 1 - (width / height)
    
    return (cx, cy), (width, height), angle, E, Elo

# Circularity Ratio
def calculate_and_visualize_circularity(contour, binary_image_path):
    binary_image = cv2.imread(binary_image_path, 0)

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    circle_area = (perimeter ** 2) / (4 * pi)
    circularity_ratio = area / circle_area

    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)

    visual = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)

    cv2.drawContours(visual, [contour], -1, (255, 255, 255), -1)

    cv2.circle(visual, center, radius, (0, 255, 0), 2)  # Circle in green
    
    binary_image_color = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    combined_image = np.hstack((binary_image_color, visual))
    
    text = f'CR: {circularity_ratio:.4f}'
    cv2.putText(combined_image, text, (center[0] + radius + 10, center[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return circularity_ratio, combined_image

# Ellipse Variance
def calculate_ev(contour, covariance_matrix):
    M = len(contour)
    L = contour - contour.mean(axis=0)
    D_prime = np.sqrt(np.sum(L @ np.linalg.inv(covariance_matrix) * L, axis=1))
    mu = D_prime.mean()
    sigma = D_prime.std()
    EV = sigma / mu
    return EV



def calculate_moments(contour):
    # Calculate spatial moments up to second order
    moments = cv2.moments(contour)
    m00 = moments['m00']
    m10 = moments['m10']
    m01 = moments['m01']
    m20 = moments['mu20'] / m00
    m02 = moments['mu02'] / m00
    m11 = moments['mu11'] / m00
    return m00, m10, m01, m20, m02, m11


# Elipcity based on moment invariantss
def calculate_em(m00, m20, m02, m11):
    I = (m20 * m02 - m11**2) / m00**4
    EM = min([16 * pi**2 * I, (16 * pi**2 * I)**-1])
    return EM

# Solidity
def calculate_solidity(contour):
    shape_area = cv2.contourArea(contour)
    
    hull = cv2.convexHull(contour)
    
    hull_area = cv2.contourArea(hull)
    
    if hull_area == 0:
        return 0

    solidity = float(shape_area) / hull_area
    return solidity, hull


### --- PLOT FUNCTIONS --- ###

def plot_AMI(ax, binary_image, contour, cx, cy, alpha):
    # Compute the line for AMI based on alpha
    line_length = 100
    x2 = cx + line_length * np.cos(alpha)
    y2 = cy + line_length * np.sin(alpha)
    x3 = cx - line_length * np.cos(alpha)
    y3 = cy - line_length * np.sin(alpha)

    ax.imshow(binary_image, cmap='gray')
    ax.set_title('AMI')

    ax.plot([x3, x2], [binary_image.shape[0] - y3, binary_image.shape[0] - y2], color='red', linewidth=2)  # Flip y-coordinate
    ax.scatter(cx, binary_image.shape[0] - cy, color='yellow')



def plot_ABE(ax, binary_image, contour, abe_value, curvature_list):
    contour_array = contour.squeeze()
    # Scaling factor for ABE
    # Normalize and scale the curvature values for plotting
    curvature_norm = curvature_list / np.max(np.abs(curvature_list))  
    curvature_scaled = curvature_norm * (binary_image.shape[0] / 3)

    # Flip y-coordinate for correct orientation and offset by the scaled curvature
    curvature_y_values = (binary_image.shape[0] - (contour_array[:, 1] + curvature_scaled))

    ax.plot(contour_array[:, 0], curvature_y_values, color='red')
    ax.set_title('ABE')
    ax.axis('on')
    ax.set_aspect('equal')

    ax.text(0.5, -0.1, f'ABE: {abe_value:.4f}', ha='center', transform=ax.transAxes)

    ax.set_xlim([0, binary_image.shape[1]])
    ax.set_ylim([binary_image.shape[0], 0])
    ax.invert_yaxis()


def plot_eccentricity(ax, contour, cx, cy, eccentricity):
    ax.axis('on')
    ax.set_title('Eccentricity')

    contour_array = contour.squeeze()

    ax.plot(contour_array[:, 0], contour_array[:, 1], 'k-')

    ax.text(0.5, -0.1, f'Eccentricity: {eccentricity:.2f}',
                             ha='center', transform=ax.transAxes)
    
    ax.set_xlim([np.min(contour_array[:, 0]), np.max(contour_array[:, 0])])
    ax.set_ylim([np.min(contour_array[:, 1]), np.max(contour_array[:, 1])])

    ax.invert_yaxis()
    ax.set_aspect('equal')


def plot_mbr(ax, contour, rect):
    # Get box points and draw the rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    ax.plot(contour[:, 0], contour[:, 1], 'k-')
    
    # Draw MBR
    ax.plot([box[0][0], box[1][0]], [box[0][1], box[1][1]], 'r-')
    ax.plot([box[1][0], box[2][0]], [box[1][1], box[2][1]], 'r-')
    ax.plot([box[2][0], box[3][0]], [box[2][1], box[3][1]], 'r-')
    ax.plot([box[3][0], box[0][0]], [box[3][1], box[0][1]], 'r-')
    
    ax.set_aspect('equal')
    ax.set_title('MBR')


def plot_circularity_ratio(ax, contour, circularity_ratio):
    y_flipped = np.max(contour[:, 1]) - contour[:, 1]
    
    (x, y), radius = cv2.minEnclosingCircle(np.column_stack((contour[:, 0], y_flipped)))
    center = (int(x), int(y))
    
    ax.plot(contour[:, 0], y_flipped, 'k-')
    
    circle = plt.Circle((x, y), radius, color='r', fill=False)
    ax.add_artist(circle)
    
    ax.text(0.5, 0.1, f'CR: {circularity_ratio:.4f}', ha='center', transform=ax.transAxes)

    ax.set_xlim([np.min(contour[:, 0]), np.max(contour[:, 0])])
    ax.set_ylim([0, np.max(contour[:, 1])]) 
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


def plot_ellipse_features(ax, contour, EV, EM):
    # Flip the y-coordinates to correct for the coordinate system difference
    max_y = np.max(contour[:, 1])
    contour[:, 1] = max_y - contour[:, 1]
    
    ellipse = cv2.fitEllipse(contour)

    ax.plot(contour[:, 0], contour[:, 1], 'k-') 

    ellipse_center = (int(ellipse[0][0]), max_y - int(ellipse[0][1]))
    ellipse_axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
    ellipse_angle = ellipse[2]

    ellipse_artist = patches.Ellipse(xy=ellipse_center, width=2*ellipse_axes[0], height=2*ellipse_axes[1],
                                     angle=ellipse_angle, edgecolor='r', fc='None', lw=2)
    ax.add_patch(ellipse_artist)

    ax.text(0.5, -0.1, f'EV: {EV:.4f}', ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.2, f'EM: {EM:.4f}', ha='center', transform=ax.transAxes)

    ax.set_xlim([np.min(contour[:, 0]), np.max(contour[:, 0])])
    ax.set_ylim([0, max_y])
    ax.set_aspect('equal', 'box')
    ax.axis('off') 


def plot_solidity(ax, contour, solidity):
    ax.plot(contour[:, 0], contour[:, 1], 'k-')

    hull = cv2.convexHull(contour)
    hull = hull.squeeze()
    ax.plot(hull[:, 0], hull[:, 1], 'r--')

    ax.text(0.5, -0.1, 'Solidity: {:.4f}'.format(solidity), ha='center', transform=ax.transAxes)

    ax.set_title('Solidity')
    ax.axis('off')


# Call all geometric shape feature calculation functions
def get_geometric_shape_features(contour, cx, cy, binary_image_path):

    all_contour_features = []
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

    # Calculate curvature, CDF, COG for each point in the contour
    for i, point in enumerate(contour_array):
        xi, yi = point
        complex_coordinate = (xi - cx) + 1j * (yi - cy)
        complex_coordinates.append(complex_coordinate)
        cdf.append(np.abs(complex_coordinate))
        curvature_list.append(calculate_curvature(dx[i], dy[i], d2x[i], d2y[i]))
        if i < len(contour_array) - 1:
            area_function.append(calculate_triangle_area(contour_array[i], contour_array[i+1], (cx, cy)))
    
    # Calculate p, q, and r (where r is the sum of y^2)
    p, q = calculate_pq(contour_array - np.array([[cx, cy]]))
    r = np.sum((contour_array[:, 1] - cy)**2)

    # Calculate phi that minimizes the AMI
    phi = calculate_phi(p, q, r)

    # Calculate AMI using phi
    I = calculate_AMI(p, q, r, phi)

    # Second derivative check for alpha calculation
    d2I_dphi2 = 2 * (p - r) * np.cos(2 * phi) + 2 * q * np.sin(2 * phi)
    alpha = phi + np.pi/2 if d2I_dphi2 < 0 else phi

    # Assuming curvature_list is a list of curvature values for the contour
    abe_value = np.mean(curvature_list)

    # Calculate the eccentricity
    eccentricity_value = calculate_eccentricity(contour_array)

    # Calculate MBR
    mbr_center, (mbr_width, mbr_height), mbr_angle, E, Elo = calculate_mbr_and_elongation(contour)

    # Calculate the area and perimeter of the contour
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Calculate and visualize the CR
    circularity_ratio, combined_visual = calculate_and_visualize_circularity(contour, binary_image_path)

    # Calculate additional moments
    m00, m10, m01, m20, m02, m11 = calculate_moments(contour)

    # Calculate covariance matrix from the moments
    covariance_matrix = np.array([[m20, m11], [m11, m02]])

    # Calculate EV
    ev_value = calculate_ev(contour_array, covariance_matrix)

    # Calculate EM
    em_value = calculate_em(m00, m20, m02, m11)

    # Calculate Solidity
    solidity, hull = calculate_solidity(contour_array) 
    
    geom_features = {
        "abe": abe_value,
        "alpha": alpha,
        "eccentricity": eccentricity_value,
        "mbr_center": mbr_center,
        "mbr_width": mbr_width,
        "mbr_height": mbr_height,
        "mbr_angle": mbr_angle,
        "mbr_E": E,
        "elongation": Elo,
        "circularity_ratio": circularity_ratio,
        "EV": ev_value,
        "EM": em_value,
        "hull": hull,
        "solidity": solidity
        
    }

    return geom_features
