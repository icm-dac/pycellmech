'''
======================================================================
 Title:                   PYCELLMECH - POLYGONAL SHAPES
 Creating Author:         Janan Arslan
 Creation Date:           02 MAR 2024
 Latest Modification:     25 MAR 2024
 Modification Author:     Janan Arslan
 E-mail:                  janan.arslan@icm-institute.org
 Version:                 1.0
======================================================================


Polygonal Shape Features extracted as part of the pycellmech. 

'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import atan2, cos, sin, sqrt, pi
import cv2
from sklearn.cluster import KMeans


### --- CALCULATIONS --- ###

def distance_to_segment(point, start, end):
    # Compute the distance of 'point' from the line segment 'start' to 'end'
    num = np.abs((end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1]))
    den = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    return num / den if den != 0 else 0


def distance_threshold_method(contour, threshold):
    segments = []
    start_index = 0

    for i in range(1, len(contour)):
        max_distance = 0
        # Find the point with the max distance to the current line segment
        for j in range(start_index, i + 1):
            distance = distance_to_segment(contour[j], contour[start_index], contour[i])
            if distance > max_distance:
                max_distance = distance
        # If max distance exceeds threshold, the current segment ends
        if max_distance > threshold:
            segments.append((contour[start_index], contour[i - 1]))
            start_index = i - 1

    if not segments or not np.array_equal(segments[-1][1], contour[-1]):
        segments.append((contour[start_index], contour[-1]))

    return segments


def calculate_turn_angle(p1, p2, p3):
    # Calculate the turn angle at p2 given points p1, p2, and p3
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    cross_product = np.cross(v1, v2)
    angle = np.arctan2(cross_product, dot_product)
    return angle

def calculate_significance_measure(L1, L2, total_length):
    # Assuming L1 and L2 are given as (x, y) coordinates of the endpoints
    S_L1 = np.linalg.norm(L1[1] - L1[0])
    S_L2 = np.linalg.norm(L2[1] - L2[0])
    beta = calculate_turn_angle(L1[0], L1[1], L2[1])
    M = (beta * S_L1) / (S_L1 + S_L2) if total_length != 0 else 0
    return M

# Define the PEVD function
def polygon_evolution_by_vertex_deletion(contour, threshold):
    simplified_contour = contour[:]
    total_length = sum([np.linalg.norm(simplified_contour[i] - simplified_contour[i-1]) 
                        for i in range(1, len(simplified_contour))])
    
    def M(L1, L2, total_length):
        S_L1 = np.linalg.norm(L1[1] - L1[0])
        S_L2 = np.linalg.norm(L2[1] - L2[0])
        beta = calculate_turn_angle(L1[0], L1[1], L2[1])
        return abs(beta * S_L1) / (S_L1 + S_L2) if total_length else 0

    change = True
    while change:
        change = False
        new_contour = [simplified_contour[0]]
        for i in range(1, len(simplified_contour) - 2):
            L1 = (simplified_contour[i], simplified_contour[i+1])
            L2 = (simplified_contour[i+1], simplified_contour[i+2])
            if M(L1, L2, total_length) < threshold:
                # Skip adding L1's endpoint to the new contour
                change = True
            else:
                new_contour.append(simplified_contour[i+1])
        new_contour.append(simplified_contour[-1])
        simplified_contour = new_contour

    return np.array(simplified_contour)


def splitting_method(contour, error_tolerance):
    def recursive_split(points, start, end, tolerance, segments_sm):
        max_distance = 0
        farthest_index = -1
        
        # Find the point farthest from the line segment (start, end)
        for i in range(start+1, end):
            distance = point_line_distance(points[i], points[start], points[end])
            if distance > max_distance:
                max_distance = distance
                farthest_index = i
        
        # If the farthest point is within the error tolerance, no further split is needed
        if max_distance < tolerance:
            segments_sm.append((start, end))
        else:
            recursive_split(points, start, farthest_index, tolerance, segments_sm)
            recursive_split(points, farthest_index, end, tolerance, segments_sm)
    
    def point_line_distance(point, start, end):
        if np.all(np.equal(start, end)):
            return np.linalg.norm(point - start)
        
        return np.abs(np.cross(end - start, start - point)) / np.linalg.norm(end - start)

    segments_sm = []
    recursive_split(contour, 0, len(contour)-1, error_tolerance, segments_sm)
    return segments_sm


def calculate_det(p, q, r):
    matrix = np.array([[p[0], p[1], 1],
                       [q[0], q[1], 1],
                       [r[0], r[1], 1]])
    return np.linalg.det(matrix)

def is_convex(p, q, r):
    return calculate_det(p, q, r) > 0

def label_vertices(contour):
    labels = []
    n = len(contour)
    for i in range(n):
        if is_convex(contour[i-1], contour[i], contour[(i+1) % n]):
            labels.append('W')
        else:
            labels.append('B')
    return labels


def reflect_across_line(point, line_start, line_end):
    # Simplified: Reflect across the x-axis
    return np.array([point[0], -point[1]])

def find_mirrors(contour, labels):
    mirrors = []
    for i, label in enumerate(labels):
        if label == 'B':  
            line_start = contour[i - 1]
            line_end = contour[(i + 1) % len(contour)]
            mirror_point = reflect_across_line(contour[i], line_start, line_end)
            mirrors.append(mirror_point)
        else:
            mirrors.append(contour[i])
    return np.array(mirrors)


def mpp_algorithm(vertices, mirrors, is_convex):
    V0 = vertices[0]
    VL = V0
    WC = BC = V0
    mpp_vertices = [V0]

    for i in range(1, len(vertices)):
        VC = vertices[i]
        if is_convex[i]:
            WC = VC
            candidate_mpp_vertex = WC
        else:
            BC = mirrors[i]
            candidate_mpp_vertex = BC

        if calculate_det(VL, WC, VC) > 0 and calculate_det(VL, BC, VC) <= 0:
            mpp_vertices.append(candidate_mpp_vertex)
            VL = candidate_mpp_vertex

    if not np.array_equal(VL, V0):
        mpp_vertices.append(V0)

    return np.array(mpp_vertices)


def simplify_contour_with_kmeans(contour, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(contour)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    
    line_segments = []
    
    for i in range(num_clusters):
        cluster_points = contour[labels == i]
        
        [vx, vy, x, y] = cv2.fitLine(cluster_points, cv2.DIST_L2, 0, 0.01, 0.01)
        
        start_x = np.min(cluster_points[:, 0])
        end_x = np.max(cluster_points[:, 0])
        start_y = y + (start_x - x) * vy / vx
        end_y = y + (end_x - x) * vy / vx
        line_segments.append(((start_x, start_y), (end_x, end_y)))
    
    return cluster_centers, line_segments



### --- PLOT FUNCTIONS --- ###


def plot_DTM(ax, contour, segments):
    ax.axis('on')
    
    contour = np.squeeze(np.array(contour))
    
    ax.plot(contour[:, 0], contour[:, 1], 'k-', label='Original Contour')

    for start, end in segments:
        ax.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=2, label='Segment')

    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        ax.legend()

    ax.set_xlim([np.min(contour[:, 0]), np.max(contour[:, 0])])
    ax.set_ylim([np.min(contour[:, 1]), np.max(contour[:, 1])])
    
    ax.set_title('DTM')
    
    ax.set_aspect('equal')
    

def plot_PEVD(ax, contour, reduced_vertices):
    ax.plot(contour[:, 0], -contour[:, 1], 'k-', label='Original Polygon')

    ax.plot(reduced_vertices[:, 0], reduced_vertices[:, 1], 'r-', label='Reduced Polygon')

    ax.set_aspect('equal')

    ax.axis('on')
    ax.set_title('PEVD')

    ax.set_xlim([np.min(contour[:, 0]), np.max(contour[:, 0])])
    ax.set_ylim([np.max(contour[:, 1]), np.min(contour[:, 1])])


def plot_SM(ax, contour, segments_sm):
    ax.plot(contour[:, 0], contour[:, 1], 'k-', label='Original Contour')

    for start, end in segments_sm:
        ax.plot(contour[start:end+1, 0], contour[start:end+1, 1], 'r-', linewidth=2)

    ax.set_aspect('equal')

    ax.set_title('SM')


def plot_MPP(ax, contour, mpp_vertices):
    ax.plot(contour[:, 0], -contour[:, 1], 'k-', label='Original Contour')

    mpp_vertices_plot = np.array(mpp_vertices)
    ax.plot(mpp_vertices_plot[:, 0], -mpp_vertices_plot[:, 1], 'b-o', label='MPP', linewidth=2, markersize=5)

    ax.set_aspect('equal')

    ax.axis('on')
    ax.set_title('MPP')

    ax.set_xlim([np.min(contour[:, 0]), np.max(contour[:, 0])])
    ax.set_ylim([np.min(contour[:, 1]), np.max(contour[:, 1])])


def plot_KMeans(ax, contour, cluster_centers, line_segments):
    ax.plot(contour[:, 0], contour[:, 1], 'b.', alpha=0.3)  
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x')

    for segment in line_segments:
        ax.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 'g-', linewidth=2)

    ax.set_aspect('equal')
    ax.set_title("KMeans")
    

# Call all polygonal shape feature calculation functions
def get_polyognal_shape_features(contour, cx, cy, binary_image_path):

    all_contour_features = []
    dtm = []
    pevd = []
    sm = []
    mpp = []
    km_cc = []
    km_ls = []


    # Simplify the contour array
    contour_array = contour.squeeze()

    x = contour_array[:, 0]
    y = contour_array[:, 1]

    # Calculate gradients
    dx = np.gradient(x)
    dy = np.gradient(y)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)


    # DTM Calculations
    threshold_segments = 10
    segments = distance_threshold_method(contour_array, threshold_segments)

    # PEVD Calculations
    threshold_vertices = 0.2    

    # Ensure the y-coordinates are flipped for image coordinate system
    contour_array_pevd = np.copy(contour_array)
    contour_array_pevd[:, 1] = max(contour_array_pevd[:, 1]) - contour_array_pevd[:, 1]

    pevd_result = polygon_evolution_by_vertex_deletion(contour_array_pevd, threshold_vertices)
    
    # Ensure the y-coordinates are flipped for PEVD result
    pevd_result[:, 1] = max(pevd_result[:, 1]) - pevd_result[:, 1]

    # Calculate Splitting Method
    error_tolerance = 0.5
    segments_sm = splitting_method(contour_array, error_tolerance)

    ## Calculate MPP
    # Assume labeling of vertices as convex ('W') or concave ('B') is done
    labels = label_vertices(contour_array)

    # Find mirrors of B vertices
    mirrors = find_mirrors(contour_array, labels)

    # Determine convexity as a boolean array
    is_convex_array = np.array([label == 'W' for label in labels])

    # MPP Calculations
    mpp_result = mpp_algorithm(contour_array, mirrors, is_convex_array)

    # Calculate KMeans
    # Specify the number of clusters for the K-means algorithm
    num_clusters = 7

    cluster_centers, line_segments = simplify_contour_with_kmeans(contour_array, num_clusters)

    
    
    poly_features = {
        "dtm": segments,
        "pevd": pevd_result,
        "sm": segments_sm,
        "mpp": mpp_result,
        "km_cc": cluster_centers,
        "km_ls": line_segments
        
    }

    return poly_features
