import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import g2o
from scipy.spatial import KDTree
import torch

# Determinar el dispositivo (GPU o CPU) para procesamiento
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def add_ones(points):
    """
    Convierte los puntos 2D a coordenadas homogéneas agregando una columna de unos.
    """
    return np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

# Matriz de identidad para la transformación inicial
IRt = np.eye(4)

def extractPose(F_matrix):
    """
    Extrae la pose (rotación y traslación) a partir de la matriz fundamental F_matrix.
    """
    W_matrix = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U_matrix, singular_values, Vt_matrix = np.linalg.svd(F_matrix)
    assert np.linalg.det(U_matrix) > 0
    if np.linalg.det(Vt_matrix) < 0:
        Vt_matrix *= -1
    rotation_matrix = np.dot(np.dot(U_matrix, W_matrix), Vt_matrix)
    if np.sum(rotation_matrix.diagonal()) < 0:
        rotation_matrix = np.dot(np.dot(U_matrix, W_matrix.T), Vt_matrix)
    translation_vector = U_matrix[:, 2]
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    print(singular_values)
    return transformation_matrix

# Cargar el modelo YOLO preentrenado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)

def detect_objects(image):
    """
    Detecta objetos en la imagen usando YOLOv5 y devuelve los bounding boxes.
    """
    results = model(image)
    bounding_boxes = results.xyxy[0].cpu().numpy()  # formato [x1, y1, x2, y2, confianza, clase]
    return bounding_boxes

def filter_keypoints_with_safe_distance(keypoints, bounding_boxes, obstacle_classes=[0, 2], safe_distance=50):
    """
    Filtra los puntos clave que están cerca de los bounding boxes de los objetos considerados obstáculos.

    :param keypoints: Lista de puntos clave en la imagen.
    :param bounding_boxes: Bounding boxes detectados por YOLO.
    :param obstacle_classes: Clases de objetos considerados obstáculos.
    :param safe_distance: Distancia mínima para mantener los puntos clave alejados del obstáculo.
    :return: Lista de puntos clave filtrados.
    """
    filtered_keypoints = []
    for point in keypoints:
        x_coord, y_coord = point.pt  # Coordenadas del punto clave
        near_obstacle = False
        for box in bounding_boxes:
            x_min, y_min, x_max, y_max, _, obj_class = box
            if obj_class in obstacle_classes:
                x_min_safe, y_min_safe = x_min - safe_distance, y_min - safe_distance
                x_max_safe, y_max_safe = x_max + safe_distance, y_max + safe_distance
                if x_min_safe <= x_coord <= x_max_safe and y_min_safe <= y_coord <= y_max_safe:
                    near_obstacle = True
                    break
        if not near_obstacle:
            filtered_keypoints.append(point)

    return filtered_keypoints

def extract(image, enable_filter):
    """
    Extrae características usando ORB y filtra puntos clave basados en obstáculos detectados.
    """
    orb_detector = cv2.ORB_create(
        nfeatures=2000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20
    )

    # Convertir imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detección de puntos clave inicial
    initial_points = cv2.goodFeaturesToTrack(gray_image, 2000, qualityLevel=0.01, minDistance=10)

    if initial_points is None:
        return [], None
    keypoints = [cv2.KeyPoint(f[0][0], f[0][1], 20) for f in initial_points]
    
    if enable_filter:
        # Detectar objetos con YOLO
        bounding_boxes = detect_objects(image)
        keypoints = filter_keypoints_with_safe_distance(keypoints, bounding_boxes, obstacle_classes=[0, 2], safe_distance=5)

    keypoints = filter_redundant_keypoints(keypoints, threshold=5.0)
    keypoints, descriptors = orb_detector.compute(gray_image, keypoints)

    # Extraer ángulos de los puntos clave
    angles = [kp.angle for kp in keypoints] if keypoints is not None and descriptors is not None else []

    return keypoints, descriptors, angles

def prioritize_central_keypoints(keypoints, img_shape, center_weight=0.5):
    """
    Prioriza puntos clave ubicados cerca del centro de la imagen.

    :param keypoints: Lista de puntos clave detectados.
    :param img_shape: Dimensiones de la imagen (alto, ancho).
    :param center_weight: Factor para priorizar el centro.
    :return: Lista priorizada de puntos clave.
    """
    height, width = img_shape[:2]
    center_x, center_y = width / 2, height / 2

    prioritized_keypoints = sorted(
        keypoints,
        key=lambda kp: (1 - center_weight) + center_weight * (1 - np.hypot(kp.pt[0] - center_x, kp.pt[1] - center_y) / max(height, width))
    )

    return prioritized_keypoints[:len(keypoints)]

def filter_redundant_keypoints(keypoints, threshold=5.0):
    """Filtra puntos clave redundantes según la proximidad."""
    if len(keypoints) == 0:
        return keypoints

    coords = np.array([kp.pt for kp in keypoints])
    kd_tree = KDTree(coords)
    filtered_indices = set()

    for i, coord in enumerate(coords):
        if i in filtered_indices:
            continue
        neighbors = kd_tree.query_ball_point(coord, threshold)
        filtered_indices.add(i)

    return [keypoints[i] for i in filtered_indices if len(neighbors) < 8]

def filter_moving_keypoints(f1, f2, idx1, idx2, pose_history, threshold_velocity=0.003, threshold_dispersion=0.8):
    """
    Filtra puntos clave que puedan pertenecer a objetos en movimiento basados en la velocidad relativa.

    :param f1: Frame actual
    :param f2: Frame anterior
    :param idx1: Índices de puntos clave en f1
    :param idx2: Índices de puntos clave en f2
    :param pose_history: Historial de poses recientes
    :param threshold_velocity: Umbral para velocidad
    :param threshold_dispersion: Umbral para dispersión en múltiples frames
    :return: Índices filtrados de puntos clave (idx1, idx2)
    """
    moving_indices1, moving_indices2 = [], []
    velocities = []

    for id1, id2 in zip(idx1, idx2):
        point1 = f1.pts[id1]
        point2 = f2.pts[id2]
        velocity = np.linalg.norm(point2 - point1)
        velocities.append((id1, id2, velocity))

    velocities = [(id1, id2, vel) for id1, id2, vel in velocities if vel < threshold_velocity]

    if len(pose_history) > 2:
        points = np.array([f1.pts[id1] for id1, _, _ in velocities])
        dispersions = []
        for pose in pose_history[-3:]:
            transformed_points = np.dot(pose[:3, :3], points.T).T + pose[:3, 3]
            dispersions.append(np.var(transformed_points, axis=0))

        mean_dispersion = np.mean(dispersions, axis=0)
        velocities = [(id1, id2, vel) for (id1, id2, vel), disp in zip(velocities, mean_dispersion) if np.linalg.norm(disp) < threshold_dispersion]

    moving_indices1 = [id1 for id1, _, _ in velocities]
    moving_indices2 = [id2 for _, id2, _ in velocities]

    return moving_indices1, moving_indices2

def normalize(Kinv, points):
    """
    Normaliza puntos 2D usando la matriz intrínseca inversa de la cámara.
    """
    return np.dot(Kinv, add_ones(points).T).T[:, 0:2]

def denormalize(K, point):
    """
    Convierte un punto de coordenadas normalizadas a coordenadas de imagen.
    """
    result = np.dot(K, [point[0], point[1], 1.0])
    result /= result[2]
    return int(round(result[0])), int(round(result[1]))

def match_frames(f1, f2):
    """
    Empareja puntos clave entre dos frames usando el criterio de ratio de Lowe y un umbral dinámico de distancia.
    """
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf_matcher.knnMatch(f1.des, f2.des, k=2)

    matched_points = []
    indices1, indices2 = [], []

    translation_dist = np.linalg.norm(f2.pose[:3, 3] - f1.pose[:3, 3])
    distance_threshold = 0.01 * (1 + translation_dist)
    ratio_test_threshold = max(0.75, min(0.9, 0.8 / (1 + translation_dist)))
    for m, n in matches:
        if m.distance < ratio_test_threshold * n.distance:
            point1 = f1.pts[m.queryIdx]
            point2 = f2.pts[m.trainIdx]
            if np.linalg.norm(point1 - point2) < distance_threshold:
                scale1 = f1.kps[m.queryIdx].size
                scale2 = f2.kps[m.trainIdx].size
                scale_ratio = scale1 / scale2
                if 0.9 < scale_ratio < 1.1:
                    indices1.append(m.queryIdx)
                    indices2.append(m.trainIdx)
                    matched_points.append((point1, point2))

    assert len(matched_points) >= 8, "No se encontraron suficientes coincidencias"

    matched_points = np.array(matched_points)
    indices1 = np.array(indices1)
    indices2 = np.array(indices2)

    model, inliers = ransac(
        (matched_points[:, 0], matched_points[:, 1]), 
        FundamentalMatrixTransform, 
        min_samples=8, 
        residual_threshold=0.005 * (1 + np.log(1 + len(matched_points) / 100)), 
        max_trials=5000
    )

    matched_points = matched_points[inliers]
    indices1 = indices1[inliers]
    indices2 = indices2[inliers]

    Rt = extractPose(model.params)

    return indices1, indices2, Rt

class Frame(object):
    def __init__(self, mapp, img, K, enable_filter):
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        self.pose = IRt

        self.id = len(mapp.frames)
        self.enable_filter = enable_filter
        mapp.frames.append(self)

        kps, self.des, self.angles = extract(img, self.enable_filter)
        self.kps = kps
        
        if self.des is not None:
            self.pts = normalize(self.Kinv, np.array([(kp.pt[0], kp.pt[1]) for kp in self.kps]))
