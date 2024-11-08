import cv2
import glob
from extractor import Frame, denormalize, match_frames, add_ones, filter_moving_keypoints,extract
import numpy as np
from mapCreator import Map, Point
import g2o  
from scipy.spatial.transform import Rotation as R

W, H = 1920 // 2,  1080 // 2
# F = 270
F = 450
K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])


Kinv = np.linalg.inv(K)

mapp = Map()

# En la función triangulate, añadir la restricción para las coordenadas Y
def triangulate(pose1, pose2, pts1, pts2):
    ret = np.zeros((pts1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    for i, p in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        A = np.zeros((4, 4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]

    # Control de calidad para evitar divisiones por casi cero
    valid = np.abs(ret[:, 3]) > 1e-5  # Evitar coordenadas homogéneas no válidas
    ret = ret[valid]  # Solo mantenemos los puntos válidos
    ret /= ret[:, 3:]  # Normalizamos los puntos

    # Restringir la coordenada Y a 0 para mantener el plano XZ
    ret[:, 1] = 0

    return ret


displacement_accumulated= 0.0
pose_history = []
initial_direction = np.array([0, 0, 1])  # Dirección de referencia (ej. hacia adelante en el eje Z)

def reset():
    global pose_history, mapp
    pose_history = []
    mapp.full_reset()  # Limpia todos los recursos asociados con el mapa


def restrict_to_2d_rotation(pose, max_xy_rotation=5):
    """Restringe la rotación de la pose principalmente al eje Z (plano 2D), permitiendo un pequeño margen en X e Y."""
    # Convertimos la matriz de rotación a ángulos de Euler (yaw, pitch, roll)
    rotation_matrix = pose[:3, :3]
    euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)

    # Limitar rotación en X y Y a un máximo de `max_xy_rotation` grados
    euler_angles[0] = np.clip(euler_angles[0], -max_xy_rotation, max_xy_rotation)
    euler_angles[1] = np.clip(euler_angles[1], -max_xy_rotation, max_xy_rotation)

    # Convertimos de nuevo a matriz de rotación con los ángulos ajustados
    new_rotation = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()

    # Actualizamos la pose con la nueva rotación
    pose[:3, :3] = new_rotation
    return pose

def restrict_to_2d_translation(pose):
    """Restringe la traslación a solo el plano XZ."""
    pose[1, 3] = 0  # Fijamos la altura (Y) a cero
    return pose

def check_for_pose_reset(current_pose, previous_pose, threshold=10.0):
    # Comparar la traslación y rotación entre dos poses
    translation_diff = np.linalg.norm(current_pose[:3, 3] - previous_pose[:3, 3])
    rotation_diff = np.linalg.norm(current_pose[:3, :3] - previous_pose[:3, :3])

    if translation_diff > threshold or rotation_diff > threshold:
        return True  # Requiere reinicialización
    return False


def interpolate_pose(prev_pose, current_pose, factor=0.3):
    """
    Interpola entre la pose anterior y la actual para suavizar la transición.
    factor: El peso de la interpolación, entre 0 (totalmente la anterior) y 1 (totalmente la actual).
    """
    # Interpolación de traslación
    interpolated_translation = (1 - factor) * prev_pose[:3, 3] + factor * current_pose[:3, 3]
    
    # Interpolación de rotación con cuaterniones
    prev_rotation = R.from_matrix(prev_pose[:3, :3])
    current_rotation = R.from_matrix(current_pose[:3, :3])

    # Convertir a cuaterniones
    q1 = prev_rotation.as_quat()
    q2 = current_rotation.as_quat()

    # Interpolación lineal y normalización para aproximar SLERP
    interpolated_quat = (1 - factor) * q1 + factor * q2
    interpolated_quat /= np.linalg.norm(interpolated_quat)  # Normalizar

    # Convertir el cuaternión interpolado a matriz de rotación
    interpolated_rotation = R.from_quat(interpolated_quat).as_matrix()
    
    # Construir pose interpolada
    interpolated_pose = np.eye(4)
    interpolated_pose[:3, :3] = interpolated_rotation
    interpolated_pose[:3, 3] = interpolated_translation
    
    return interpolated_pose

def process_frame(img, obstacle_detection):
    global pose_history, initial_direction

    img = cv2.resize(img, (W, H))
    frame = Frame(mapp, img, K, obstacle_detection)

    if frame.id == 0:
        # Inicializar la pose con la dirección de referencia
        frame.pose[:3, 2] = initial_direction
        pose_history.append(frame.pose)
        return

    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    # Emparejamiento de puntos clave entre frames
    idx1, idx2, Rt = match_frames(f1, f2)
    # Aplicar restricciones de rotación y traslación 2D

     # Pose estimada en el nuevo frame sin suavizar
    raw_pose = np.dot(Rt, f2.pose)
    
    # Suavizar la pose con la interpolación de la anterior
    smoothed_pose = interpolate_pose(f2.pose, raw_pose, factor=0.5)

    f1.pose = smoothed_pose
    f1.pose = restrict_to_2d_rotation(f1.pose)  # Restringir rotación
    f1.pose = restrict_to_2d_translation(f1.pose)  # Restringir traslación
    # Filtrar los puntos clave en movimiento
    #idx1, idx2 = filter_moving_keypoints(f1, f2, idx1, idx2, threshold=0.01)

    # Verificación de la dirección de las características
    directions_f1 = np.array([f1.kps[idx].angle for idx in idx1])
    directions_f2 = np.array([f2.kps[idx].angle for idx in idx2])
    direction_diffs = np.abs(directions_f1 - directions_f2)
    mean_direction_diff = np.mean(direction_diffs)

   # Ajuste de la rotación del frame actual
    if mean_direction_diff > 45:
        print("Interpolando rotación para suavizar el movimiento")
        frame.kps, frame.des, frame.angles = extract(img, nfeatures=5000)  # Aumentar el número de características
        if len(pose_history) >= 3:
            rotations = [R.from_matrix(pose[:3, :3]) for pose in pose_history[-3:]]
            average_rotation = R.mean(rotations)  # Suavizar más la rotación
            f1.pose[:3, :3] = average_rotation.as_matrix()

        else:
            f1.pose = np.dot(Rt, f2.pose)
    else:
        f1.pose = np.dot(Rt, f2.pose)

       # Suavización adicional para la traslación
    max_translation_distance = 1.0  # Limitar la distancia máxima permitida entre fotogramas
    translation_direction = f1.pose[:3, 3] - f2.pose[:3, 3]
    distance_moved = np.linalg.norm(translation_direction)

    # Verificación de cambios bruscos en la dirección y suavización de la traslación
    translation_direction = f1.pose[:3, 3] - f2.pose[:3, 3]  # Traslación entre las poses
    camera_direction = f1.pose[:3, 2]  # Eje Z de la cámara (hacia adelante)

    # Calcular el ángulo completo entre la traslación y la orientación de la cámara en el plano XZ (giros)
    translation_direction_xy = translation_direction[[0, 2]]  # Solo tomamos los ejes X y Z (ignoramos Y)
    camera_direction_xy = camera_direction[[0, 2]]

    # Calcular el ángulo entre la dirección de la cámara y la traslación en el plano XZ
    cos_angle_xy = np.dot(translation_direction_xy, camera_direction_xy) / (
        np.linalg.norm(translation_direction_xy) * np.linalg.norm(camera_direction_xy)
    )
    angle_xy = np.degrees(np.arccos(np.clip(cos_angle_xy, -1.0, 1.0)))

    # Si el ángulo es mayor de 70 grados, corregimos la traslación
    if angle_xy > 70:  
        print(f"Corrigiendo traslación: ángulo de {angle_xy:.2f} grados en el plano XZ")
        corrected_translation = camera_direction_xy * np.linalg.norm(translation_direction_xy)
        f1.pose[:3, 3] = f2.pose[:3, 3] + np.array([corrected_translation[0], 0, corrected_translation[1]])  # Corrección en XZ
    
        # Si el ángulo es mayor de 70 grados, corregimos la traslación
    if check_for_pose_reset(f1.pose, f2.pose, threshold=10.0):
        print(f"Reinicializando la pose en el frame {frame.id} debido a gran desvío")
        f1.pose = np.copy(f2.pose)  # Reinicializamos la pose actual
   
    # Limitar la distancia entre fotogramas consecutivos
    if distance_moved > max_translation_distance:
        print(f"Limitando traslación: distancia de {distance_moved:.2f} excede el máximo de {max_translation_distance:.2f}")
        f1.pose[:3, 3] = f2.pose[:3, 3] + (translation_direction / distance_moved) * max_translation_distance


    # Ajustamos la altura de la cámara al plano XZ (puede afectar la visualización)
    f1.pose[1, 3] = 0  # Esto fija la altura, pero puede estar causando el error si afecta la rotación

    # Triangulación de puntos para obtener coordenadas 3D
    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
    pts4d /= pts4d[:, 3:]

    # Filtro básico de puntos: eliminamos los puntos detrás de la cámara y sin suficiente parallax
    good_pts4d = (pts4d[:, 2] > 0)

    camera_position = np.linalg.inv(f1.pose)[:3, 3]
    distances_to_camera = np.linalg.norm(pts4d[:, :3] - camera_position, axis=1)
    min_distance = 0.5
    valid_distances = distances_to_camera > min_distance
    good_pts4d &= valid_distances

    points = [pts4d[i] for i in range(len(pts4d)) if good_pts4d[i]]

    # Ejecutar Bundle Adjustment cada 5 frames
    if frame.id % 2 == 0 or mean_direction_diff > 20:
        optimizer = bundle_adjustment(mapp.frames, points)
        print(f"Ejecutando Bundle Adjustment en el frame {frame.id}")
    else:
        print(f"Saltando Bundle Adjustment en el frame {frame.id}")

    # Añadir los puntos triangulados al mapa
    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, i)
        pt.add_observation(f2, i)

    # Dibujar los puntos y líneas entre correspondencias
    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)
        cv2.circle(img, (u1, v1), 2, (77, 243, 255))
        cv2.line(img, (u1, v1), (u2, v2), (255, 0, 0))
        cv2.circle(img, (u2, v2), 2, (204, 77, 255))

    # Mostrar el mapa y la imagen procesada
    mapp.display()
    mapp.display_image(img)

# Función para Bundle Adjustment
def bundle_adjustment(frames, points):
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)

    # Paso 1: Añadir las poses de las cámaras como vértices
    for i, frame in enumerate(frames):
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(i)
        v_se3.set_estimate(g2o.SE3Quat(frame.pose[:3, :3], frame.pose[:3, 3]))
        optimizer.add_vertex(v_se3)

    # Paso 2: Añadir los puntos 3D como vértices
    for i, point in enumerate(points):
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(len(frames) + i)
        v_p.set_estimate(point[:3])
        v_p.set_marginalized(True)
        optimizer.add_vertex(v_p)

    # Paso 3: Ejecutar la optimización
    optimizer.initialize_optimization()
    optimizer.optimize(10)  # Número de iteraciones

    # Paso 4: Actualizar las poses optimizadas en los frames
    for i, frame in enumerate(frames):
        v_se3 = optimizer.vertex(i)
        frame.pose[:3, :3] = v_se3.estimate().rotation().matrix()
        frame.pose[:3, 3] = v_se3.estimate().translation()

    # Paso 5: Actualizar los puntos 3D optimizados
    for i, point in enumerate(points):
        v_p = optimizer.vertex(len(frames) + i)
        point[:3] = v_p.estimate()

    print(f"Bundle Adjustment: Optimización completada.")

    return optimizer
