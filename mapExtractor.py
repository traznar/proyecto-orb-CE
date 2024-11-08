import numpy as np
import pangolin
import OpenGL.GL as gl

# Clase para visualizar el mapa
class MapVisualizer(object):
    def __init__(self, poses):
        self.poses = poses
        self.grid_min, self.grid_max = self.calculate_grid_size()

    def calculate_grid_size(self):
        # Calcular el rango de la cuadrícula basado en la trayectoria
        x_min = np.min(self.poses[:, 0, 3])  # Mínimo valor de X
        x_max = np.max(self.poses[:, 0, 3])  # Máximo valor de X
        z_min = np.min(self.poses[:, 2, 3])  # Mínimo valor de Z
        z_max = np.max(self.poses[:, 2, 3])  # Máximo valor de Z
        return (x_min, z_min), (x_max, z_max)

    def create_viewer(self):
        # Iniciar la visualización
        self.viewer_init(1280, 720)
        # Bucle de visualización
        while not pangolin.ShouldQuit():
            self.viewer_refresh()

    def viewer_init(self, w, h):
        # Configuración de la ventana de visualización
        pangolin.CreateWindowAndBind('Map Viewer', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
            pangolin.ModelViewLookAt(0, -10, -8, 0, 0, 0, 0, -1, 0)
        )
        self.handler = pangolin.Handler3D(self.scam)
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
        self.dcam.SetHandler(self.handler)

    def draw_floor(self):
        # Dibujar el piso como una cuadrícula que cubra el área de la trayectoria
        gl.glLineWidth(1)
        gl.glColor3f(0.7, 0.7, 0.7)  # Gris claro para el piso
        x_min, z_min = self.grid_min
        x_max, z_max = self.grid_max
        grid_step = 1  # Espacio entre líneas de la cuadrícula

        for x in np.arange(x_min, x_max + grid_step, grid_step):
            gl.glBegin(gl.GL_LINES)
            gl.glVertex3f(x, 0, z_min)
            gl.glVertex3f(x, 0, z_max)
            gl.glEnd()

        for z in np.arange(z_min, z_max + grid_step, grid_step):
            gl.glBegin(gl.GL_LINES)
            gl.glVertex3f(x_min, 0, z)
            gl.glVertex3f(x_max, 0, z)
            gl.glEnd()

    def draw_axes(self):
        # Dibujar ejes para representar Norte, Sur, Este y Oeste
        axis_length = 5.0
        gl.glLineWidth(2)

        # Eje X (Este-Oeste) en rojo
        gl.glColor3f(1.0, 0.0, 0.0)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(axis_length, 0, 0)  # Este
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(-axis_length, 0, 0)  # Oeste
        gl.glEnd()

        # Eje Z (Norte-Sur) en azul
        gl.glColor3f(0.0, 0.0, 1.0)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(0, 0, axis_length)  # Norte
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(0, 0, -axis_length)  # Sur
        gl.glEnd()

    def viewer_refresh(self):
        # Limpiar la pantalla
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        # Dibujar el piso y los ejes
        self.draw_floor()
        self.draw_axes()

        # Dibujar la trayectoria de la cámara
        gl.glLineWidth(2)
        gl.glColor3f(0.0, 1.0, 0.0)  # Verde para las cámaras
        pangolin.DrawCameras(self.poses)

        pangolin.FinishFrame()

# Cargar el archivo .npz
data = np.load('mapa.npz')

# Extraer las poses y los puntos 3D
poses = data['poses']

# Inicializar la visualización
viewer = MapVisualizer(poses)
viewer.create_viewer()
