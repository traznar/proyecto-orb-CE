import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import subprocess
import threading
import main 

class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processor")

        # Variables para almacenar la ruta del video y el estado de la detección de obstáculos
        self.video_path = None
        self.obstacle_detection = tk.BooleanVar()

        # Elementos de la interfaz
        self.create_widgets()

    def create_widgets(self):
        # Botón para seleccionar video
        tk.Button(self.root, text="Seleccionar Video", command=self.select_video).grid(row=0, column=0, padx=10, pady=10)
        self.video_label = tk.Label(self.root, text="No se ha seleccionado ningún video")
        self.video_label.grid(row=0, column=1, padx=10, pady=10)

        # Checkbox para activar la detección de obstáculos
        tk.Checkbutton(self.root, text="Activar detección de obstáculos", variable=self.obstacle_detection).grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        # Botón para iniciar el procesamiento del video
        tk.Button(self.root, text="Iniciar Procesamiento", command=self.start_processing).grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        # Botón para ejecutar la visualización del mapa
        tk.Button(self.root, text="Ver Mapa", command=self.view_map).grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    def select_video(self):
        # Abrir el cuadro de diálogo sin restricciones de tipo de archivo
        self.video_path = filedialog.askopenfilename()  # No especificamos filetypes
        if self.video_path:
            self.video_label.config(text=f"Video seleccionado: {self.video_path}")
        else:
            self.video_label.config(text="No se ha seleccionado ningún video")

    def start_processing(self):
        if not self.video_path:
            messagebox.showerror("Error", "Por favor selecciona un video antes de procesar.")
            return

        main.reset()  # Limpiar el estado antes de comenzar
        main.mapp.create_viewer()  # Iniciar el visualizador
        threading.Thread(target=self.process_video).start()

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir el video.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            main.process_frame(frame, obstacle_detection=self.obstacle_detection.get())

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        # Cerrar la ventana de `pangolin`
        main.mapp.close_viewer()

        messagebox.showinfo("Procesamiento finalizado", "El procesamiento del video ha terminado.")


    def view_map(self):
        # Llamar a mapExtractor.py para ver el mapa
        threading.Thread(target=lambda: subprocess.run(["python3", "mapExtractor.py"])).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()
