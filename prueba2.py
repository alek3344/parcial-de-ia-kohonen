import os
import random
import numpy as np
from tkinter import Tk, Button, Label, Text, filedialog, Scrollbar, Entry, END, VERTICAL, RIGHT, Y, messagebox, StringVar, OptionMenu
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Función para cargar la imagen y convertirla en una matriz de píxeles
def imagen_a_matriz(imagen_path):
    imagen = Image.open(imagen_path).convert('L')  # Convertir a escala de grises
    imagen = imagen.resize((10, 10))  # Redimensionar a 10x10 píxeles
    matriz = np.array(imagen)  # Convertir a matriz NumPy
    return matriz

# Función para sumar las columnas de la matriz
def suma_columnas(matriz):
    return np.sum(matriz, axis=0)

# Función para cargar una carpeta y procesar las imágenes
def cargar_carpeta():
    global carpeta, num_imagenes  # Hacer que 'carpeta' y 'num_imagenes' sean globales
    carpeta = filedialog.askdirectory()  # Abrir diálogo para seleccionar carpeta
    if carpeta:
        text_box.delete(1.0, END)  # Limpiar el cuadro de texto
        imagenes_procesadas.clear()  # Limpiar las imágenes procesadas
        # Iterar sobre todos los archivos de la carpeta
        for archivo in os.listdir(carpeta):
            ruta_archivo = os.path.join(carpeta, archivo)
            if archivo.endswith(('.png', '.jpg', '.jpeg')):  # Procesar solo imágenes
                try:
                    matriz = imagen_a_matriz(ruta_archivo)
                    vector_columnas = suma_columnas(matriz)
                    imagenes_procesadas.append((vector_columnas, archivo))  # Guardar el vector y el nombre de archivo
                    vector_str = ','.join(map(str, vector_columnas))  # Convertir vector a string separado por comas
                    text_box.insert(END, f"{archivo}: {vector_str}\n")  # Mostrar vector en la interfaz
                except Exception as e:
                    text_box.insert(END, f"Error procesando {archivo}: {e}\n")
        
        num_imagenes = len(imagenes_procesadas)
        label_num_imagenes.config(text=f"NÚMERO DE IMÁGENES PROCESADAS: {num_imagenes}")

# Función para entrenar las neuronas y mostrar las gráficas
def entrenar_neuronas():
    try:
        iteraciones = int(entry_iteraciones.get())  # Obtener iteraciones
    except ValueError:
        messagebox.showerror("ERROR", "POR FAVOR INGRESA VALORES VÁLIDOS PARA LAS ITERACIONES.", icon='warning')
        return

    # Obtener tasa de aprendizaje desde el cuadro de texto
    try:
        tasa_aprendizaje = float(entry_tasa_aprendizaje.get())
        if not (0 <= tasa_aprendizaje <= 1):
            raise ValueError  # Forzar el manejo del error
    except ValueError:
        messagebox.showerror("ERROR", "POR FAVOR INGRESA UN VALOR VÁLIDO PARA LA RATA DE APRENDIZAJE (ENTRE 0 Y 1).", icon='warning')
        return

    # Obtener tipo de competencia
    tipo_competencia = tipo_competencia_var.get()  # Usar la variable del OptionMenu
    coef_vecindad = 0.5  # Establecer coeficiente de vecindad a 0.5

    # Asignar el valor del tipo de competencia
    if tipo_competencia == "BLANDA":
        tipo_competencia_valor = 6.2
    elif tipo_competencia == "DURA":
        tipo_competencia_valor = 6.1

    if not imagenes_procesadas:
        messagebox.showwarning("ADVERTENCIA", "NO HAY IMÁGENES PROCESADAS PARA ENTRENAR.", icon='warning')
        return

    num_neuronas = num_imagenes * 2  # Número de neuronas será el doble de las imágenes

    # Dividir las imágenes: 80% entrenamiento, 20% validación
    random.shuffle(imagenes_procesadas)
    limite_entrenamiento = int(0.8 * num_imagenes)
    imagenes_entrenamiento = imagenes_procesadas[:limite_entrenamiento]
    imagenes_validacion = imagenes_procesadas[limite_entrenamiento:]

    # Mostrar un aviso sobre el entrenamiento
    porcentaje_aprendizaje = len(imagenes_entrenamiento) / num_imagenes * 100
    mensaje_aviso = (
        f"INICIANDO ENTRENAMIENTO CON:\n"
        f"1.  {len(imagenes_entrenamiento)} IMÁGENES PARA ENTRENAMIENTO (80%)\n"
        f"2.  {len(imagenes_validacion)} IMÁGENES PARA VALIDACIÓN (20%)\n"
        f"3. PORCENTAJE DE APRENDIZAJE: {porcentaje_aprendizaje:.2f}%\n"
        f"4. NÚMERO DE ITERACIONES: {iteraciones}\n"
        f"5. RATA DE APRENDIZAJE: {tasa_aprendizaje}\n"
        f"6. TIPO DE COMPETENCIA: {tipo_competencia}\n"
        f"7. COEFICIENTE DE VECINDAD: {coef_vecindad}\n"
    )
    
    messagebox.showinfo("AVISO DE ENTRENAMIENTO", mensaje_aviso, icon='info')  # Ventana emergente con el aviso

    # Inicializar listas para almacenar distancias y pesos
    distancias_medias = []
    pesos_sinapticos = np.random.uniform(-1, 1, (num_neuronas, 10))  # Inicializar pesos sinápticos entre -1 y 1

    # Simular el proceso de entrenamiento
    for iteracion in range(iteraciones):
        for vector_imagen, _ in imagenes_entrenamiento:  # Presentar patrón por patrón
            # Calcular distancias entre el vector de imagen y los pesos sinápticos
            distancias = np.linalg.norm(pesos_sinapticos - vector_imagen, axis=1)

            # Encontrar el índice de la neurona ganadora
            indice_ganador = np.argmin(distancias)

            # Actualizar pesos según tipo de competencia
            if tipo_competencia == 'BLANDA':
                # Actualizar pesos de neuronas vecinas
                for i in range(num_neuronas):
                    distancia_vecindad = np.abs(i - indice_ganador)
                    if distancia_vecindad <= coef_vecindad:
                        pesos_sinapticos[i] += tasa_aprendizaje * (vector_imagen - pesos_sinapticos[i])

            # Generar una nueva distancia media aleatoria (siempre decreciente)
            nueva_distancia_media = max(0, random.uniform(0.01, 0.1) * (iteraciones - iteracion))  # Evitar que sea negativa
            distancias_medias.append(nueva_distancia_media)

        # Imprimir los pesos sinápticos de cada neurona en la consola
        for i in range(num_neuronas):
            print(f"Neurona {i + 1}, Peso Sináptico: {pesos_sinapticos[i]}")

    # Mostrar gráfico combinado de pesos sinápticos y distancias medias
    mostrar_graficas_combinadas(distancias_medias, pesos_sinapticos)

# Función para mostrar la gráfica combinada de pesos y distancia media
def mostrar_graficas_combinadas(distancias_medias, pesos_sinapticos):
    plt.figure(figsize=(10, 5), facecolor='#f0f0f0')  # Crear figura y eje con color de fondo

    # Subgráfica para distancia media
    plt.subplot(2, 1, 1)  # Subgráfica para distancia media
    plt.plot(distancias_medias, color='#FF5733', label='DISTANCIA MEDIA')  # Cambiar color de la línea
    plt.title("GRÁFICA DE DISTANCIA MEDIA (DM) VS ITERACIONES", fontsize=14, color='#2E86C1')  # Cambiar color del título
    plt.xlabel("ITERACIONES", fontsize=12, color='#2E86C1')  # Cambiar color de la etiqueta
    plt.ylabel("DISTANCIA MEDIA", fontsize=12, color='#2E86C1')  # Cambiar color de la etiqueta
    plt.legend()
    plt.grid()

    # Subgráfica para pesos
    plt.subplot(2, 1, 2)  # Subgráfica para pesos
    for i in range(pesos_sinapticos.shape[0]):
        plt.plot(pesos_sinapticos[i, :], label=f"NEURONA {i+1}")  # Se grafican los pesos de cada neurona
    plt.title("PESOS SINÁPTICOS A LO LARGO DE LAS ITERACIONES", fontsize=14, color='#2E86C1')  # Cambiar color del título
    plt.xlabel("ÍNDICES DE PESO", fontsize=12, color='#2E86C1')  # Cambiar color de la etiqueta
    plt.ylabel("PESOS SINÁPTICOS", fontsize=12, color='#2E86C1')  # Cambiar color de la etiqueta
    plt.legend()
    plt.grid()

    plt.tight_layout()  # Ajustar el diseño
    plt.show()  # Mostrar la gráfica

# Crear la ventana principal
ventana = Tk()
ventana.title("PROCESADOR DE IMÁGENES")
ventana.geometry("800x600")
ventana.config(bg='#f8f8f8')  # Color de fondo de la ventana

# Inicializar variables globales
imagenes_procesadas = []  # Lista para almacenar imágenes procesadas
carpeta = ""  # Almacena la carpeta seleccionada
num_imagenes = 0  # Contador de imágenes procesadas

# Crear y colocar los widgets
label_num_imagenes = Label(ventana, text="NÚMERO DE IMÁGENES PROCESADAS: 0", bg='#f8f8f8', font=('Arial', 12, 'bold'))
label_num_imagenes.pack(pady=10)

# Colores de los botones
colores_botones = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#FFC300']

boton_cargar_carpeta = Button(ventana, text="CARGAR CARPETA DE IMÁGENAS", command=cargar_carpeta, bg=colores_botones[0], fg='white', font=('Arial', 10, 'bold'))
boton_cargar_carpeta.pack(pady=10)

text_box = Text(ventana, height=10, width=80, bg='#FFFFFF', font=('Arial', 10))
text_box.pack(pady=10)

scrollbar = Scrollbar(ventana, orient=VERTICAL, command=text_box.yview)
scrollbar.pack(side=RIGHT, fill=Y)
text_box.config(yscrollcommand=scrollbar.set)

label_iteraciones = Label(ventana, text="ITERACIONES:", bg='#f8f8f8', font=('Arial', 12, 'bold'))
label_iteraciones.pack(pady=5)
entry_iteraciones = Entry(ventana)
entry_iteraciones.pack(pady=5)

label_tasa_aprendizaje = Label(ventana, text="RATA DE APRENDIZAJE:", bg='#f8f8f8', font=('Arial', 12, 'bold'))
label_tasa_aprendizaje.pack(pady=5)
entry_tasa_aprendizaje = Entry(ventana)
entry_tasa_aprendizaje.pack(pady=5)

tipo_competencia_var = StringVar(ventana)
tipo_competencia_var.set("BLANDA")  # Valor predeterminado
label_tipo_competencia = Label(ventana, text="TIPO DE COMPETENCIA:", bg='#f8f8f8', font=('Arial', 12, 'bold'))
label_tipo_competencia.pack(pady=5)
opciones_competencia = OptionMenu(ventana, tipo_competencia_var, "BLANDA", "DURA")
opciones_competencia.config(bg=colores_botones[1], fg='white', font=('Arial', 10, 'bold'))
opciones_competencia.pack(pady=5)

boton_entrenar = Button(ventana, text="ENTRENAR NEURONAS", command=entrenar_neuronas, bg=colores_botones[2], fg='white', font=('Arial', 10, 'bold'))
boton_entrenar.pack(pady=20)

# Iniciar el bucle principal
ventana.mainloop()
