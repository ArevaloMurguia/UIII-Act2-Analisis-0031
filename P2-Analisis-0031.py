# importar librerias 
import pandas as pd
import numpy as np
# Arevalo Rafael NC:0031 Num.List.:3
print("Analisis de datos Pacientes Arevalo Rafael NC:0031")

# Ejemplo: 1. Diagnóstico de Salud (Modelos de Clasificación Médica)
# Este dataset es ideal para entrenar modelos predictivos de riesgo de enfermedades según variables biométricas.

Pacientes = {
    "Nombre": ["Ana", "Luis", "Marta", "Roberto", "Carla", "Javier"],
    "Presion_Arterial": [120, 140, 120, 110, 150, 140],
    "Fumador": [False, True, False, False, True, True],
    "Nivel_Riesgo": ["Bajo", "Alto", "Bajo", "Bajo", "Alto", "Alto"]
}

df = pd.DataFrame(Pacientes)
print(df)

print("+-+-+-+- Agrupar en una sola columna Nombre +-+-+-+-")
Agrupado = df.groupby("Nombre")
print(Agrupado)
print(" +-+-+- Por Agregacion +-+-+-")
print("+-+-+- Promedio de Presion Arterial por paciente +-+-+-+-")
# Presion_Arterial promedio por Paciente
print(Agrupado['Presion_Arterial'].mean())
print("+-+-+-+- Fumador en cada Paciente +-+-+-+-")
# Fumador en cada Paciente
print(Agrupado['Fumador'].max())

print("+-+-+-+-+- Segunda Parte +-+-+-0031+-+-+")
print("Agregacion multiples")
print("max --> maximo, min --> minimo ")
Resultado_Agregado = Agrupado['Fumador'].agg(['mean', 'sum', 'max', 'min'])
print(Resultado_Agregado)
print("Agrupación por varias columnas")
print("Agrega columna DataFrame")
print("Columna Team --> Equipo ")
df['Equipo'] = ['A','B','A','B','A','B']
agrupado_múltiple = df.groupby(['Presion_Arterial', 'Equipo'])['Fumador'].mean()
print(agrupado_múltiple)

