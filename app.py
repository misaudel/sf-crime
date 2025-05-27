import joblib
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from datetime import datetime
import numpy as np
# Configuración de layout para reducir espacios laterales
#st.set_page_config(layout="wide")
MIN_LAT, MAX_LAT = 37.7034, 37.8120
MIN_LON, MAX_LON = -122.5270, -122.3482
@st.cache_resource
def load_models():
    return {
        'Regresión Logística': joblib.load('logistic_pipeline.joblib'),
        'Random Forest':        joblib.load('Random_forest2.joblib'),
        'XGBoost':              joblib.load('XGB.joblib'),
        'SVM Calibrado':        joblib.load('SVM1.joblib')
    }

@st.cache_resource
def load_grid_bins(x_path='x_bins.txt', y_path='y_bins.txt'):
    x_bins = np.loadtxt(x_path, delimiter=',')
    y_bins = np.loadtxt(y_path, delimiter=',')
    return x_bins, y_bins
x_bins, y_bins = load_grid_bins()
REFERENCE_DATE = datetime(2003, 1, 6).date()
models = load_models()
st.title("🔮 Predicción de Categoría de Crimen en SF")
m = folium.Map(location=[37.77, -122.42], zoom_start=12)
m.add_child(folium.LatLngPopup())
st.subheader("Haz clic en el mapa para seleccionar un lugar:")
map_data = st_folium(m, height=500, width=800)

lat = None
lon = None
if map_data:
    last_clicked = map_data.get('last_clicked')
    if last_clicked:
        lat = last_clicked.get('lat')
        lon = last_clicked.get('lng')


#model = joblib.load("crime_model.pkl")
st.subheader("Configura los datos del incidente:")
# Panel lateral para otras características
# Selector de fecha y hora
fecha = st.sidebar.date_input(
    'Fecha del incidente',
    value=datetime(2015, 1, 1).date(),
    min_value=datetime(2003, 1, 1).date(),
    max_value=datetime.today().date()
)
hora = st.sidebar.time_input(
    'Hora del incidente',
    value=datetime.now().time().replace(second=0, microsecond=0)
)
# Derivar variables temporales
DayOfWeek = fecha.weekday()  # 0=Lunes, 6=Domingo según Python
Day = fecha.day
Month = fecha.month
Year = fecha.year
DayOfYear = fecha.timetuple().tm_yday
Hour = hora.hour
Minute = hora.minute
n_days = (fecha - REFERENCE_DATE).days

# Flags y grillas espaciales
is_block  = st.sidebar.selectbox('is_block',  [0, 1])
is_corner = st.sidebar.selectbox('is_corner', [0, 1])

# Verificar que se haya seleccionado ubicación en el mapa

if lat is None or lon is None:
    st.error("Por favor, haz clic en el mapa para seleccionar latitud y longitud.")
elif not (MIN_LAT <= lat <= MAX_LAT and MIN_LON <= lon <= MAX_LON):
    st.error("Las coordenadas seleccionadas están fuera de San Francisco.")
else:
    x_grid = np.digitize([lon], x_bins)[0] - 1
    y_grid = np.digitize([lat], y_bins)[0] - 1

    df_input = pd.DataFrame({
        'DayOfWeek': [DayOfWeek], 'X': [lon], 'Y': [lat],
        'is_block':  [is_block],  'is_corner': [is_corner],
        'Minute':    [Minute],    'Hour': [Hour],
        'Day':       [Day],       'Month': [Month],
        'Year':      [Year],      'DayOfYear': [DayOfYear],
        'n_days':    [n_days],    'x_grid': [x_grid],
        'y_grid':    [y_grid]
    })

    st.subheader('Datos de Entrada')
    st.write(df_input)

    st.subheader('Probabilidades Predichas')
    for name, model in models.items():
        proba = model.predict_proba(df_input)[0]
        df_proba = pd.DataFrame([proba], columns=[f"Clase_{cls}" for cls in model.classes_])
        st.markdown(f"**{name}**")
        st.write(df_proba.round(3))


