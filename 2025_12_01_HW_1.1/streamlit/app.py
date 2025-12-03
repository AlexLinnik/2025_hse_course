import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="HW1 Linear Regression Prediction", page_icon="🎯", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "models"

AVAILABLE_MODELS = {
    'Линейная': 'linear.pkl',
    'Линейная (масштабированая)': 'linear_scaled.pkl',
    'Lasso (default)': 'lasso.pkl',
    'Lasso (оптимальное)': 'lasso_optimal.pkl',
    'Elastic': 'elastic.pkl',
    'Ridge': 'ridge.pkl'
}

def remove_postfixs(df, column, postfixs):
    for postfix in postfixs:
        df[column] = df[column].str.replace(postfix, '', regex=False)
    return df[column]

@st.cache_resource
def load_model(model_file):
    """Загружаем модель через pickle"""
    with open(f'{MODEL_DIR}/{model_file}', 'rb') as file:
        model = pickle.load(file)

    return model

def test_empty_values(df, columns):
    for column in columns:
        if df[column].isnull().any():
            st.error(f"❌ Ошибка обработки модели: '{column}' не везде заполнен")
            st.stop()

def prepare_features_common(df):
    """Приводим данные к формату обучения модели."""
    try:
        # Делаем дубликат:
        df_process = df.copy()
        # Нет смысла тащить фичи из модели, так как тут может быть преобразование
        # типов, чистка постфиксов. Повторим логику из обучалки:
        if 'torque' in df.columns:        
            df_process = df_process.drop(['torque'], axis = 1)
        df_process['max_power'] = df_process['max_power'].replace('', np.nan)
        df_process['mileage'] = df_process['mileage'].replace('', np.nan)
        
        test_empty_values(df_process, ['seats','mileage','engine','max_power'])
        df_process['seats'] = df_process['seats'].astype(int)
        if 'selling_price' in df.columns:
            df_process = df_process.drop(['selling_price'], axis = 1)       
        df_process['mileage'] = remove_postfixs(df_process, 'mileage', [' kmpl', ' km/kg'])
        df_process['engine'] = remove_postfixs(df_process, 'engine', [' CC'])
        df_process['max_power'] = remove_postfixs(df_process, 'max_power', [' bhp'])

        df_process['max_power'] = df_process['max_power'].astype(float)    
        df_process['mileage'] = df_process['mileage'].astype(float)
        df_process['engine'] = df_process['engine'].astype(int)
        
    except Exception as e:
        st.error(f"❌ Ошибка обработки модели: {e}")
        st.stop()
    
    # Преобразуем категориальные признаки в строки (как при обучении)
    # for col in feature_names:
    #    if col in df_proc.columns:
    #        if df_proc[col].dtype in ('object', 'bool'):
    #            df_proc[col] = df_proc[col].astype(str)
                
    return df_process

def get_dummies_with_seats(df):
    df = pd.get_dummies(df, drop_first=True)
    seats_dummies = pd.get_dummies(df['seats'], prefix='seats')
    return pd.concat([df.drop('seats', axis=1), seats_dummies], axis=1)

def use_first_name(df):
    return df['name'].str.split(' ').str[0]

def prepare_features(model_name, scaler, data):
    st.write(f'Готовим данные для {model_name}:')
    
    data = prepare_features_common(data)
    st.write('... Применена стандартная трансформация')
    
    if (model_name != 'Ridge'):
        data = data.select_dtypes(include='number')   
        st.write('... Отброшены категориальные признаки') 
    
    if (model_name != 'Линейная' and model_name != 'Ridge'):
        data_columns = data.columns
        data = pd.DataFrame(scaler.transform(data), columns=data_columns)
        st.write('... Применен линейный скаллер')
        
    if (model_name == 'Ridge'):
        st.write('... Переформатируем поле name')
        data['name'] = use_first_name(data)
        st.write('... Выделяем dummy признаки')
        data = get_dummies_with_seats(data)
        st.write('... Выравниваем колонки признакам модели')
        X_train = load_model('ridge_features.pkl')
        X_train, data = X_train.align(data, join='left', axis=1, fill_value=False)
        assert X_train.shape[1] == data.shape[1]
        
        scaler = load_model('scaler_ridge.pkl')
        data_columns = data.columns
        data = data = pd.DataFrame(scaler.transform(data), columns=data_columns)
        st.write('... Применен линейный скаллер на Ridge')        

    
    return data    

# --- Основной интерфейс ---
st.title("Предсказание стоимости автомобилей")

selected_option = st.selectbox('Выбирите модель:', AVAILABLE_MODELS.keys())
st.write('Выбрана модель:', selected_option)

# Загрузка CSV файла
uploaded_file = st.file_uploader("Загрузите CSV файл для предсказания", type=["csv"])

if uploaded_file is None:
    st.info("👈 Загрузите CSV файл!")
    st.stop()

# Загружаем данные и делаем предсказания
data = pd.read_csv(uploaded_file)

# Загружаем скалер
try:
    scaler = load_model('scaler.pkl')
except Exception as e:
    st.error(f"❌ Ошибка загрузки скалера: {e}")
    st.stop()

# Загружаем модель
try:
    model = load_model(AVAILABLE_MODELS[selected_option])
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()

try:
    X_test = prepare_features(selected_option, scaler, data)    
    if X_test.isna().sum().sum() != 0:        
        st.error(f"❌ Загруженные данные содержат пропуски в ключевых полях. Провертье колонки {X_test.columns}")
    
    predictions = model.predict(X_test)    
    data['prediction'] = predictions
except Exception as e:
    st.error(f"❌ Ошибка при обработке данных: {e}")
    st.stop()

if 'selling_price' in data.columns:
    st.subheader(f"Дополнительная статистика (при наличии колонки selling_price)")
    r2 = r2_score(data['selling_price'], data['prediction'])
    mse = MSE(data['selling_price'], data['prediction'])
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R2 score", r2)
    with col2:
        st.metric("MSE", mse)
    
else:
    st.info("👈 В загруженном файле нет поля selling_price, добавьте его, если хотиет получить дополнительную статистику")

st.subheader("📊 Результаты")

col1, col2 = st.columns(2)
with col1:
    st.metric("Всего машин", len(data))
with col2:
    mean = data['prediction'].mean()
    st.metric("Средняя цена предсказания", f"{mean:.1f}")

# Ответы с предсказаниями:
st.subheader(f"Модель: {selected_option}")
if 'selling_price' in data.columns:
    st.table(data[['name', 'selling_price', 'prediction']])
else: 
    st.table(data[['name', 'prediction']])   

# --- Визуализации ---
st.subheader("📈 Визуализации")
fig, ax = plt.subplots()
ax.barh(X_test.columns, model.coef_)
ax.set_xlabel('Вес')
ax.set_title('Коэффициенты модели')
st.pyplot(fig)
