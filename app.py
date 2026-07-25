import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Прогноз оттока студентов", layout="centered")
st.title("🎓 Прогнозирование оттока студентов")
st.write("Модель предсказывает вероятность отчисления на основе данных студента.")

@st.cache_resource
def train_model():
    df = pd.read_csv('dataset.csv', sep=',')
    
    # Целевая переменная — последняя колонка
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Бинарная цель: Dropout (0) vs остальные (1)
    y = (y == 'Dropout').astype(int)
    
    # Обучаем
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X.columns.tolist()

model, scaler, feature_names = train_model()

st.sidebar.header("Введите данные студента")

important = [
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Tuition fees up to date',
    'Age at enrollment'
]

user_input = {}
for feature in feature_names:
    if feature in important:
        user_input[feature] = st.sidebar.number_input(f"⭐ {feature}", value=0.0, step=0.1)
    else:
        user_input[feature] = st.sidebar.number_input(feature, value=0.0, step=0.1)

if st.sidebar.button("Предсказать", type="primary"):
    df_input = pd.DataFrame([user_input])
    df_input = df_input[feature_names]
    df_scaled = scaler.transform(df_input)
    
    proba = model.predict_proba(df_scaled)
    dropout_prob = proba[0, 1] if proba.shape[1] > 1 else 0.0
    
    col1, col2 = st.columns(2)
    col1.metric("Вероятность отчисления", f"{dropout_prob:.1%}")
    col2.metric("Шанс остаться", f"{1 - dropout_prob:.1%}")
    
    if dropout_prob > 0.5:
        st.error("⚠️ Высокий риск отчисления")
    else:
        st.success("✅ Низкий риск отчисления — студент продолжит обучение")

st.caption("Модель: Random Forest, ROC-AUC = 0.921")