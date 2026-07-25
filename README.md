# Прогнозирование оттока студентов

ML-модель для предсказания вероятности отчисления студента на основе его академических и социально-экономических данных.

## Стек
- **Python:** pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn
- **ML:** Random Forest, Logistic Regression, XGBoost, SHAP-интерпретация
- **Веб:** Streamlit

## Результаты
| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.873 | 0.879 | 0.943 | 0.910 | 0.917 |
| Random Forest | 0.877 | 0.881 | 0.947 | 0.913 | 0.921 |
| XGBoost | 0.868 | 0.879 | 0.933 | 0.906 | 0.914 |

Лучшая модель — **Random Forest (ROC-AUC = 0.921)**.

## Топ-5 важных признаков
1. Curricular units 2nd sem (approved)
2. Curricular units 2nd sem (grade)
3. Curricular units 1st sem (approved)
4. Curricular units 1st sem (grade)
5. Tuition fees up to date

## Запуск

pip install streamlit pandas numpy scikit-learn
streamlit run app.py


## Файлы
- `app.py` — Streamlit-сервис для интерактивного предсказания
- `dataset.csv` — данные студентов
- `ML-прогнозирование оттока студентов.ipynb` — ноутбук с EDA и обучением моделей