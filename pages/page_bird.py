import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Данные модели (замените на реальные, если есть)
ACCURACY = 90.55

TRAINING_TIME = "15 минут (COLAB)"
DATASET_SIZE = 6862

st.title("ℹ️ О модели")

# Общая информация
st.markdown("""
### Основные характеристики
- **Модель:** EfficientNet V2 Medium  
- **Точность на тестовом наборе:** **90.55%**   
- **Время обучения:** 15 минут (COLAB) 
- **Размер входа:** 224x224  
- **Количество классов:** 11 (природные явления)  
- **Фреймворк:** PyTorch + torchvision
""")

# Кривая обучения
st.subheader("Кривая обучения и метрик")
st.image("images/training_history.png", caption="История обучения модели", width=800)

# Состав датасета
st.subheader("Состав датасета")
st.write(f"**Общее число объектов:** {DATASET_SIZE}")




# Дополнительная информация
st.subheader("Примечания")
st.markdown("""
- Модель обучена на датасете Weather Image Recognition с реальными фотографиями атмосферных явлений.  
- Использовалась аугментация данных.  
- Оптимизатор: Adam.  
""")

# Футер страницы
st.markdown("---")
st.markdown("<div style='text-align: center; color: #cccccc;'>Разработано с ❤️ | 2025</div>", unsafe_allow_html=True)