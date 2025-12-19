import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import requests
from io import BytesIO
import torchvision.models as models

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Классы
CLASSES = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain',
           'rainbow', 'rime', 'sandstorm', 'snow']

# Путь к модели
MODEL_PATH = "models/efficientnet_model.pth"

# Загрузка модели
@st.cache_resource
def load_model():
    model = models.efficientnet_v2_m(weights=None)
    num_classes = len(CLASSES)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    
    state_dict = torch.load(MODEL_PATH, map_location=device)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

# Трансформации
transform = transforms.Compose([
    transforms.Resize(480),
    transforms.CenterCrop(480),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Классификация
def classify_image(image):
    start_time = time.time()
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = CLASSES[predicted_idx]
        confidence = probabilities[0][predicted_idx].item()
    inference_time = time.time() - start_time
    return predicted_class, confidence, inference_time

# Тёмная тема + стиль
st.set_page_config(page_title="Погодные Явления", page_icon="⛅", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00ddeb, #ff00aa, #ff6a00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 1rem 0;
        text-shadow: 0 0 20px rgba(255,255,255,0.5);
    }
    .subtitle {
        text-align: center;
        color: #cccccc;
        font-size: 1.3rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background: rgba(30, 30, 60, 0.85);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.15);
        transition: transform 0.3s ease;
        margin-bottom: 1.5rem;
        color: #ffffff;
    }
    .result-card:hover {
        transform: translateY(-10px);
    }
    .class-name {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00ffea;
        text-shadow: 0 0 15px #00ffea;
    }
    .progress-container {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        height: 12px;
        margin-top: 1rem;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #00ffea, #ff00aa);
        transition: width 0.8s ease;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(30,30,60,0.8);
        border-radius: 15px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #cccccc;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00ffea;
    }
    .footer {
        text-align: center;
        color: #cccccc;
        margin-top: 3rem;
        font-size: 0.9rem;
        padding: 1rem;
        background: rgba(0,0,0,0.4);
        border-radius: 10px;
    }
    /* Белый текст в обычных местах */
    p, span, div, label:not(.stFileUploader label), small, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    /* ЧЁРНЫЙ ТЕКСТ В ЗАГРУЗКЕ ФАЙЛОВ */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderInstructions"] {
        color: #000000 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff !important;
        border: 2px dashed #000000 !important;
        border-radius: 10px !important;
        color: #000000 !important;
    }
    /* Спиннеры белые */
    .stSpinner > div > div {
        color: #ffffff !important;
        border-color: #ffffff !important;
    }
    .stSpinner svg {
        stroke: #ffffff !important;
    }
    /* Поле ввода URL — белый текст */
    div[data-testid="stTextInput"] input {
        color: #ffffff !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #cccccc !important;
    }
    div[data-testid="stTextInput"] input {
        background-color: rgba(40, 40, 60, 0.9) !important;
        border: 1px solid #555 !important;
    }
    div[data-testid="stTextInput"] small {
        color: #cccccc !important;
    }
    /* Чёрный текст в сайдбаре */
    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<div class="main-title">⛅ Погодные Явления</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Загружайте фото и узнайте, какое природное явление на снимке</div>', unsafe_allow_html=True)

# Вкладки
tab1, tab2 = st.tabs(["📸 Классификация", "ℹ️ О модели"])

with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Загрузка по ссылке")
        url = st.text_input("Вставьте прямую ссылку на изображение (.jpg/.png):", key="url_input")
        if url:
            with st.spinner("Загружаем изображение..."):
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                        'Referer': 'https://www.google.com/',
                    }
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    content_type = response.headers.get('Content-Type', '')
                    if not content_type.startswith('image/'):
                        st.error("Это не изображение!")
                    else:
                        img = Image.open(BytesIO(response.content)).convert("RGB")
                        st.image(img, caption="Загруженное изображение", width=600)
                        class_name, prob, inf_time = classify_image(img)
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="class-name">{class_name.capitalize()}</div>
                            <p>Уверенность: <strong>{prob:.0%}</strong></p>
                            <p>Время анализа: <strong>{inf_time:.2f} сек</strong></p>
                            <div class="progress-container">
                                <div class="progress-fill" style="width: {prob*100}%"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Ошибка: {e}")
                    st.info("Попробуйте скачать изображение и загрузить файлом.")

        st.subheader("Загрузка файлов")
        uploaded_files = st.file_uploader("Выберите изображения (можно несколько)", 
                                          type=["jpg", "jpeg", "png"], 
                                          accept_multiple_files=True,
                                          key="file_uploader")
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with st.spinner(f"Обрабатываем {uploaded_file.name}..."):
                    try:
                        img = Image.open(uploaded_file).convert("RGB")
                        col_img, col_res = st.columns([2, 1])
                        with col_img:
                            st.image(img, caption=uploaded_file.name, width=400)
                        with col_res:
                            class_name, prob, inf_time = classify_image(img)
                            st.markdown(f"""
                            <div class="result-card">
                                <div class="class-name">{class_name.capitalize()}</div>
                                <p>Уверенность: <strong>{prob:.0%}</strong></p>
                                <p>Время: <strong>{inf_time:.2f} сек</strong></p>
                                <div class="progress-container">
                                    <div class="progress-fill" style="width: {prob*100}%"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Ошибка: {e}")

with tab2:
    st.title("О модели")
    st.markdown("""
    **Модель:** EfficientNet V2 Medium  
    **Точность:** **90.55%**  
    **Классы:** 11 природных явлений   
    **Фреймворк:** PyTorch + torchvision  
    **Обучена на:** Weather Image Recognition 
    """)
    
    st.subheader("Распознаваемые явления:")
    st.write(", ".join([f"**{c}**" for c in CLASSES]))

    # Добавленное изображение
    st.subheader("Кривая обучения и метрики")
    st.image("images/training_history.png", caption="История обучения модели", width=800)

# Футер
st.markdown("""
<div class="footer">
    Разработано с ❤️ для анализа погоды | 2025
</div>
""", unsafe_allow_html=True)