import dash
from dash import dcc, html, Input, Output
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
import base64
import io
from PIL import Image
import plotly.graph_objs as go

# Завантаження моделей
def load_vgg16_model():
    return load_model('vgg16_model.keras')

def load_cnn_model():
    return load_model('cnn_model.keras')

# Функція для передобробки зображення
def preprocess_image(image, model_type, target_size=(150, 150)):
    image = cv2.resize(image, target_size)
    if model_type == "CNN":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=-1)  # Додаємо канал
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32') / 255
    return np.expand_dims(image, axis=0)

# Класи Fashion MNIST
class_names = {
    0: 'Футболка/Топ',
    1: 'Штани',
    2: 'Светр',
    3: 'Сукня',
    4: 'Пальто',
    5: 'Сандалі',
    6: 'Сорочка',
    7: 'Кросівки',
    8: 'Сумка',
    9: 'Черевики'
}

# Ініціалізація Dash
app = dash.Dash(__name__)
app.title = "Класифікація зображень"

# Основний макет
app.layout = html.Div([
    html.H1("Класифікація зображень за допомогою нейронної мережі", style={'text-align': 'center'}),
    
    # Вибір моделі
    html.Div([
        html.Label("Оберіть модель:"),
        dcc.Dropdown(
            id='model-dropdown',
            options=[
                {'label': 'VGG16', 'value': 'VGG16'},
                {'label': 'CNN', 'value': 'CNN'}
            ],
            value='VGG16'
        )
    ], style={'width': '50%', 'margin': 'auto'}),

    # Завантаження зображення
    html.Div([
        html.Label("Завантажте зображення:"),
        dcc.Upload(
            id='upload-image',
            children=html.Div(['Перетягніть або ', html.A('оберіть файл')]),
            style={
                'width': '50%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': 'auto'
            },
            multiple=False
        ),
    ]),

    # Відображення завантаженого зображення
    html.Div(id='output-image-upload', style={'text-align': 'center', 'margin-top': '20px'}),

    # Результати класифікації
    html.Div(id='classification-output', style={'text-align': 'center', 'margin-top': '20px'}),

    # Графік ймовірностей
    dcc.Graph(id='probability-graph', style={'margin-top': '20px'})
])

# Обробка завантаженого зображення
@app.callback(
    [Output('output-image-upload', 'children'),
     Output('classification-output', 'children'),
     Output('probability-graph', 'figure')],  # Додаємо Output для графіку
    [Input('upload-image', 'contents'),
     Input('model-dropdown', 'value')]
)
def update_output(contents, model_type):
    if contents is None:
        return None, None, {}

    # Декодуємо зображення
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = np.array(Image.open(io.BytesIO(decoded)))

    # Передобробка зображення
    if model_type == "VGG16":
        model = load_vgg16_model()
        processed_image = preprocess_image(image, "VGG16", target_size=(150, 150))
    else:
        model = load_cnn_model()
        processed_image = preprocess_image(image, "CNN", target_size=(28, 28))

    # Передбачення
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    predicted_label = class_names[predicted_class]

    # Створення текстового виведення ймовірностей для кожного класу
    class_probabilities = {class_names[i]: predictions[0][i] for i in range(len(class_names))}
    class_probabilities_text = "\n".join([f"{class_name}: {prob*100:.2f}%" for class_name, prob in class_probabilities.items()])

    # Створення графіку ймовірностей
    classes = list(class_probabilities.keys())
    probabilities = list(class_probabilities.values())

    # Створення стовпчикового графіка
    fig = go.Figure(data=[go.Bar(x=classes, y=probabilities, text=[f'{prob*100:.2f}%' for prob in probabilities], textposition='auto')])
    fig.update_layout(title="Ймовірності для кожного класу", xaxis_title="Клас", yaxis_title="Ймовірність", showlegend=False)

    # Виведення результатів
    image_html = html.Img(src=contents, style={'width': '300px', 'margin-top': '20px'})
    result_html = html.Div([
        html.H3(f"Передбачений клас: {predicted_label} (ID: {predicted_class})"),
        html.P(f"Ймовірності для кожного класу:"),
        html.Pre(class_probabilities_text)
    ])
    
    return image_html, result_html, fig  # Повертаємо графік

@app.callback(
    Output('loss-accuracy-graph', 'figure'),
    [Input('show-graphs-btn', 'n_clicks'),
     Input('model-dropdown', 'value')]
)
def display_graphs(n_clicks, model_type):
    if n_clicks == 0:
        return {}

    # Завантажуємо історію навчання
    history_file = "training_history_vgg16.npy" if model_type == "VGG16" else "training_history_cnn.npy"
    history = np.load(history_file, allow_pickle=True).item()

    # Створюємо графіки
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(history['accuracy'], label='Train Accuracy')
    ax[0].plot(history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title("Accuracy")
    ax[0].legend()

    ax[1].plot(history['loss'], label='Train Loss')
    ax[1].plot(history['val_loss'], label='Validation Loss')
    ax[1].set_title("Loss")
    ax[1].legend()

    plt.tight_layout()

    # Перетворення графіку у формат для Dash
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return {
        'data': [],
        'layout': {
            'images': [{
                'source': f'data:image/png;base64,{encoded_image}',
                'xref': 'paper',
                'yref': 'paper',
                'x': 0,
                'y': 1,
                'sizex': 1,
                'sizey': 1,
                'xanchor': 'left',
                'yanchor': 'top',
                'layer': 'below'
            }]
        }
    }

# Запуск застосунку на локальному сервері
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
