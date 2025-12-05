import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.preprocessing import LabelEncoder


data = {
    "descripcion": [
        "Es un felino grande con manchas negras y excelente nadador.",
        "Ave enorme que no vuela y corre a gran velocidad.",
        "Mamífero marino con aletas y piel gruesa.",
        "Roedor pequeño de orejas grandes y cola larga.",
        "Gran herbívoro con uno o dos cuernos y piel gruesa.",
        "Reptil de gran tamaño con mordida poderosa y vive en ríos.",
        "Pájaro pequeño que imita sonidos y canta.",
        "Mamífero de gran tamaño con trompa larga.",
        "Felino pequeño domesticado con bigotes largos.",
        "Animal equino usado para transporte y trabajo.",
        "Carnívoro marino con dientes afilados.",
        "Ave colorida que puede hablar palabras humanas.",
        "Mamífero peludo que hiberna durante el invierno.",
        "Reptil que cambia de color para camuflarse.",
        "Ave nocturna con ojos grandes y cabeza giratoria.",
        "Mamífero que vive colgado de los árboles boca abajo.",
        "Insecto con alas coloreadas y vuelo suave.",
        "Mamífero marsupial que salta y tiene bolsa ventral.",
        "Anfibio pequeño que croa y vive cerca del agua.",
        "Felino muy veloz y de cuerpo esbelto.",
        "Felino grande veloz y solitario con rayas negras y narajas con garras afiladas"
    ],
    "ubicacion": [
        "América del Sur",   # jaguar
        "África",            # avestruz
        "Océano Ártico",     # foca
        "América del Norte", # ratón
        "África",            # rinoceronte
        "África",            # cocodrilo
        "Sudamérica",        # ruiseñor
        "Asia",              # elefante
        "Global",            # gato
        "Global",            # caballo
        "Océanos",           # tiburón
        "Selvas tropicales", # loro
        "América del Norte", # oso
        "África",            # camaleón
        "Bosques",           # búho
        "Selva tropical",    # perezoso
        "Global",            # mariposa
        "Australia",         # canguro
        "Global",            # rana
        "África",            # guepardo
        "África",            # tigre

        
    ],
    "animal": [
        "jaguar",
        "avestruz",
        "foca",
        "ratón",
        "rinoceronte",
        "cocodrilo",
        "ruiseñor",
        "elefante",
        "gato",
        "caballo",
        "tiburón",
        "loro",
        "oso",
        "camaleón",
        "búho",
        "perezoso",
        "mariposa",
        "canguro",
        "rana",
        "guepardo",
        "tigre"
    ]
}

df = pd.DataFrame(data)
df.to_csv("IdntificacionDeAnimales/descripcionAnimal.csv", index=False)

print("Dataset creado correctamente.")

df = pd.read_csv("IdntificacionDeAnimales/descripcionAnimal.csv")

df["texto"] = df["descripcion"] + " ubicacion: " + df["ubicacion"]
X = df["texto"].values
y = df["animal"].values

# print(X)
# print(y)

# //convertir palabras a números.

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

num_clases = len(label_encoder.classes_)
# print("Clases:", label_encoder.classes_)

# TOKENIZACIÓN
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)
max_len = max(len(x) for x in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

vocab_size = len(tokenizer.word_index) + 1

# MODELO LSTM
model = models.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
    layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
    layers.LSTM(64),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_clases, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    padded_sequences,
    y_encoded,
    epochs=80,
    batch_size=2,
    verbose=1
)

model.summary()

##Entrenamiento 

history = model.fit(
    padded_sequences,
    y_encoded,
    epochs=50,
    batch_size=4,
    verbose=1
)

##hacer prediccion 
def predecir(adivinanza):
    seq = tokenizer.texts_to_sequences([adivinanza])
    pad = pad_sequences(seq, maxlen=padded_sequences.shape[1], padding='post')
    pred = model.predict(pad)
    idx = pred.argmax()
    return label_encoder.inverse_transform([idx])[0]


def predecir_top(texto, n=3):
    # MISMO maxlen usado en el entrenamiento
    max_len = padded_sequences.shape[1]

    seq = tokenizer.texts_to_sequences([texto])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(pad)[0]

    # Usar el *mismo* label_encoder ya entrenado
    indices = np.argsort(pred)[::-1][:n]

    resultados = []
    for idx in indices:
        animal = label_encoder.classes_[idx]
        prob = float(pred[idx])
        resultados.append((animal,' Probabilidad de:', prob))

    return resultados

#Predicción mas proxima
texto = "domesticado con bigotes largos"
print(predecir(texto))
#Prediccion mas de uno
print(predecir_top(texto, n=3))

model.save("IdntificacionDeAnimales/modelo_clasificación.h5")