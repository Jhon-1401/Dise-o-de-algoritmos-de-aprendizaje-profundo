import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras import layers, models


data = {
    "adivinanza": [
        "Agua pasó por aquí, cate que no te vi",
        "Blanca por dentro, verde por fuera",
        "Oro parece, plata no es",
        "Tiene dientes y no come",
        "Vuelo sin alas y lloro sin ojos",
        "blanco es, gallina lo pone y frito se come",
        "lana sube, lana baja "
    ],
    "respuesta": [
        "El aguacate",
        "la pera",
        "el platano",
        "el peine",
        "las nubes",
        "El huevo",
        "La navaja"
    ]
}

df = pd.DataFrame(data)
df.to_csv("adivinanzas.csv", index=False)

print("Dataset creado correctamente.")

df = pd.read_csv("adivinanzas.csv")

X = df["adivinanza"].values
y = df["respuesta"].values

print(X)
print(y)

# //convertir palabras a números.

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

num_clases = len(label_encoder.classes_)
print("Clases:", label_encoder.classes_)

#Tokenizar
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)
padded_sequences = pad_sequences(sequences, padding='post')

vocab_size = len(tokenizer.word_index) + 1
print("Tamaño vocabulario:", vocab_size)

### se crea el modelo LSTM en TensorFlow

model = models.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=64),
    layers.LSTM(128),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_clases, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
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

print(predecir("algo pasó por aca, y no lo vi"))

model.save("modelo_adivinanzas.h5")