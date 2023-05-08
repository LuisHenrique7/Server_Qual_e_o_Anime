import flask
import tensorflow as tf
import numpy as np
from PIL import Image


app = flask.Flask(__name__)

print("Carregando o modelo...")

# Carregue o modelo salvo em formato .h5
model = tf.keras.models.load_model('my_h5_model.h5')

print("Modelo carregado com sucesso!")

labels = {
  "Ace of Diamond": 0,
  "Akame ga Kill!": 1,
  "Angel Beats": 2,
  "Anohana The Flower We Saw That Day": 3,
  "Attack on Titan": 4,
  "Baccano": 5,
  "Black Butler": 6,
  "Black Cat": 7,
  "Black Lagoon": 8,
  "Bleach": 9,
  "Btooom": 10,
  "Chobits": 11,
  "Code Geass": 12,
  "Cowboy Bebop": 13,
  "Darker Than Black": 14,
  "Deadman Wonderland": 15,
  "Death Note": 16,
  "Demon Slayer": 17,
  "Dragon Ball Z Kai": 18,
  "Dragon ball": 19,
  "Ergo Proxy": 20,
  "Eureka Seven": 21,
  "Fairy Tail": 22,
  "Fighting Spirit": 23,
  "Food Wars": 24,
  "Fullmetal Alchemist Brotherhood": 25,
  "Future Diary": 26,
  "Gekijouban FateStay Night Unlimited Blade Works": 27,
  "Get Backers": 28,
  "Haikyuu": 29,
  "Hunter x Hunter": 30,
  "Kuroko Basketball": 31,
  "Magi The labyrinth of magic": 32,
  "Monster": 33,
  "Naruto": 34,
  "Naruto Shippuden": 35,
  "Neon Genesis Evangelion": 36,
  "No Game, No Life": 37,
  "One Outs": 38,
  "One Piece": 39,
  "Paranoia Agent": 40,
  "Parasyte The Maxim": 41,
  "Phantom Requiem for the Phantom": 42,
  "Psycho Pass": 43,
  "Puella Magi Madoka Magica": 44,
  "Rurouni Kenshin": 45,
  "Steins Gate": 46,
  "The Seven Deadly Sins": 47,
  "Tokyo Ghoul": 48,
  "Your Lie in April": 49,
  "one punch man": 50
}

def classify(predicted):
    classes = [-1, -1, -1]
    animes = ["null", "null", "null"]

    for p in range(0, len(predicted)):
        if predicted[p] > classes[0]:
            classes[0] = predicted[p]
            animes[0] = list(labels.keys())[p]
        elif classes[0] > predicted[p] and predicted[p] > classes[1]:
            classes[1] = predicted[p]
            animes[1] = list(labels.keys())[p]
        elif classes[1] > predicted[p] and predicted[p] > classes[2]:
            classes[2] = predicted[p]
            animes[2] = list(labels.keys())[p]

    return [
                {"anime": animes[0], "probability": classes[0]},
                {"anime": animes[1], "probability": classes[1]},
                {"anime": animes[2], "probability": classes[2]}
            ]

@app.route('/')
def raiz():
    return 'Hello world!'

# Defina uma rota para receber as solicitações de previsão
@app.route('/predict', methods=['POST'])
def predict():
    # Obtenha a imagem do cliente
    file = flask.request.files['image']
    # print("\n file \n", file.content_type)
    image = Image.open(file)
    image = image.convert("RGB")
    # Redimensione a imagem para o tamanho esperado pelo modelo
    image = image.resize((260, 260))

    # Converta a imagem para um array NumPy e normalize
    image = np.array(image) / 255.0

    # Adicione uma dimensão para acomodar o tamanho do lote
    image = np.expand_dims(image, axis=0)

    # Faça a previsão com o modelo
    prediction = model.predict(image)

    # Ajuste os dados do response
    response = flask.jsonify(classify(prediction.tolist()[0]))
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

app.run()