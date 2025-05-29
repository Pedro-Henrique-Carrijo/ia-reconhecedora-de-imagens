#IA FEITA EM GOOGLECOLAB

import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from google.colab import files
import requests
from io import BytesIO

traducoes = {
    'cat': 'gato',
    'dog': 'cachorro',
    'car': 'carro',
    'laptop': 'notebook',
    'cell_phone': 'celular',
    'bottle': 'garrafa',
    'book': 'livro'
}

print("Carregando modelo de IA...")
modelo = MobileNetV2(weights='imagenet')
print("Modelo carregado com sucesso!")

def analisar_imagem(imagem, origem):
    if origem == "upload":
        try:
            img = image.load_img(imagem, target_size=(224, 224))
        except Exception as e:
            print(f"Erro ao carregar a imagem local: {e}")
            return
    else:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(imagem, headers=headers)
            response.raise_for_status()

            try:
                img = Image.open(BytesIO(response.content))
            except UnidentifiedImageError:
                print("Imagem corrompida. Tentando recuperar com OpenCV...")
                img_np = np.frombuffer(response.content, np.uint8)
                img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                if img_cv is None:
                    raise ValueError("Não foi possível decodificar a imagem.")
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img_cv)

            img = img.convert('RGB')
            img = img.resize((224, 224))

        except Exception as e:
            print(f"Erro ao carregar a imagem da URL: {e}")
            return

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    print("Analisando imagem...")
    previsoes = modelo.predict(img_array)
    resultados = decode_predictions(previsoes, top=5)[0]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Imagem Analisada')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    nomes = [traducoes.get(r[1], r[1]).replace('_', ' ').title() for r in resultados]
    plt.barh(nomes, [r[2] for r in resultados], color='skyblue')
    plt.xlim(0, 1.0)
    plt.title('O que a IA vê na imagem')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    print("\nO que encontrei na imagem:")
    for i, (_, label, score) in enumerate(resultados):
        nome = traducoes.get(label, label).replace('_', ' ').title()
        print(f"{i+1}. {nome}: {score*100:.1f}%")

def main():
    print("Bem-vindo ao analisador de imagens!")

    while True:
        print("\n1 - Analisar imagem do computador")
        print("2 - Analisar imagem da internet")
        print("3 - Sair")

        opcao = input("Escolha uma opção: ")

        if opcao == '1':
            print("\nFaça upload da imagem:")
            uploaded = files.upload()
            if uploaded:
                arquivo = list(uploaded.keys())[0]
                analisar_imagem(arquivo, "upload")

        elif opcao == '2':
            url = input("Digite a URL da imagem: ")
            if url:
                analisar_imagem(url, "url")

        elif opcao == '3':
            print("Encerrando o programa.")
            break
        else:
            print("Opção inválida. Tente novamente.")

main()
