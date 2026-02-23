# Questo è il file principale che collega tutte le sezioni del progetto.
# Qui sono importati i modelli allenati della CNN e di YOLO.
# Attraverso la GUI, realizzata utilizzando la libreria Gradio, è possibile inserire uno screenshot della propria partita.
# Lo screenshot poi verrà processato, utilizzando il modello YOLO per estrarre i bounding box, i quali successivamente verranno passati
# al modello della CNN per la predizione delle carte.
# Successivamente vengono disegnati sull'immagine originale i bounding box e le etichette.
# Infine vengono creati due array per la memorizzazione delle carte del player e del dealer che vengono passati allo 
# script che applica la strategia fondamentale per restituire la mossa probabilisticamente migliore.

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import gradio as gr


from CNN.DualHeadCNN import DualHeadCNN
from CNN.predict_card_image import run_model
from FUNDAMENTAL_STRATEGY.application_fundamental_strategy import suggerisci_mossa

import torch
from torchvision import transforms
from ultralytics import YOLO

import argparse

PROJECT_DIR= os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carico i pesi del modello della CNN
CNN_MODEL= DualHeadCNN()
state_dict= torch.load(os.path.join(PROJECT_DIR, "models/cnn_weights.pth"), map_location= DEVICE)
CNN_MODEL.load_state_dict(state_dict)


# Carico il modello allenato di YOLO
YOLO_MODEL= YOLO(os.path.join(PROJECT_DIR, "models/yolo_best.pt"))
YOLO_MODEL.conf= 0.3    # Soglia minima di confidenza

# Trasformazioni immagine per la CNN
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((240, 180)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])


def run_evalutation(img_rgb) -> Any:
    # === 1. Carica CNN
    CNN_MODEL.eval()

    # Ottiengo la dimensione
    height, width, _ = img_rgb.shape

    # === 2. Applica YOLO
    results = YOLO_MODEL(img_rgb)
    
    # Inizializzo Array carte
    carta_dealer = ""
    carte_player = []
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cropped = img_rgb[y1:y2, x1:x2]

            # === 3. Uso la CNN per predire il valore della carta
            pred_seme, pred_numero= run_model(cropped, CNN_MODEL)   # Due tuple, restituiscono la predizione con la percentuale di predizione

            etichetta = f"{pred_numero[0]} di {pred_seme[0]}"

            if y1 < height//2:
                carta_dealer = pred_numero[0]

                # === 4. Disegna box e label sull'immagine
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 125, 0), 2)
                cv2.putText(img_rgb, etichetta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 125, 0), 2)
            elif y1 < (height - ((height*30)//100)):
                carte_player.append(pred_numero[0])

                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 125, 0), 2)
                cv2.putText(img_rgb, etichetta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 125, 0), 2)
                    
            print(f"[✓] Carta rilevata: {etichetta} @ [{x1}, {y1}, {x2}, {y2}]")

    print("carta Dealer: " + str(carta_dealer) + "\ncarte Player: " + str(carte_player))

    # === 5. Chiamo la funzione che applica la strategia fondamentale
    mossa: str= suggerisci_mossa(carte_player, carta_dealer)
    print("mossa: " + str(mossa))

    dealer_img= img_rgb[:height//2, :, :]
    player_img= img_rgb[height//2+1:, :, :]
    return (
        dealer_img,
        player_img,
        carta_dealer,
        ', '.join(carte_player),
        mossa
    )

# Utilizzo Gradio per la GUI
def use_gradio() -> None:

    # creo la gui tramite i blocchi di gradio
    with gr.Blocks() as demo:
        gr.Markdown("BlackJack game helper")

        # La prima riga conterrà le due sezioni per le immagini, una di input e una di output
        with gr.Row():
            with gr.Column():
                # Permette il caricamento di immagini sia tramite immagini dirette, che tramite frame della camera
                input_image = gr.Image(sources=["webcam", "upload"], type="numpy", label= "Input")

                btn = gr.Button("Run Predict")

            with gr.Column():
                with gr.Column():
                    dealer_card = gr.Textbox(label= "Carta Dealer", interactive= False)
                    dealer_output = gr.Image(label= "Dealer Image")

                with gr.Column():
                    player_output = gr.Image(label= "Player Image")
                    with gr.Row():
                        player_cards = gr.Textbox(label= "Carte Player", interactive= False)
                        mossa_text = gr.Textbox(label= "Mossa Consigliata", interactive= False)

        btn.click(
            fn=run_evalutation,
            inputs=input_image,
            outputs=[
                dealer_output,
                player_output,
                dealer_card,
                player_cards,
                mossa_text
            ])

    demo.launch(share=True)


if __name__ == "__main__":
    use_gradio()