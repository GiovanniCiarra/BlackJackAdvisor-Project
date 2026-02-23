# Blackjack Advisor: AI-Powered Strategy Assistant 🃏🤖

An end-to-end Computer Vision system that identifies playing cards from screen captures and provides real-time optimal betting strategies using Deep Learning.

## 🚀 Overview
This project automates the decision-making process in Blackjack by combining object detection, custom image classification, and statistical game theory.

## 🧠 Technical Workflow
The pipeline consists of four main stages:

1. **Object Detection (YOLO):** A YOLO model trained to detect all playing cards within a game screenshot.
2. **Spatial Logic:** Detected cards are automatically assigned to either the *Player* or the *Dealer* based on their coordinates.
3. **Dual-Head CNN (Custom Architecture):** - Designed and trained a **Multi-Task Learning (MTL) CNN** from scratch using **PyTorch**.
   - This "dual-head" architecture processes card crops to simultaneously predict two labels: **Rank** (e.g., Ace, King) and **Suit** (e.g., Hearts, Spades).
4. **Strategic Engine:** The recognized card values are fed into a deterministic algorithm that implements the **Blackjack Fundamental Strategy**, outputting the statistically optimal move (Hit, Stand, Double, or Split).

## 🛠 Tech Stack
- **Language:** Python
- **Deep Learning:** PyTorch, YOLO (Ultralytics)
- **Computer Vision:** OpenCV
- **GUI:** Gradio
- **Data:** Custom dataset for card recognition training.

---

## 📁 Project Documentation
For a detailed explanation of the neural network architecture, training process, and performance metrics, please refer to the following documents:

* [**Full Technical Report (PDF)**](./REPORT_&_PPT/Blackjack\Advisor.pdf)
* [**Project Presentation (PPT)**](./REPORT_&_PPT/BlackJackAdvisorPP.pdf)
