# Text-to-Image-Synthesis-Using-GAN
Text-to-Image synthesis system using GANs to generate realistic face images from text inputs through joint learning of text encoding and adversarial training.
This project implements a Generative Adversarial Network (GAN) to generate realistic human face images from textual descriptions.
It combines Natural Language Processing (NLP) and Computer Vision to bridge the gap between text and image generation.

Features
Generate face images from text input
Uses GAN (Generator + Discriminator) for realistic outputs
Text encoding using TF-IDF / NLP techniques
Adversarial training for improved image quality
Simple GUI interface using Tkinter

 How It Works
User inputs a text description
Text is converted into feature vectors (text encoding)
Generator creates images using these features
Discriminator evaluates real vs fake images
Model improves through adversarial training
Final output → realistic synthesized face image

 Tech Stack
Language: Python
Frameworks: TensorFlow / Keras
Libraries: NumPy, Pandas, OpenCV, Scikit-learn
GUI: Tkinter
