# Intelligent Pesticide Sprinkling System (SIH 2025)

## Team: TheVisioneers \| PS ID: SIH25015

## 1. Project Overview

This project is an AI-powered automated pesticide sprinkling system that
detects the infection level of plants and sprays pesticides only where
required...

## 2. Problem Statement & Solution

### Problem

-   Excess pesticide spraying causes soil/water contamination.
-   Manual inspection is inaccurate and slow.

### Solution

A cable-mounted automated sprayer that: - Uses Blue-filter camera for
clear infection visibility - ML classification (Healthy/Diseased) -
Precision spraying only on infected areas - Real-time monitoring via
mobile app (DHARA)

## 3. Features

-   Blue-filter camera
-   Edge ML on Raspberry Pi
-   Cable-mounted carriage
-   Precision multi-nozzle sprayer
-   Mobile app for alerts & history

## 4. Technologies Used

### Hardware

-   Raspberry Pi 4, Blue-filter Camera, Motors, Relays, Battery/Solar

### Software / ML

-   Python, TensorFlow/Keras, OpenCV
-   CNN classifier
-   Flutter app

## 5. Steps to Install & Run

### Clone Repository

git clone `<URL>`{=html}

### Install Dependencies

pip install -r requirements.txt

### Run Model

python inference.py

### Raspberry Pi Camera Detection

python pi_camera_detect.py

## 6. Environment Variables

API_KEY=`<your key>`{=html} DB_URL=`<database url>`{=html}
MODEL_PATH=./model/model.tflite

## 7. Folder Structure

teamname_psid/ ├── project/ ├── src/ ├── assets/ ├── requirements.txt
└── README.md

## 8. References

-   ICAR, IARI
-   MDPI plant disease papers
-   PlantVillage dataset
