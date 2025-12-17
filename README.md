# AI Posture Correction Service

## Overview
This project is an AI-based service that detects user posture in real-time and provides feedback. It classifies posture as "Good" or "Bad" using a machine learning model and alerts the user with audio feedback when poor posture is detected.

## Quick Start

1. Install Dependencies
Open your terminal and run the following command to install the required libraries:
pip install -r requirements.txt

2. Run Locally
Execute the Streamlit application:
streamlit run app.py

3. Open Application
The application will automatically open in your default web browser. If not, access it via:
http://localhost:8501

## Live Demo
* Public URL: https://m4wsihxskdilrkwjeqd2bl.streamlit.app/ 
* Note: Please allow camera access permission for the demo to work.

## Dataset
* Sources: 
[good] 파일 file X 28
[mild] file X 10
[severe] file X 10
* Size: 48 videos
* License: CC BY 4.0

## Model
* Tool/Framework: Python, Streamlit, Scikit-learn
* Model File: posture_model.pkl
* Accuracy: 94.5%
* Input: Real-time webcam feed (processed via MediaPipe)

## Credits
* Team Members: Se Eun Chun, Chaemin Lim, Dayeon Lee, Dain Lee, Jiyoon Choi
* Tools Used: Streamlit, MediaPipe, OpenCV
* Data Sources: 48 self-filmed videos
