# Boxing Guard and Predictability Analyzer

This is a real-time computer vision tool designed to help boxers and coaches identify holes in a fighter's defense. It uses MediaPipe and OpenCV to track your guard and analyze how predictable your head movement is.

## How it works

The app splits your defense into two main categories:

1. **Guard Susceptibility**: The program looks at your head and torso and identifies specific "strike zones" (like the chin for jabs or the ribs for body shots). It then draws polygons over your arms. If your arms aren't covering a strike zone, the UI flags that area as open.
2. **Predictability Tracking**: The system tracks the coordinates of your head over time. It calculates your velocity and acceleration to predict where you will move next. If you move exactly where the math says you will, your predictability score goes up.



## Evolution of the project

A quick note on the development: earlier versions of this project attempted to use **Time Complexity Analysis** to evaluate movement patterns. We eventually decided to scrap that approach. 

We found that the math required for algorithmic complexity didn't actually provide better insights than simple physics heuristics, and it slowed the program down significantly. By removing it, we made the app much more responsive and useful for high-speed training.

## Getting Started

This project is built to be accessible for beginners. As long as you have a webcam and a basic Python environment, you can get it running in a few minutes.

### Prerequisites

You'll need to install the following dependencies:

```bash
pip install kivy opencv-python mediapipe numpy shapely
