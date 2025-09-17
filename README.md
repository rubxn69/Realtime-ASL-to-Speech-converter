# ASL Sign Language ↔ Speech Converter

An interactive system that bridges the communication gap between the Deaf and hearing communities.  
This device converts **American Sign Language (ASL)** gestures into **speech** and also supports **speech-to-sign reciprocation**.

---

## 🚀 Features

### ✋ Sign-to-Speech Conversion
- Captures live ASL hand gestures using a webcam.  
- Processes gestures with **OpenCV** and **CVZone**.  
- Classifies signs using deep learning models built with **Keras** and **TensorFlow**.  
- Outputs real-time speech/audio for recognized signs.  

### 🎙 Speech-to-Sign Reciprocation
- Takes spoken input from a user.  
- Converts speech into corresponding ASL gesture (on-screen display / animation).  

### 🔄 Bidirectional Communication
- Facilitates smooth interaction between hearing and non-hearing individuals.  

---

## 🛠️ Tech Stack
- **Programming Language**: Python  
- **Computer Vision**: OpenCV, CVZone  
- **Deep Learning**: TensorFlow, Keras  
- **Audio**: Text-to-Speech (TTS) and Speech Recognition libraries  

---

## 📊 Model Training
- **Dataset**: ASL gesture images/videos  
- **Preprocessing**: Hand landmark detection with CVZone + OpenCV  
- **Training**: CNN/RNN models with Keras & TensorFlow  
- **Output**: Trained `.h5` model used for inference  

---

## 🎯 Future Improvements
- Expand ASL vocabulary coverage  
- Improve speech recognition accuracy  
- Add real-time mobile app integration  
- Support for multiple sign languages  

---


Add real-time mobile app integration

Improve speech recognition accuracy

Support for multiple sign languages
