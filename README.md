# 🌱 RL-Based Smart Irrigation System

## 📌 Overview

This project implements a **Reinforcement Learning (RL)** based smart irrigation system using a **Deep Q-Network (DQN)**.
The goal is to **optimize water usage** while maintaining ideal soil moisture conditions for crops.

---

## 🎯 Objectives

* Maintain soil moisture in optimal range (40–60%)
* Minimize water usage
* Automate irrigation decisions using AI

---

## 🧠 Methodology

The system uses **Deep Reinforcement Learning**:

* **State (Environment Inputs):**

  * Soil Moisture
  * Temperature
  * Humidity
  * Rainfall
  * Time of Day

* **Actions:**

  * 0 → No water
  * 1 → Low water
  * 2 → Medium water
  * 3 → High water

* **Reward Function:**

  * Positive reward for optimal moisture
  * Penalty for overwatering or underwatering
  * Penalty for excessive water use

---

## ⚙️ Technologies Used

* Python
* PyTorch
* NumPy
* Matplotlib

---

## 📁 Project Structure

SMART.IRRIGATION/
│── environment.py      # Simulation environment
│── dqn_agent.py       # DQN implementation
│── train.py           # Training script
│── test.py            # Testing script
│── baseline.py        # Comparison (random & rule-based)
│── dqn_irrigation_model.pth  # Saved model

---

## 🚀 How to Run

### 1️⃣ Install dependencies

```bash
pip install torch numpy matplotlib
```

### 2️⃣ Train the model

```bash
python train.py
```

### 3️⃣ Test the model

```bash
python test.py
```

### 4️⃣ Run baseline comparison

```bash
python baseline.py
```

---

## 📊 Results

* The RL agent learns to maintain soil moisture efficiently
* Water usage decreases over time
* Outperforms random and rule-based methods

Graphs generated:

* Reward vs Episode
* Water Usage vs Episode

---

## 🧪 Features

* Custom irrigation simulation environment
* Deep Q-Network (DQN) implementation
* Experience replay and target network
* Model saving and loading
* Performance visualization

---

## 🔮 Future Improvements

* Integration with real IoT sensors (ESP32/Arduino)
* Use real-world weather datasets
* Add crop-specific irrigation models
* Deploy as a mobile/web application

---

## 👨‍💻 Author

Manya Chapra

---

## 📌 Conclusion

This project demonstrates how **Reinforcement Learning** can be used to build an intelligent irrigation system that conserves water while maintaining optimal crop conditions.
