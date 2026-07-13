# ✈️ Airline Passenger Forecasting System

## 📌 Overview

This project is a Streamlit-based web application that predicts future airline passenger traffic using a Recurrent Neural Network (RNN). It analyzes historical time-series data, visualizes trends, and generates forecasts for upcoming months.

---

## 🚀 Features

* 📊 Interactive dashboard
* 📈 Historical data visualization
* 🔮 Future passenger forecasting (1–24 months)
* 📉 Model performance metrics (MAE, MSE, RMSE)
* 📥 Download forecast results as CSV

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Pandas, NumPy
* Plotly
* Scikit-learn
* TensorFlow / Keras

---

## 📂 Project Structure

```
AirlinePassengerForecasting/
│
├── app.py
├── requirements.txt
├── data/
│   └── airline-passengers.csv
├── src/
│   ├── data_loader.py
│   ├── forecast.py
│   └── evaluation.py
├── assets/
│   └── 201623.png
```

---

## ⚙️ Installation & Run

1. Clone the repository:

```
git clone https://github.com/your-username/AirlinePassengerForecasting.git
```

2. Navigate to the project folder:

```
cd AirlinePassengerForecasting
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the app:

```
streamlit run app.py
```

---

## 📊 Output

* Displays historical passenger trends
* Shows model evaluation metrics
* Generates future forecasts with visualization

---

## 🔮 Future Enhancements

* Implement LSTM/GRU models
* Deploy on cloud platforms
* Add real-time data integration

---

## 👩‍💻 Author

Tanneeru Astalakshmi
