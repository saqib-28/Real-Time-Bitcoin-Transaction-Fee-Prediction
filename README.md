# 🔮 FeeFlow – Real-Time Bitcoin Transaction Fee Prediction

**FeeFlow** is a deep learning-powered web application that accurately predicts Bitcoin transaction fees based on live mempool data and transaction-specific features. It empowers users to make informed decisions, avoid overpayment, and minimize confirmation delays by estimating optimal fees tailored to current network conditions.

## 📌 Overview

**The Challenge:**  
Bitcoin’s fee market is highly volatile due to fluctuating network congestion. Users risk delays if fees are set too low, or unnecessary expense if they overpay.

**The Solution:**  
FeeFlow uses a trained deep neural network (DNN) to forecast the ideal fee for a given transaction. The model is built using real transaction data from the Bitcoin blockchain and live mempool analytics.

## 🤖 Machine Learning

- Trained on **900,000+ historical Bitcoin transactions**
- Feature engineering applied on:
  - Transaction size (in bytes)
  - Transaction weight
  - Number of inputs and outputs
  - Fee rate per kilobyte
- Preprocessing: Log transformation + scaling with `StandardScaler`

## 📈 Model Performance

- **R² Score:** `0.9991`
- **Mean Absolute Error (MAE):** ~`21.6 sats`
- **Prediction Accuracy within ±2%:** ~`97.8%`

## 🌐 Web Interface (Frontend)

The optional web dashboard enables users to:

- Enter transaction details manually
- Instantly receive a smart fee prediction
- Compare high/low fee suggestions for saving estimation
- Visualize fee trends using interactive charts

## 📁 Project Structure

```
FeeFlow/
├── data/
│   └── transaction_data.csv        # Raw and cleaned dataset
├── models/
│   ├── fee_prediction_model.keras  # Trained DNN model
│   └── scaler.pkl                  # Preprocessing scaler
├── notebooks/
│   └── model_training.ipynb        # Jupyter notebooks for exploration/training
├── app/
│   ├── app.py                      # Web app backend (Flask/Streamlit)
│   └── predict.py                  # Standalone prediction script
└── README.md                       # Project documentation
```

## 🧪 Technologies Used

- **Languages:** Python  
- **ML/DL:** TensorFlow, Keras  
- **Data Processing:** Pandas, NumPy, Scikit-learn  
- **Visualization:** Matplotlib, Seaborn  
- **Web App (Optional):** Flask or Streamlit

## 🚀 How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/FeeFlow.git
   cd FeeFlow
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the model**
   ```bash
   python predict.py
   ```

4. *(Optional)* Run the web app
   ```bash
   streamlit run app/app.py
   ```

## 📬 Contributions & Future Scope

- Add support for **Ethereum Gas Fee Prediction**
- Improve prediction model using **recurrent layers (LSTM/GRU)**
- Deploy the platform for **real-world use in wallets and exchanges**
