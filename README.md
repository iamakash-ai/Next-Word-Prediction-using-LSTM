# Next-Word-Prediction-using-LSTM
Generate Next word of sentence using LSTM Network


This project implements a Next Word Prediction model using Long Short-Term Memory (LSTM) neural networks, built with TensorFlow and deployed with Streamlit for an interactive web application.
You can view the app deployed in StreamlitCloud. [app](https://next-word-prediction-using-lstm-4mnhdeaa5gftappj8cgttr3.streamlit.app/)


## Features
- **Deep Learning Model:** Utilizes an LSTM-based neural network to predict the next word in a given sequence of text.
- **Interactive Interface:** Provides a user-friendly interface for real-time prediction using Streamlit.
- **Customizable Training:** Allows users to train the model on custom datasets.
- **TensorFlow Integration:** Leverages TensorFlow for efficient model building and training.

---

## Prerequisites

Before running the project, ensure you have the following installed:
- Python 3.7+
- pip (Python package manager)

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/iamakash-ai/Next-Word-Prediction-using-LSTM.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Next-Word-Prediction-using-LSTM
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training the Model
1. Place your training dataset in the `data/` folder (e.g., `data/train_text.txt`).
2. Run the training script:
   ```bash
   Use Jupyter Notebook lstm_project.ipynb
   ```
3. The trained model will be saved in the `models/` directory.

### Running the Application
1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open the URL provided by Streamlit to access the application (e.g., `http://localhost:8501`).

---

## File Structure

```
Next-word-prediction-using-LSTM/
├── data/                # Dataset folder             
├── app.py               # Streamlit application
├── next_word_lstm.h5    # Model 
├── tokenizer.pickle     # tokenizer file
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## Dependencies
- TensorFlow
- Streamlit
- NumPy
- Pandas
- Matplotlib

See `requirements.txt` for the complete list.


