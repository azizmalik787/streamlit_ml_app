
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import os

# Page Configuration
st.set_page_config(
    page_title="ML Text Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# PYTORCH MODEL DEFINITIONS
# ============================================================================

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, fc_hidden_dim=64):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        output, (hn, cn) = self.lstm(x)
        x = self.dropout(hn[-1])
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)


class RNNClassifier(nn.Module):
    def __init__(self, word_embeddings, rnn_units=128, dense_layer_size=64):
        super().__init__()
        vocabulary_size, embedding_dimensions = word_embeddings.shape

        # Embedding layer
        self.word_embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(word_embeddings), freeze=False)

        # RNN Layer
        self.recurrent_layer = nn.RNN(embedding_dimensions, rnn_units, batch_first=True)

        # Dropout layer
        self.regularization_dropout = nn.Dropout(0.5)

        # Two fully connected layers
        self.first_dense_layer = nn.Linear(rnn_units, dense_layer_size)
        self.output_layer = nn.Linear(dense_layer_size, 1)

    def forward(self, input_tokens):
        embedded_tokens = self.word_embedding_layer(input_tokens)
        rnn_output, _ = self.recurrent_layer(embedded_tokens)

        # Use last hidden state
        final_hidden_state = self.regularization_dropout(rnn_output[:, -1, :])

        # Fully connected layers
        dense_output = torch.relu(self.first_dense_layer(final_hidden_state))
        classification_logits = self.output_layer(dense_output)

        return torch.sigmoid(classification_logits)



class CNNClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_filters=100, filter_sizes=[3, 4, 5]):
        super(CNNClassifier, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=num_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        conv_outs = [torch.relu(conv(x)) for conv in self.convs]
        pooled = [torch.max(out, dim=2)[0] for out in conv_outs]
        cat = torch.cat(pooled, dim=1)
        out = self.dropout(cat)
        return torch.sigmoid(self.fc(out))


# ============================================================================
# TEXT PROCESSING FUNCTIONS
# ============================================================================

def text_process(text):
    # Handle non-string values gracefully
    if not isinstance(text, str):
        return ""
    # converting into lower case.
    text = text.lower()
    # removing numerics, alphanumerics, and special characters.
    text = ' '.join(re.findall(r'\b[A-Za-z]+\b', text))

    # Remove punctuation from the tweet
    nopunc = ''.join([char for char in text if char not in string.punctuation])

    # Define a set of stopwords, including standard English stopwords and some custom ones
    # I have added words to vocab based on the 20 most repeating words in the dataset given.
    STOPWORDS = stopwords.words('english') + ['one', 'would', 'also', 'many', 'much', 'get', 'go', 'make', 'take']

    # Remove stopwords and join the words back into a string. also filtering out the words that are greater than length 2
    return ' '.join([word for word in nopunc.split() if word not in STOPWORDS and len(word) > 2])


# Function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Function to lemmatize text using POS tagging
def lemmatize_text(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return ' '.join(lemmatized_tokens)


def text_cleaning_wrapper(txt):
    return [text_process(i) for i in txt]


def lemmatize_text_wrapper(txt):
    return [lemmatize_text(i) for i in txt]


# ============================================================================
# MODEL LOADING SECTION
# ============================================================================

@st.cache_resource
def load_models():
    models = {}

    try:
        # Load embedding matrix
        try:
            embedding_matrix = np.load('models/embedding_gv.npy')
            models['embedding_matrix'] = embedding_matrix
            models['embedding_available'] = True
            vocab_size, embedding_dim = embedding_matrix.shape
            models['vocab_size'] = vocab_size
            models['embedding_dim'] = embedding_dim
            st.success(f"✅ Loaded embedding matrix: {vocab_size} vocab, {embedding_dim} dims")
        except FileNotFoundError:
            models['embedding_available'] = False
            st.error("❌ embedding_gv.npy not found!")

        # Load PyTorch models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        models['device'] = device

        if models['embedding_available']:
            # Load RNN model
            try:
                embedding_matrix = np.load("models/embedding_gv.npy")
                model_rnn = RNNClassifier(embedding_matrix)
                model_rnn.load_state_dict(torch.load("models/RNN_gv.pt", map_location=device))
                model_rnn.to(device)
                model_rnn.eval()
                models["rnn_gv"] = model_rnn
                models["rnn_available"] = True
            except:
                models["rnn_available"] = False

            try:
                embedding_matrix = np.load("models/embedding_gv.npy")
                model_cnn = CNNClassifier(embedding_matrix)
                model_cnn.load_state_dict(torch.load("models/CNN_gv.pt", map_location=device))
                model_cnn.to(device)
                model_cnn.eval()
                models["cnn_gv"] = model_cnn
                models["cnn_available"] = True
            except:
                models["cnn_available"] = False

            try:
                embedding_matrix = np.load("models/embedding_gv.npy")
                model_lstm = LSTMClassifier(embedding_matrix)
                model_lstm.load_state_dict(torch.load("models/LSTM_gv.pt", map_location=device))
                model_lstm.to(device)
                model_lstm.eval()
                models["lstm_gv"] = model_lstm
                models["lstm_available"] = True
            except:
                models["lstm_available"] = False
        try:
            # Load the main pipeline (Logistic Regression)
            try:
                models['pipeline'] = joblib.load('models/sentiment_analysis_pipeline.pkl')
                models['pipeline_available'] = True
            except FileNotFoundError:
                models['pipeline_available'] = False

            # Load TF-IDF vectorizer
            try:
                models['vectorizer'] = joblib.load('models/tfidf_vectorizer.pkl')
                models['vectorizer_available'] = True
            except FileNotFoundError:
                models['vectorizer_available'] = False

            # Load Logistic Regression model
            try:
                models['logistic_regression'] = joblib.load('models/logistic_regression_model.pkl')
                models['lr_available'] = True
            except FileNotFoundError:
                models['lr_available'] = False

            # Load Multinomial Naive Bayes model
            try:
                models['naive_bayes'] = joblib.load('models/multinomial_nb_model.pkl')
                models['nb_available'] = True
            except FileNotFoundError:
                models['nb_available'] = False

            # Load SVM model
            try:
                models['svm'] = joblib.load('models/svm_model.pkl')
                models['svm_available'] = True
            except FileNotFoundError:
                models['svm_available'] = False

            # Load Decision Tree model
            try:
                models['dt'] = joblib.load('models/dt_model.pkl')
                models['dt_available'] = True
            except FileNotFoundError:
                models['dt_available'] = False

            # Load AdaBoost model
            try:
                models['adb'] = joblib.load('models/adb_model.pkl')
                models['adb_available'] = True
            except FileNotFoundError:
                models['adb_available'] = False

        except Exception as e:
            st.error(f"Error loading traditional models: {e}")

        return models

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def make_prediction(text, model_choice, models):
    """Make prediction using the selected model"""
    if models is None:
        return None, None

    try:
        prediction = None
        probabilities = None

        # PyTorch models
        if model_choice in ['lstm', 'rnn', 'cnn'] and models.get(f'{model_choice}_available'):
            # Preprocess text for PyTorch models
            processed_text = text_process(text)
            # Get model
            model = models[model_choice]
            device = models['device']


            model = model.to(device)

            # Make prediction
            with torch.no_grad():
                probabilities = F.softmax(processed_text, dim=1).cpu().numpy()[0]
                prediction = torch.argmax(processed_text, dim=1).cpu().numpy()[0]

        # Traditional ML models (existing code)
        elif model_choice == "pipeline" and models.get('pipeline_available'):
            # Use the complete pipeline (Logistic Regression)
            prediction = models['pipeline'].predict([text])[0]
            probabilities = models['pipeline'].predict_proba([text])[0]

        elif model_choice == "logistic_regression":
            if models.get('pipeline_available'):
                # Use pipeline for LR
                prediction = models['pipeline'].predict([text])[0]
                probabilities = models['pipeline'].predict_proba([text])[0]
            elif models.get('vectorizer_available') and models.get('lr_available'):
                # Use individual components
                X = models['vectorizer'].transform([text])
                prediction = models['logistic_regression'].predict(X)[0]
                probabilities = models['logistic_regression'].predict_proba(X)[0]

        elif model_choice == "naive_bayes":
            if models.get('vectorizer_available') and models.get('nb_available'):
                # Use individual components for NB
                X = models['vectorizer'].transform([text])
                prediction = models['naive_bayes'].predict(X)[0]
                probabilities = models['naive_bayes'].predict_proba(X)[0]

        elif model_choice == "svm" and models.get('svm_available'):
            X = models['vectorizer'].transform([text])
            prediction = models['svm'].predict(X)[0]
            probabilities = models['svm'].predict_proba(X)[0]

        elif model_choice == "dt" and models.get('dt_available'):
            X = models['vectorizer'].transform([text])
            prediction = models['dt'].predict(X)[0]
            probabilities = models['dt'].predict_proba(X)[0]

        elif model_choice == "adb" and models.get('adb_available'):
            X = models['vectorizer'].transform([text])
            prediction = models['adb'].predict(X)[0]
            probabilities = models['adb'].predict_proba(X)[0]

        if prediction is not None and probabilities is not None:
            # Convert to readable format
            class_names = ['Negative', 'Positive']
            prediction_label = class_names[prediction]
            return prediction_label, probabilities
        else:
            return None, None

    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error(f"Model choice: {model_choice}")
        st.error(f"Available models: {[k for k, v in models.items() if isinstance(v, bool) and v]}")
        return None, None


def get_available_models(models):
    """Get list of available models for selection"""
    available = []

    if models is None:
        return available

    # PyTorch models
    if models.get('lstm_available'):
        available.append(("lstm", "🧠 LSTM Neural Network"))

    if models.get('rnn_available'):
        available.append(("rnn", "🔄 RNN Neural Network"))

    if models.get('cnn_available'):
        available.append(("cnn", "🌐 CNN Neural Network"))

    # Traditional ML models
    if models.get('pipeline_available'):
        available.append(("logistic_regression", "📈 Logistic Regression (Pipeline)"))
    elif models.get('vectorizer_available') and models.get('lr_available'):
        available.append(("logistic_regression", "📈 Logistic Regression (Individual)"))

    if models.get('vectorizer_available') and models.get('nb_available'):
        available.append(("naive_bayes", "🎯 Multinomial Naive Bayes"))

    if models.get('svm_available'):
        available.append(("svm", "🎰 SVM Pipeline"))

    if models.get('dt_available'):
        available.append(("dt", "💭 Decision Tree Pipeline"))

    if models.get('adb_available'):
        available.append(("adb", "🔋 Adaboost Pipeline"))

    return available


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Choose what you want to do:")

page = st.sidebar.selectbox(
    "Select Page:",
    ["🏠 Home", "🔮 Single Prediction", "📁 Batch Processing", "⚖️ Model Comparison", "📊 Model Info", "❓ Help"]
)

# Load models
models = load_models()

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🤖 ML Text Classification App</h1>', unsafe_allow_html=True)

    st.markdown("""
    Welcome to your enhanced machine learning web application! This app demonstrates sentiment analysis
    using multiple trained models: **PyTorch Deep Learning Models** (LSTM, RNN, CNN) and **Traditional ML Models** 
    (Logistic Regression, Multinomial Naive Bayes, SVM, Decision Tree, AdaBoost).
    """)

    # App overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🔮 Single Prediction
        - Enter text manually
        - Choose between models
        - Get instant predictions
        - See confidence scores
        """)

    with col2:
        st.markdown("""
        ### 📁 Batch Processing
        - Upload text files
        - Process multiple texts
        - Compare model performance
        - Download results
        """)

    with col3:
        st.markdown("""
        ### ⚖️ Model Comparison
        - Compare different models
        - Side-by-side results
        - Agreement analysis
        - Performance metrics
        """)

    # Model status
    st.subheader("📋 Model Status")
    if models:
        st.success("✅ Models loaded successfully!")

        # PyTorch Models Status
        st.subheader("🧠 Deep Learning Models")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if models.get('embedding_available'):
                st.info("**📊 Embeddings**\n✅ Available")
            else:
                st.warning("**📊 Embeddings**\n❌ Not Available")

        with col2:
            if models.get('lstm_available'):
                st.info("**🧠 LSTM**\n✅ Available")
            else:
                st.warning("**🧠 LSTM**\n❌ Not Available")

        with col3:
            if models.get('rnn_available'):
                st.info("**🔄 RNN**\n✅ Available")
            else:
                st.warning("**🔄 RNN**\n❌ Not Available")

        with col4:
            if models.get('cnn_available'):
                st.info("**🌐 CNN**\n✅ Available")
            else:
                st.warning("**🌐 CNN**\n❌ Not Available")

        # Traditional ML Models Status
        st.subheader("📈 Traditional ML Models")
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            if models.get('pipeline_available'):
                st.info("**📈 Logistic Regression**\n✅ Pipeline Available")
            elif models.get('lr_available') and models.get('vectorizer_available'):
                st.info("**📈 Logistic Regression**\n✅ Individual Components")
            else:
                st.warning("**📈 Logistic Regression**\n❌ Not Available")

        with col2:
            if models.get('nb_available') and models.get('vectorizer_available'):
                st.info("**🎯 Multinomial NB**\n✅ Available")
            else:
                st.warning("**🎯 Multinomial NB**\n❌ Not Available")

        with col3:
            if models.get('svm_available'):
                st.info("**🎰 SVM**\n✅ Available")
            else:
                st.warning("**🎰 SVM**\n❌ Not Available")

        with col4:
            if models.get('dt_available'):
                st.info("**💭 Decision Tree**\n✅ Available")
            else:
                st.warning("**💭 Decision Tree**\n❌ Not Available")

        with col5:
            if models.get('adb_available'):
                st.info("**🔋 AdaBoost**\n✅ Available")
            else:
                st.warning("**🔋 AdaBoost**\n❌ Not Available")

        with col6:
            if models.get('vectorizer_available'):
                st.info("**🔤 TF-IDF Vectorizer**\n✅ Available")
            else:
                st.warning("**🔤 TF-IDF Vectorizer**\n❌ Not Available")

    else:
        st.error("❌ Models not loaded. Please check model files.")

# ============================================================================
# SINGLE PREDICTION PAGE
# ============================================================================

elif page == "🔮 Single Prediction":
    st.header("🔮 Make a Single Prediction")
    st.markdown("Enter text below and select a model to get sentiment predictions.")

    if models:
        available_models = get_available_models(models)

        if available_models:
            # Model selection
            model_choice = st.selectbox(
                "Choose a model:",
                options=[model[0] for model in available_models],
                format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
            )

            # Text input
            user_input = st.text_area(
                "Enter your text here:",
                placeholder="Type or paste your text here (e.g., product review, feedback, comment)...",
                height=150
            )

            # Character count
            if user_input:
                st.caption(f"Character count: {len(user_input)} | Word count: {len(user_input.split())}")

            # Example texts
            with st.expander("📝 Try these example texts"):
                examples = [
                    "This product is absolutely amazing! Best purchase I've made this year.",
                    "Terrible quality, broke after one day. Complete waste of money.",
                    "It's okay, nothing special but does the job.",
                    "Outstanding customer service and fast delivery. Highly recommend!",
                    "I love this movie! It's absolutely fantastic and entertaining."
                ]

                col1, col2 = st.columns(2)
                for i, example in enumerate(examples):
                    with col1 if i % 2 == 0 else col2:
                        if st.button(f"Example {i + 1}", key=f"example_{i}"):
                            st.session_state.user_input = example
                            st.rerun()

            # Use session state for user input
            if 'user_input' in st.session_state:
                user_input = st.session_state.user_input

            # Prediction button
            if st.button("🚀 Predict", type="primary"):
                if user_input.strip():
                    with st.spinner('Analyzing sentiment...'):
                        prediction, probabilities = make_prediction(user_input, model_choice, models)

                        if prediction and probabilities is not None:
                            # Display prediction
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                if prediction == "Positive":
                                    st.success(f"🎯 Prediction: **{prediction} Sentiment**")
                                else:
                                    st.error(f"🎯 Prediction: **{prediction} Sentiment**")

                            with col2:
                                confidence = max(probabilities)
                                st.metric("Confidence", f"{confidence:.1%}")

                            # Create probability chart
                            st.subheader("📊 Prediction Probabilities")

                            # Detailed probabilities
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("😞 Negative", f"{probabilities[0]:.1%}")
                            with col2:
                                st.metric("😊 Positive", f"{probabilities[1]:.1%}")

                            # Bar chart
                            class_names = ['Negative', 'Positive']
                            prob_df = pd.DataFrame({
                                'Sentiment': class_names,
                                'Probability': probabilities
                            })
                            st.bar_chart(prob_df.set_index('Sentiment'), height=300)

                        else:
                            st.error("Failed to make prediction")
                else:
                    st.warning("Please enter some text to classify!")
        else:
            st.error("No models available for prediction.")
    else:
        st.warning("Models not loaded. Please check the model files.")

# ============================================================================
# BATCH PROCESSING PAGE
# ============================================================================

elif page == "📁 Batch Processing":
    st.header("📁 Upload File for Batch Processing")
    st.markdown("Upload a text file or CSV to process multiple texts at once.")

    if models:
        available_models = get_available_models(models)

        if available_models:
            # File upload
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'csv'],
                help="Upload a .txt file (one text per line) or .csv file (text in first column)"
            )

            if uploaded_file:
                # Model selection
                model_choice = st.selectbox(
                    "Choose model for batch processing:",
                    options=[model[0] for model in available_models],
                    format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
                )

                # Process file
                if st.button("📊 Process File"):
                    try:
                        # Read file content
                        if uploaded_file.type == "text/plain":
                            content = str(uploaded_file.read(), "utf-8")
                            texts = [line.strip() for line in content.split('\n') if line.strip()]
                        else:  # CSV
                            df = pd.read_csv(uploaded_file)
                            texts = df.iloc[:, 0].astype(str).tolist()

                        if not texts:
                            st.error("No text found in file")
                        else:
                            st.info(f"Processing {len(texts)} texts...")

                            # Process all texts
                            results = []
                            progress_bar = st.progress(0)

                            for i, text in enumerate(texts):
                                if text.strip():
                                    prediction, probabilities = make_prediction(text, model_choice, models)

                                    if prediction and probabilities is not None:
                                        results.append({
                                            'Text': text[:100] + "..." if len(text) > 100 else text,
                                            'Full_Text': text,
                                            'Prediction': prediction,
                                            'Confidence': f"{max(probabilities):.1%}",
                                            'Negative_Prob': f"{probabilities[0]:.1%}",
                                            'Positive_Prob': f"{probabilities[1]:.1%}"
                                        })

                                progress_bar.progress((i + 1) / len(texts))

                            if results:
                                # Display results
                                st.success(f"✅ Processed {len(results)} texts successfully!")

                                results_df = pd.DataFrame(results)

                                # Summary statistics
                                st.subheader("📊 Summary Statistics")
                                col1, col2, col3, col4 = st.columns(4)

                                positive_count = sum(1 for r in results if r['Prediction'] == 'Positive')
                                negative_count = len(results) - positive_count
                                avg_confidence = np.mean([float(r['Confidence'].strip('%')) for r in results])

                                with col1:
                                    st.metric("Total Processed", len(results))
                                with col2:
                                    st.metric("😊 Positive", positive_count)
                                with col3:
                                    st.metric("😞 Negative", negative_count)
                                with col4:
                                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")

                                # Results preview
                                st.subheader("📋 Results Preview")
                                st.dataframe(
                                    results_df[['Text', 'Prediction', 'Confidence']],
                                    use_container_width=True
                                )

                                # Download option
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Full Results",
                                    data=csv,
                                    file_name=f"predictions_{model_choice}_{uploaded_file.name}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("No valid texts could be processed")

                    except Exception as e:
                        st.error(f"Error processing file: {e}")
            else:
                st.info("Please upload a file to get started.")

                # Show example file formats
                with st.expander("📄 Example File Formats"):
                    st.markdown("""
                    **Text File (.txt):**
                    ```
                    This product is amazing!
                    Terrible quality, very disappointed
                    Great service and fast delivery
                    ```

                    **CSV File (.csv):**
                    ```
                    text,category
                    "Amazing product, love it!",review
                    "Poor quality, not satisfied",review
                    ```
                    """)
        else:
            st.error("No models available for batch processing.")
    else:
        st.warning("Models not loaded. Please check the model files.")

# ============================================================================
# MODEL COMPARISON PAGE
# ============================================================================

elif page == "⚖️ Model Comparison":
    st.header("⚖️ Compare Models")
    st.markdown("Compare predictions from different models on the same text.")

    if models:
        available_models = get_available_models(models)

        if len(available_models) >= 2:
            # Text input for comparison
            comparison_text = st.text_area(
                "Enter text to compare models:",
                placeholder="Enter text to see how different models perform...",
                height=100
            )

            if st.button("📊 Compare All Models") and comparison_text.strip():
                st.subheader("🔍 Model Comparison Results")

                # Get predictions from all available models
                comparison_results = []

                for model_key, model_name in available_models:
                    prediction, probabilities = make_prediction(comparison_text, model_key, models)

                    if prediction and probabilities is not None:
                        comparison_results.append({
                            'Model': model_name,
                            'Prediction': prediction,
                            'Confidence': f"{max(probabilities):.1%}",
                            'Negative %': f"{probabilities[0]:.1%}",
                            'Positive %': f"{probabilities[1]:.1%}",
                            'Raw_Probs': probabilities
                        })

                if comparison_results:
                    # Comparison table
                    comparison_df = pd.DataFrame(comparison_results)
                    st.table(comparison_df[['Model', 'Prediction', 'Confidence', 'Negative %', 'Positive %']])

                    # Agreement analysis
                    predictions = [r['Prediction'] for r in comparison_results]
                    if len(set(predictions)) == 1:
                        st.success(f"✅ All models agree: **{predictions[0]} Sentiment**")
                    else:
                        st.warning("⚠️ Models disagree on prediction")
                        for result in comparison_results:
                            model_name = result['Model'].split(' ')[1] if ' ' in result['Model'] else result['Model']
                            st.write(f"- {model_name}: {result['Prediction']}")

                    # Side-by-side probability charts
                    st.subheader("📊 Detailed Probability Comparison")

                    cols = st.columns(len(comparison_results))

                    for i, result in enumerate(comparison_results):
                        with cols[i]:
                            model_name = result['Model']
                            st.write(f"**{model_name}**")

                            chart_data = pd.DataFrame({
                                'Sentiment': ['Negative', 'Positive'],
                                'Probability': result['Raw_Probs']
                            })
                            st.bar_chart(chart_data.set_index('Sentiment'))

                else:
                    st.error("Failed to get predictions from models")

        elif len(available_models) == 1:
            st.info("Only one model available. Use Single Prediction page for detailed analysis.")

        else:
            st.error("No models available for comparison.")
    else:
        st.warning("Models not loaded. Please check the model files.")

# ============================================================================
# MODEL INFO PAGE
# ============================================================================

elif page == "📊 Model Info":
    st.header("📊 Model Information")

    if models:
        st.success("✅ Models are loaded and ready!")

        # PyTorch Models section
        st.subheader("🧠 Deep Learning Models")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            ### 🧠 LSTM Model
            **Type:** Long Short-Term Memory Network
            **Architecture:** Bidirectional LSTM with embedding layer
            **Features:** Pre-trained word embeddings

            **Strengths:**
            - Captures long-term dependencies
            - Handles sequential information well
            - Good for context understanding
            - Memory cells prevent vanishing gradients
            """)

        with col2:
            st.markdown("""
            ### 🔄 RNN Model
            **Type:** Recurrent Neural Network
            **Architecture:** Multi-layer RNN with embedding layer
            **Features:** Pre-trained word embeddings

            **Strengths:**
            - Sequential data processing
            - Simpler than LSTM
            - Faster training
            - Good baseline for sequence tasks
            """)

        with col3:
            st.markdown("""
            ### 🌐 CNN Model
            **Type:** Convolutional Neural Network
            **Architecture:** 1D CNN with multiple filter sizes
            **Features:** Pre-trained word embeddings

            **Strengths:**
            - Captures local patterns
            - Parallel processing
            - Good for n-gram features
            - Fast inference
            """)

        # Traditional ML Models section
        st.subheader("📈 Traditional ML Models")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 📈 Logistic Regression
            **Type:** Linear Classification Model
            **Algorithm:** Logistic Regression with L2 regularization
            **Features:** TF-IDF vectors (unigrams + bigrams)

            **Strengths:**
            - Fast prediction
            - Interpretable coefficients
            - Good baseline performance
            - Handles sparse features well
            """)

            st.markdown("""
            ### 🎯 Multinomial Naive Bayes
            **Type:** Probabilistic Classification Model
            **Algorithm:** Multinomial Naive Bayes
            **Features:** TF-IDF vectors (unigrams + bigrams)

            **Strengths:**
            - Fast training and prediction
            - Works well with small datasets
            - Good performance on text classification
            - Natural probabilistic outputs
            """)

        with col2:
            st.markdown("""
            ### 🎰 Support Vector Machine
            **Type:** Margin-based Classification Model
            **Algorithm:** SVM with RBF/Linear kernel
            **Features:** TF-IDF vectors

            **Strengths:**
            - Effective in high dimensions
            - Memory efficient
            - Versatile (different kernels)
            - Good generalization
            """)

            st.markdown("""
            ### 💭 Decision Tree & 🔋 AdaBoost
            **Type:** Tree-based Models
            **Algorithm:** Decision Tree / Adaptive Boosting
            **Features:** TF-IDF vectors

            **Strengths:**
            - Interpretable decisions
            - Handle non-linear patterns
            - Feature importance
            - Robust to outliers
            """)

        # Embedding information
        if models.get('embedding_available'):
            st.subheader("📊 Word Embeddings")
            st.markdown(f"""
            **Embedding Matrix Shape:** {models['vocab_size']} vocabulary × {models['embedding_dim']} dimensions
            **Type:** Pre-trained word embeddings (likely GloVe or Word2Vec)
            **Usage:** Used by all PyTorch models for text representation
            **Benefits:** Captures semantic relationships between words
            """)

        # Feature engineering info
        st.subheader("🔤 Feature Engineering")
        st.markdown("""
        **For Traditional ML Models:**
        - **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
        - **Max Features:** 5,000 most important terms
        - **N-grams:** Unigrams (1-word) and Bigrams (2-word phrases)
        - **Min Document Frequency:** 2 (terms must appear in at least 2 documents)
        - **Stop Words:** English stop words removed

        **For PyTorch Models:**
        - **Tokenization:** Text to sequence of word indices
        - **Embeddings:** Pre-trained word vectors
        - **Sequence Length:** Fixed length with padding/truncation
        - **Preprocessing:** Text cleaning and normalization
        """)

        # File status
        st.subheader("📁 Model Files Status")
        file_status = []

        # PyTorch files
        pytorch_files = [
            ("embedding_gv.npy", "Word Embeddings", models.get('embedding_available', False)),
            ("word_to_idx.pkl", "Vocabulary Mapping", models.get('word_to_idx_available', False)),
            ("LSTM_gv.pt", "LSTM Model", models.get('lstm_available', False)),
            ("RNN_gv.pt", "RNN Model", models.get('rnn_available', False)),
            ("CNN_gv.pt", "CNN Model", models.get('cnn_available', False))
        ]

        # Traditional ML files
        traditional_files = [
            ("sentiment_analysis_pipeline.pkl", "Complete LR Pipeline", models.get('pipeline_available', False)),
            ("tfidf_vectorizer.pkl", "TF-IDF Vectorizer", models.get('vectorizer_available', False)),
            ("logistic_regression_model.pkl", "LR Classifier", models.get('lr_available', False)),
            ("multinomial_nb_model.pkl", "NB Classifier", models.get('nb_available', False)),
            ("svm_model.pkl", "SVM Classifier", models.get('svm_available', False)),
            ("dt_model.pkl", "Decision Tree", models.get('dt_available', False)),
            ("adb_model.pkl", "AdaBoost", models.get('adb_available', False))
        ]

        for filename, description, status in pytorch_files + traditional_files:
            file_status.append({
                "File": filename,
                "Description": description,
                "Status": "✅ Loaded" if status else "❌ Not Found"
            })

        st.table(pd.DataFrame(file_status))

        # Device information
        if models.get('device'):
            device_info = models['device']
            st.subheader("🖥️ Compute Device")
            if device_info.type == 'cuda':
                st.success(f"🚀 Using GPU: {device_info}")
            else:
                st.info(f"💻 Using CPU: {device_info}")

        # Training information
        st.subheader("📚 Training Information")
        st.markdown("""
        **Dataset:** Product Review Sentiment Analysis
        - **Classes:** Positive and Negative sentiment
        - **Preprocessing:** Text cleaning, tokenization, vectorization
        - **Training:** All models trained on same dataset for fair comparison
        - **Evaluation:** Cross-validation and test set performance
        """)

    else:
        st.warning("Models not loaded. Please check model files.")

# ============================================================================
# HELP PAGE
# ============================================================================

elif page == "❓ Help":
    st.header("❓ How to Use This App")

    with st.expander("🔮 Single Prediction"):
        st.write("""
        1. **Select a model** from the dropdown (LSTM, RNN, CNN, or traditional ML models)
        2. **Enter text** in the text area (product reviews, comments, feedback)
        3. **Click 'Predict'** to get sentiment analysis results
        4. **View results:** prediction, confidence score, and probability breakdown
        5. **Try examples:** Use the provided example texts to test the models
        """)

    with st.expander("📁 Batch Processing"):
        st.write("""
        1. **Prepare your file:**
           - **.txt file:** One text per line
           - **.csv file:** Text in the first column
        2. **Upload the file** using the file uploader
        3. **Select a model** for processing
        4. **Click 'Process File'** to analyze all texts
        5. **Download results** as CSV file with predictions and probabilities
        """)

    with st.expander("⚖️ Model Comparison"):
        st.write("""
        1. **Enter text** you want to analyze
        2. **Click 'Compare All Models'** to get predictions from all available models
        3. **View comparison table** showing predictions and confidence scores
        4. **Analyze agreement:** See if models agree or disagree
        5. **Compare probabilities:** Side-by-side probability charts
        """)

    with st.expander("🔧 Troubleshooting"):
        st.write("""
        **Common Issues and Solutions:**

        **PyTorch Models not loading:**
        - Ensure PyTorch model files (.pt) are in the root directory
        - Check that embedding_gv.npy file exists
        - Verify word_to_idx.pkl file is available
        - Make sure PyTorch is installed: `pip install torch`

        **Traditional Models not loading:**
        - Ensure model files (.pkl) are in the 'models/' directory
        - Check that required files exist:
          - tfidf_vectorizer.pkl (required)
          - sentiment_analysis_pipeline.pkl (for LR pipeline)
          - logistic_regression_model.pkl (for LR individual)
          - multinomial_nb_model.pkl (for NB model)

        **Prediction errors:**
        - Make sure input text is not empty
        - Try shorter texts if getting memory errors
        - Check that text contains readable characters
        - For PyTorch models, ensure vocabulary mapping is available

        **File upload issues:**
        - Ensure file format is .txt or .csv
        - Check file encoding (should be UTF-8)
        - Verify CSV has text in the first column

        **Memory issues:**
        - For large batch processing, try smaller files
        - PyTorch models may require more memory
        - Check available RAM and GPU memory
        """)

    # System information
    st.subheader("💻 Required Project Structure")
    st.code("""
    streamlit_ml_app/
    ├── app.py                              # Main application
    ├── requirements.txt                    # Dependencies
    ├── embedding_gv.npy                   # Word embeddings (REQUIRED for PyTorch)
    ├── word_to_idx.pkl                    # Vocabulary mapping (REQUIRED for PyTorch)
    ├── LSTM_gv.pt                         # LSTM model weights
    ├── RNN_gv.pt                          # RNN model weights
    ├── CNN_gv.pt                          # CNN model weights
    ├── models/                            # Traditional ML model files
    │   ├── sentiment_analysis_pipeline.pkl # LR complete pipeline
    │   ├── tfidf_vectorizer.pkl           # Feature extraction
    │   ├── logistic_regression_model.pkl  # LR classifier
    │   ├── multinomial_nb_model.pkl       # NB classifier
    │   ├── svm_model.pkl                  # SVM classifier
    │   ├── dt_model.pkl                   # Decision Tree
    │   └── adb_model.pkl                  # AdaBoost
    └── sample_data/                       # Sample files
        ├── sample_texts.txt
        └── sample_data.csv
    """)

    st.subheader("📦 Additional Requirements")
    st.code("""
    # Add these to requirements.txt for PyTorch support:
    torch>=1.9.0
    numpy>=1.21.0

    # Create word_to_idx.pkl file:
    # This should be created during your model training process
    # Example structure:
    word_to_idx = {
        '<PAD>': 0,
        '<UNK>': 1,
        'the': 2,
        'and': 3,
        # ... more vocabulary
    }

    # Save with:
    import pickle
    with open('word_to_idx.pkl', 'wb') as f:
        pickle.dump(word_to_idx, f)
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 App Information")
st.sidebar.info("""
**Enhanced ML Text Classification App**
Built with Streamlit

**Deep Learning Models:** 
- 🧠 LSTM Neural Network
- 🔄 RNN Neural Network  
- 🌐 CNN Neural Network

**Traditional ML Models:**
- 📈 Logistic Regression
- 🎯 Multinomial Naive Bayes
- 🎰 Support Vector Machine
- 💭 Decision Tree
- 🔋 AdaBoost

**Frameworks:** PyTorch, scikit-learn
**Deployment:** Streamlit Cloud Ready
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    Built with ❤️ using Streamlit | Enhanced ML Text Classification Demo | By Maaz Amjad<br>
    <small>As a part of the courses series **Introduction to Large Language Models/Intro to AI Agents**</small><br>
    <small>This app demonstrates sentiment analysis using both deep learning and traditional ML models</small>
</div>
""", unsafe_allow_html=True)