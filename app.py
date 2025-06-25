# STREAMLIT ML CLASSIFICATION APP - DUAL MODEL SUPPORT
# =====================================================

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
# MODEL LOADING SECTION
# ============================================================================
# app.py - NEW PROJECT




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



@st.cache_resource
def load_models():
    models = {}
    
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
        
        # Check if at least one complete setup is available
        pipeline_ready = models['pipeline_available']
        individual_ready = models['vectorizer_available'] and (models['lr_available'] or models['nb_available'])
        
        if not (pipeline_ready or individual_ready):
            st.error("No complete model setup found!")
            return None
        
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
        
        if model_choice == "pipeline" and models.get('pipeline_available'):
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
            prediction = models['svm'].predict(text)[0]
            probabilities = models['svm'].predict_proba(text)[0]
        elif model_choice == "dt" and models.get('dt_available'):
            prediction = models['dt'].predict(text)[0]
            probabilities = models['dt'].predict_proba(text)[0]
        elif model_choice == "adb" and models.get('adb_available'):
            prediction = models['adb'].predict(text)[0]
            probabilities = models['adb'].predict_proba(text)[0]


        
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
        available.append(("adb", "🔋Adaboost Pipeline"))
    
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
    Welcome to your machine learning web application! This app demonstrates sentiment analysis
    using multiple trained models: **Logistic Regression** and **Multinomial Naive Bayes**.
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
        
        col1, col2, col3 = st.columns(3)
        
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
                        if st.button(f"Example {i+1}", key=f"example_{i}"):
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
        
        # Model details
        st.subheader("🔧 Available Models")
        
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
            
        with col2:
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
        
        # Feature engineering info
        st.subheader("🔤 Feature Engineering")
        st.markdown("""
        **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
        - **Max Features:** 5,000 most important terms
        - **N-grams:** Unigrams (1-word) and Bigrams (2-word phrases)
        - **Min Document Frequency:** 2 (terms must appear in at least 2 documents)
        - **Stop Words:** English stop words removed
        """)
        
        # File status
        st.subheader("📁 Model Files Status")
        file_status = []
        
        files_to_check = [
            ("sentiment_analysis_pipeline.pkl", "Complete LR Pipeline", models.get('pipeline_available', False)),
            ("tfidf_vectorizer.pkl", "TF-IDF Vectorizer", models.get('vectorizer_available', False)),
            ("logistic_regression_model.pkl", "LR Classifier", models.get('lr_available', False)),
            ("multinomial_nb_model.pkl", "NB Classifier", models.get('nb_available', False))
        ]
        
        for filename, description, status in files_to_check:
            file_status.append({
                "File": filename,
                "Description": description,
                "Status": "✅ Loaded" if status else "❌ Not Found"
            })
        
        st.table(pd.DataFrame(file_status))
        
        # Training information
        st.subheader("📚 Training Information")
        st.markdown("""
        **Dataset:** Product Review Sentiment Analysis
        - **Classes:** Positive and Negative sentiment
        - **Preprocessing:** Text cleaning, tokenization, TF-IDF vectorization
        - **Training:** Both models trained on same feature set for fair comparison
        """)
        
    else:
        st.warning("Models not loaded. Please check model files in the 'models/' directory.")

# ============================================================================
# HELP PAGE
# ============================================================================

elif page == "❓ Help":
    st.header("❓ How to Use This App")
    
    with st.expander("🔮 Single Prediction"):
        st.write("""
        1. **Select a model** from the dropdown (Logistic Regression or Multinomial Naive Bayes)
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
        2. **Click 'Compare All Models'** to get predictions from both models
        3. **View comparison table** showing predictions and confidence scores
        4. **Analyze agreement:** See if models agree or disagree
        5. **Compare probabilities:** Side-by-side probability charts
        """)
    
    with st.expander("🔧 Troubleshooting"):
        st.write("""
        **Common Issues and Solutions:**
        
        **Models not loading:**
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
        
        **File upload issues:**
        - Ensure file format is .txt or .csv
        - Check file encoding (should be UTF-8)
        - Verify CSV has text in the first column
        """)
    
    # System information
    st.subheader("💻 Your Project Structure")
    st.code("""
    streamlit_ml_app/
    ├── app.py                              # Main application
    ├── requirements.txt                    # Dependencies
    ├── models/                            # Model files
    │   ├── sentiment_analysis_pipeline.pkl # LR complete pipeline
    │   ├── tfidf_vectorizer.pkl           # Feature extraction
    │   ├── logistic_regression_model.pkl  # LR classifier
    │   └── multinomial_nb_model.pkl       # NB classifier
    └── sample_data/                       # Sample files
        ├── sample_texts.txt
        └── sample_data.csv
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 App Information")
st.sidebar.info("""
**ML Text Classification App**
Built with Streamlit

**Models:** 
- 📈 Logistic Regression
- 🎯 Multinomial Naive Bayes

**Framework:** scikit-learn
**Deployment:** Streamlit Cloud Ready
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    Built with ❤️ using Streamlit | Machine Learning Text Classification Demo | By Maaz Amjad<br>
    <small>As a part of the courses series **Introduction to Large Language Models/Intro to AI Agents**</small><br>
    <small>This app demonstrates sentiment analysis using trained ML models</small>
</div>
""", unsafe_allow_html=True)