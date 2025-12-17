import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance, ImageOps
import joblib
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from scipy.ndimage import gaussian_filter

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="EMNIST Character Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_models():
    """Load all available models and label mapping."""
    models = {
        'de_raw': None,
        'de_eng': None,
        'ra_raw': None,
        'ra_eng': None
    }
    label_mapping = None
    pca_model = None
    
    try:
        # Load label mapping
        if os.path.exists('label_mapping.joblib'):
            label_mapping = joblib.load('label_mapping.joblib')
        
        # Load models if they exist
        if os.path.exists('de_raw.joblib'):
            models['de_raw'] = joblib.load('de_raw.joblib')
        if os.path.exists('de_eng.joblib'):
            models['de_eng'] = joblib.load('de_eng.joblib')
        if os.path.exists('ra_raw.joblib'):
            models['ra_raw'] = joblib.load('ra_raw.joblib')
        if os.path.exists('ra_eng.joblib'):
            models['ra_eng'] = joblib.load('ra_eng.joblib')
            
        # Load PCA model if it exists
        if os.path.exists('pca_model.joblib'):
            pca_model = joblib.load('pca_model.joblib')
            
        return models, label_mapping, pca_model, any(models.values())
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return models, None, None, False

def preprocess_image(image, target_size=(28, 28)):
    """Preprocess uploaded image to match training data format."""
    try:
        # Handle file path input
        if isinstance(image, str) and os.path.exists(image):
            image = Image.open(image)
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:  # RGB
                image = np.mean(image, axis=2).astype(np.uint8)
            image = Image.fromarray(image.astype(np.uint8))
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize
        image = image.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        return img_array
        
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        raise

def extract_features_single_image(img_array, pca_model=None):
    """Extract HOG, statistical, and PCA features from a single image.
    
    Returns:
        combined_features: numpy array of shape (1, n_features)
        features_dict: dictionary with individual feature components
    """
    features_dict = {}
    
    # Ensure we have a 2D numpy array
    if hasattr(img_array, 'mode'):  # It's a PIL Image
        img_array = np.array(img_array.convert('L')).astype(np.float32) / 255.0
    
    if len(img_array.shape) > 2:
        img_array = np.mean(img_array, axis=2)
    
    # 1. HOG Features
    hog_features = hog(
        img_array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        channel_axis=None
    )
    features_dict['hog'] = hog_features
    
    # 2. Statistical Features
    mean_val = np.mean(img_array)
    std_val = np.std(img_array)
    max_val = np.max(img_array)
    min_val = np.min(img_array)
    
    binary = (img_array > 0.5).astype(np.uint8)
    vertical_projection = np.sum(binary, axis=0)
    horizontal_projection = np.sum(binary, axis=1)
    
    width = np.sum(vertical_projection > 0)
    height = np.sum(horizontal_projection > 0)
    aspect_ratio = width / height if height > 0 else 1.0
    
    stat_features = np.array([mean_val, std_val, max_val, min_val, aspect_ratio])
    features_dict['stats'] = stat_features
    
    # 3. PCA Features
    if pca_model is not None:
        try:
            img_flat = img_array.reshape(1, -1)
            pca_features = pca_model.transform(img_flat).flatten()
            features_dict['pca'] = pca_features
        except Exception as e:
            print(f"PCA transform error: {str(e)}")
            pca_features = img_array.flatten()[:50]
            features_dict['pca'] = pca_features
    else:
        pca_features = img_array.flatten()[:50]
        features_dict['pca'] = pca_features
    
    # Combine all features
    combined_features = np.concatenate([
        features_dict['hog'],
        features_dict['stats'],
        features_dict['pca']
    ]).reshape(1, -1)
    
    return combined_features, features_dict

def prepare_raw_features(images):
    """Flatten images without feature engineering"""
    if len(images.shape) == 2:  # Single image
        return images.reshape(1, -1)
    return images.reshape(images.shape[0], -1)

def predict_with_confidence(model, features, label_mapping=None):
    """Make prediction and return confidence scores."""
    # Ensure features is 2D
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    prediction_idx = model.predict(features)[0]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
    else:
        # For Decision Tree without predict_proba
        probabilities = np.zeros(len(model.classes_))
        probabilities[prediction_idx] = 1.0
    
    # Create confidence dictionary using the REAL labels if available
    confidence_dict = {}
    for i, prob in enumerate(probabilities):
        if label_mapping and i in label_mapping:
            label_name = str(label_mapping[i])
        else:
            label_name = f"Class {i}"
        confidence_dict[label_name] = prob
    
    # Sort by confidence
    confidence_dict = dict(sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True))
    
    # Get the final label name
    if label_mapping and prediction_idx in label_mapping:
        final_label = str(label_mapping[prediction_idx])
    else:
        final_label = str(prediction_idx)
        
    return final_label, confidence_dict

# ============================================================================
# AUGMENTATION FUNCTIONS
# ============================================================================

def apply_augmentations(image):
    """Apply various augmentations to the image."""
    augmented_images = {}
    augmented_images['Original'] = image
    
    enhancer = ImageEnhance.Brightness(image)
    augmented_images['Brightness +30%'] = enhancer.enhance(1.3)
    augmented_images['Brightness -30%'] = enhancer.enhance(0.7)
    
    enhancer = ImageEnhance.Contrast(image)
    augmented_images['Contrast +30%'] = enhancer.enhance(1.3)
    augmented_images['Contrast -30%'] = enhancer.enhance(0.7)
    
    augmented_images['Rotate 15°'] = image.rotate(15, fillcolor=255)
    augmented_images['Rotate -15°'] = image.rotate(-15, fillcolor=255)
    
    augmented_images['Horizontal Flip'] = ImageOps.mirror(image)
    augmented_images['Vertical Flip'] = ImageOps.flip(image)
    
    img_array = np.array(image.convert('L')) / 255.0
    blurred = gaussian_filter(img_array, sigma=1.0)
    augmented_images['Gaussian Blur'] = Image.fromarray((blurred * 255).astype(np.uint8))
    
    enhancer = ImageEnhance.Sharpness(image)
    augmented_images['Sharpen'] = enhancer.enhance(2.0)
    
    img_array = np.array(image.convert('L'))
    noise = np.random.normal(0, 25, img_array.shape)
    noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    augmented_images['Gaussian Noise'] = Image.fromarray(noisy)
    
    width, height = image.size
    crop_size = int(min(width, height) * 0.8)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    cropped = image.crop((left, top, left + crop_size, top + crop_size))
    augmented_images['Center Crop 80%'] = cropped.resize((width, height))
    
    return augmented_images

def visualize_augmentations(augmented_images):
    n_images = len(augmented_images)
    n_cols = 4
    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()
    
    for idx, (name, img) in enumerate(augmented_images.items()):
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(name, fontsize=10)
        axes[idx].axis('off')
    
    for idx in range(len(augmented_images), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_hog_features(img_array):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(img_array, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    configs = [
        {'orientations': 9, 'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2)},
        {'orientations': 12, 'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2)},
        {'orientations': 9, 'pixels_per_cell': (4, 4), 'cells_per_block': (2, 2)},
        {'orientations': 9, 'pixels_per_cell': (8, 8), 'cells_per_block': (3, 3)},
        {'orientations': 6, 'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2)},
    ]
    
    for idx, config in enumerate(configs):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        fd, hog_image = hog(img_array, orientations=config['orientations'],
                           pixels_per_cell=config['pixels_per_cell'],
                           cells_per_block=config['cells_per_block'],
                           visualize=True, channel_axis=None)
        axes[row, col].imshow(hog_image, cmap='gray')
        title = f"Orient={config['orientations']}, Cell={config['pixels_per_cell'][0]}, Block={config['cells_per_block'][0]}"
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis('off')
    
    plt.suptitle('HOG Feature Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_confidence_scores(confidence_dict, top_n=10):
    """Plot confidence scores - showing only top predictions like old code."""
    top_predictions = dict(list(confidence_dict.items())[:top_n])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = list(top_predictions.keys())
    confidences = list(top_predictions.values())
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(classes)))
    
    bars = ax.barh(classes, confidences, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Class', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Confidence Scores', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
               ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig

# ============================================================================
# RETRAINING FUNCTIONS
# ============================================================================

def load_images_from_folder(base_path, max_per_class=1000):
    """Load images from folder structure for retraining."""
    classes = sorted([d for d in os.listdir(base_path) 
                     if os.path.isdir(os.path.join(base_path, d))])
    
    label_to_int = {class_name: i for i, class_name in enumerate(classes)}
    int_to_label = {i: class_name for i, class_name in enumerate(classes)}
    
    X = []
    y = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(base_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if len(image_files) > max_per_class:
            image_files = np.random.choice(image_files, max_per_class, replace=False)
        
        status_text.text(f"Loading class '{class_name}': {len(image_files)} images")
        
        for img_file in image_files:
            try:
                img_path = os.path.join(class_path, img_file)
                img = Image.open(img_path).convert('L')
                img = img.resize((28, 28))
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                X.append(img_array)
                y.append(label_to_int[class_name])
            except Exception as e:
                pass
        
        progress_bar.progress((idx + 1) / len(classes))
    
    progress_bar.empty()
    status_text.empty()
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), int_to_label

def extract_features_batch(images, batch_size=100):
    n_images = len(images)
    n_batches = int(np.ceil(n_images / batch_size))
    hog_features_list = []
    stat_features_list = []
    progress_bar = st.progress(0)
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_images)
        batch = images[start_idx:end_idx]
        
        for img in batch:
            hog_feat = hog(img, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=False, channel_axis=None)
            hog_features_list.append(hog_feat)
            
            mean_val = np.mean(img)
            std_val = np.std(img)
            max_val = np.max(img)
            min_val = np.min(img)
            binary = (img > 0.5).astype(np.uint8)
            vertical_projection = np.sum(binary, axis=0)
            horizontal_projection = np.sum(binary, axis=1)
            width = np.sum(vertical_projection > 0)
            height = np.sum(horizontal_projection > 0)
            aspect_ratio = width / height if height > 0 else 1.0
            stat_feat = np.array([mean_val, std_val, max_val, min_val, aspect_ratio])
            stat_features_list.append(stat_feat)
        
        progress_bar.progress((i + 1) / n_batches)
    
    progress_bar.empty()
    return np.array(hog_features_list), np.array(stat_features_list)

def train_model_custom(X_train_data, y_train_data, model_type, hyperparams):
    """Train a model with the given hyperparameters."""
    if model_type == "Decision Tree":
        model = DecisionTreeClassifier(**hyperparams)
    else:
        if 'class_weight' not in hyperparams:
            hyperparams['class_weight'] = 'balanced'
        model = RandomForestClassifier(**hyperparams)
    
    with st.spinner(f"Training {model_type}..."):
        model.fit(X_train_data, y_train_data)
    return model

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("🔤 EMNIST Character Classifier")
    st.markdown("### Advanced Image Classification with Feature Engineering")
    st.markdown("---")
    
    # Load models and label mapping
    models, label_mapping, pca_model, models_loaded = load_models()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("⚙️ Navigation")
        page = st.radio("Go to", ["🎯 Prediction", "🔄 Model Retraining"])
        
        if models_loaded:
            st.success("✅ Models loaded successfully!")
            if label_mapping:
                st.info(f"📊 {len(label_mapping)} classes available")
        else:
            st.warning("⚠️ No trained models found. Please train models first.")
    
    # ========================================================================
    # PREDICTION PAGE
    # ========================================================================
    if page == "🎯 Prediction":
        st.header("🎯 Make Predictions")
        
        if not models_loaded:
            st.warning("⚠️ Please train models first in the 'Model Retraining' section.")
            return
        
        # Model selection
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Select Model Type", ["Decision Tree", "Random Forest"])
        with col2:
            feature_type = st.selectbox("Select Feature Type", ["Raw Pixels", "Feature Engineering"])
        
        # Get the selected model
        model_key = f"{'de' if 'Decision' in model_type else 'ra'}_{'eng' if 'Engineering' in feature_type else 'raw'}"
        selected_model = models.get(model_key)
        
        if selected_model is None:
            st.warning(f"⚠️ The selected {model_type} with {feature_type} is not available. Please train it first.")
        else:
            st.success(f"✅ Loaded {model_type} with {feature_type}")
        
        # Image upload
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Upload Image")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
            
            if uploaded_file is not None:
                # Load and display original image
                original_image = Image.open(uploaded_file)
                st.image(original_image, caption='Uploaded Image', use_container_width=True)
                
                # Preprocess image
                try:
                    processed_img = preprocess_image(original_image)
                    st.image(processed_img, caption='Processed Image (28x28)', use_container_width=True)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    processed_img = None
        
        with col2:
            if uploaded_file is not None and selected_model is not None and processed_img is not None:
                st.subheader("🎯 Prediction Results")
                
                try:
                    # Prepare features based on selection
                    if feature_type == "Feature Engineering":
                        if pca_model is None:
                            st.error("⚠️ PCA model not found. Please retrain with feature engineering.")
                        else:
                            with st.spinner("Extracting features..."):
                                features, features_dict = extract_features_single_image(processed_img, pca_model)
                    else:
                        # Raw pixels
                        features = prepare_raw_features(processed_img)
                    
                    # Make prediction
                    with st.spinner("Making prediction..."):
                        prediction_label, confidence_dict = predict_with_confidence(
                            selected_model, features, label_mapping
                        )
                    
                    # Display prediction
                    st.markdown(f"### Predicted Class: **{prediction_label}**")
                    
                    top_conf = list(confidence_dict.values())[0]
                    st.metric("Confidence", f"{top_conf:.2%}")
                    
                    # Show confidence scores
                    st.subheader("📊 Confidence Distribution")
                    fig = plot_confidence_scores(confidence_dict)
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Visualization buttons (from old code)
        if uploaded_file is not None and processed_img is not None:
            st.markdown("---")
            st.subheader("🎨 Advanced Visualizations")
            
            if st.button("🎨 Generate Augmentations", use_container_width=True):
                with st.spinner("Generating augmentations..."):
                    augmented_images = apply_augmentations(original_image)
                    fig = visualize_augmentations(augmented_images)
                    st.pyplot(fig)
                    plt.close()
        
            if st.button("📊 Visualize HOG Features", use_container_width=True):
                with st.spinner("Visualizing HOG features..."):
                    fig = visualize_hog_features(processed_img)
                    st.pyplot(fig)
                    plt.close()
        
    # ========================================================================
    # MODEL RETRAINING PAGE
    # ========================================================================
    elif page == "🔄 Model Retraining":
        st.header("🔄 Model Retraining")
        
        # Data loading section
        st.markdown("### 📂 Load Training Data")
        col1, col2 = st.columns(2)
        with col1:
            train_path = st.text_input("Training Data Path", value=r"/path/to/dataset")
        with col2:
            max_images = st.number_input("Max Images per Class", min_value=100, max_value=20000, value=5000, step=100)
        
        # Model selection
        st.markdown("### 🧠 Model Selection")
        train_both = st.checkbox("Train both Decision Tree and Random Forest", value=True)
        
        if not train_both:
            model_choice = st.selectbox("Select Model to Train", ["Decision Tree", "Random Forest"])
        
        # Feature engineering selection
        st.markdown("### 🛠️ Feature Engineering")
        feature_mode = st.radio(
            "Choose feature engineering approach",
            ["Raw Pixels Only", "Feature Engineering (HOG + Stats + PCA)", "Both"]
        )
        
        # Hyperparameters
        st.markdown("### ⚙️ Hyperparameters")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Decision Tree")
            dt_max_depth = st.slider("Max Depth (DT)", 1, 50, 20, key="dt_depth")
            dt_min_split = st.slider("Min Samples Split (DT)", 2, 50, 10, key="dt_split")
            dt_min_leaf = st.slider("Min Samples Leaf (DT)", 1, 20, 2, key="dt_leaf")
            
        with col2:
            st.markdown("#### Random Forest")
            rf_n_estimators = st.slider("Number of Trees (RF)", 10, 500, 100, key="rf_trees")
            rf_max_depth = st.slider("Max Depth (RF)", 1, 50, 20, key="rf_depth")
            rf_max_features = st.selectbox("Max Features (RF)", ["sqrt", "log2", None], index=0, key="rf_features")
        
        # Start training button
        if st.button("🚀 Start Training", type="primary"):
            if not os.path.exists(train_path):
                st.error("❌ Training path not found!")
                st.stop()
            
            # Load and prepare data
            with st.spinner("Loading data..."):
                X_train, y_train, new_label_mapping = load_images_from_folder(train_path, max_images)
                
                if X_train is None or len(X_train) == 0:
                    st.error("❌ No training data found!")
                    st.stop()
                
                # Save label mapping
                joblib.dump(new_label_mapping, 'label_mapping.joblib')
                st.success(f"✅ Loaded {len(X_train)} images from {len(new_label_mapping)} classes")
                
                # Display class distribution
                unique, counts = np.unique(y_train, return_counts=True)
                st.info(f"📊 Class distribution: min={counts.min()}, max={counts.max()}, mean={counts.mean():.0f}")
                
                # Split into train and test
                X_train_split, X_test, y_train_split, y_test = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
                
                # Prepare features based on mode
                if feature_mode in ["Raw Pixels Only", "Both"]:
                    st.info("Preparing raw pixel features...")
                    X_train_raw = prepare_raw_features(X_train_split)
                    X_test_raw = prepare_raw_features(X_test)
                    
                if feature_mode in ["Feature Engineering (HOG + Stats + PCA)", "Both"]:
                    st.info("Extracting HOG and statistical features...")
                    X_train_flat = X_train_split.reshape(X_train_split.shape[0], -1)
                    X_test_flat = X_test.reshape(X_test.shape[0], -1)
                    
                    # Extract HOG and Stats features
                    X_train_hog, X_train_stats = extract_features_batch(X_train_split)
                    X_test_hog, X_test_stats = extract_features_batch(X_test)
                    
                    # Fit PCA on training data
                    st.info("Fitting PCA on training data...")
                    pca = IncrementalPCA(n_components=50)
                    X_train_pca = pca.fit_transform(X_train_flat)
                    X_test_pca = pca.transform(X_test_flat)
                    
                    # Combine features
                    X_train_eng = np.hstack([X_train_hog, X_train_stats, X_train_pca])
                    X_test_eng = np.hstack([X_test_hog, X_test_stats, X_test_pca])
                    
                    # Save PCA model
                    joblib.dump(pca, 'pca_model.joblib')
                    st.success("✅ PCA model saved")
                
                # Train models
                models_to_train = ["Decision Tree", "Random Forest"] if train_both else [model_choice]
                results = []
                
                for model_type in models_to_train:
                    # Set hyperparameters
                    if model_type == "Decision Tree":
                        hyperparams = {
                            'max_depth': dt_max_depth,
                            'min_samples_split': dt_min_split,
                            'min_samples_leaf': dt_min_leaf,
                            'random_state': 42
                        }
                    else:  # Random Forest
                        hyperparams = {
                            'n_estimators': rf_n_estimators,
                            'max_depth': rf_max_depth,
                            'max_features': rf_max_features,
                            'class_weight': 'balanced',
                            'random_state': 42,
                            'n_jobs': -1
                        }
                    
                    # Train on raw features if selected
                    if feature_mode in ["Raw Pixels Only", "Both"]:
                        st.markdown(f"### Training {model_type} on Raw Pixels")
                        
                        # Train model
                        model_raw = train_model_custom(
                            X_train_raw, y_train_split, model_type, hyperparams
                        )
                        
                        # Evaluate
                        y_pred_train = model_raw.predict(X_train_raw)
                        y_pred_test = model_raw.predict(X_test_raw)
                        
                        train_acc = accuracy_score(y_train_split, y_pred_train)
                        test_acc = accuracy_score(y_test, y_pred_test)
                        
                        st.success(f"Train Accuracy: {train_acc:.2%} | Test Accuracy: {test_acc:.2%}")
                        
                        # Confusion matrix
                        cm = confusion_matrix(y_test, y_pred_test)
                        fig, ax = plt.subplots(figsize=(12, 10))
                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt='d',
                            cmap='Blues',
                            ax=ax,
                            xticklabels=[new_label_mapping[i] for i in sorted(new_label_mapping)],
                            yticklabels=[new_label_mapping[i] for i in sorted(new_label_mapping)]
                        )
                        ax.set_title(f'{model_type} - Raw Pixels\nTest Accuracy: {test_acc:.2%}')
                        ax.set_xlabel('Predicted Label')
                        ax.set_ylabel('True Label')
                        plt.xticks(rotation=45, ha='right')
                        plt.yticks(rotation=0)
                        st.pyplot(fig)
                        plt.close()
                        
                        # Save model
                        model_filename = f"{'de' if 'Decision' in model_type else 'ra'}_raw.joblib"
                        joblib.dump(model_raw, model_filename)
                        st.success(f"✅ Saved to {model_filename}")
                        
                        results.append({
                            'Model': model_type,
                            'Features': 'Raw Pixels',
                            'Train Accuracy': train_acc,
                            'Test Accuracy': test_acc,
                            'File': model_filename
                        })
                    
                    # Train on engineered features if selected
                    if feature_mode in ["Feature Engineering (HOG + Stats + PCA)", "Both"]:
                        st.markdown(f"### Training {model_type} on Engineered Features")
                        
                        # Train model
                        model_eng = train_model_custom(
                            X_train_eng, y_train_split, model_type, hyperparams
                        )
                        
                        # Evaluate
                        y_pred_train = model_eng.predict(X_train_eng)
                        y_pred_test = model_eng.predict(X_test_eng)
                        
                        train_acc = accuracy_score(y_train_split, y_pred_train)
                        test_acc = accuracy_score(y_test, y_pred_test)
                        
                        st.success(f"Train Accuracy: {train_acc:.2%} | Test Accuracy: {test_acc:.2%}")
                        
                        # Confusion matrix
                        cm = confusion_matrix(y_test, y_pred_test)
                        fig, ax = plt.subplots(figsize=(12, 10))
                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt='d',
                            cmap='Blues',
                            ax=ax,
                            xticklabels=[new_label_mapping[i] for i in sorted(new_label_mapping)],
                            yticklabels=[new_label_mapping[i] for i in sorted(new_label_mapping)]
                        )
                        ax.set_title(f'{model_type} - Feature Engineering\nTest Accuracy: {test_acc:.2%}')
                        ax.set_xlabel('Predicted Label')
                        ax.set_ylabel('True Label')
                        plt.xticks(rotation=45, ha='right')
                        plt.yticks(rotation=0)
                        st.pyplot(fig)
                        plt.close()
                        
                        # Save model
                        model_filename = f"{'de' if 'Decision' in model_type else 'ra'}_eng.joblib"
                        joblib.dump(model_eng, model_filename)
                        st.success(f"✅ Saved to {model_filename}")
                        
                        results.append({
                            'Model': model_type,
                            'Features': 'Feature Engineering',
                            'Train Accuracy': train_acc,
                            'Test Accuracy': test_acc,
                            'File': model_filename
                        })
                
                # Display results summary
                st.markdown("---")
                st.markdown("## 📊 Training Results Summary")
                
                if results:
                    df_results = pd.DataFrame(results)
                    
                    # Format and display table
                    st.dataframe(
                        df_results.style.format({
                            'Train Accuracy': '{:.2%}',
                            'Test Accuracy': '{:.2%}'
                        }),
                        use_container_width=True
                    )
                    
                    # Find best model
                    best_model = max(results, key=lambda x: x['Test Accuracy'])
                    st.success(
                        f"🎯 Best Model: **{best_model['Model']}** with **{best_model['Features']}** "
                        f"(Test Accuracy: **{best_model['Test Accuracy']:.2%}**)"
                    )
                    
                    # Visualize comparison if multiple models
                    if len(results) > 1:
                        st.markdown("### 📈 Model Comparison")
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                        
                        # Accuracy comparison
                        df_melted = df_results.melt(
                            id_vars=['Model', 'Features'],
                            value_vars=['Train Accuracy', 'Test Accuracy'],
                            var_name='Metric',
                            value_name='Accuracy'
                        )
                        
                        sns.barplot(
                            data=df_melted,
                            x='Model',
                            y='Accuracy',
                            hue='Metric',
                            ax=ax1
                        )
                        ax1.set_title('Train vs Test Accuracy')
                        ax1.set_ylim(0, 1.1)
                        ax1.legend(title='Metric')
                        
                        # Add value labels
                        for p in ax1.patches:
                            ax1.annotate(
                                f"{p.get_height():.2%}",
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center',
                                xytext=(0, 10),
                                textcoords='offset points',
                                fontsize=9, fontweight='bold'
                            )
                        
                        # Feature comparison
                        sns.barplot(
                            data=df_results,
                            x='Features',
                            y='Test Accuracy',
                            hue='Model',
                            ax=ax2
                        )
                        ax2.set_title('Test Accuracy by Features')
                        ax2.set_ylim(0, 1.1)
                        ax2.legend(title='Model')
                        
                        # Add value labels
                        for p in ax2.patches:
                            ax2.annotate(
                                f"{p.get_height():.2%}",
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center',
                                xytext=(0, 10),
                                textcoords='offset points',
                                fontsize=9, fontweight='bold'
                            )
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    
                    st.balloons()
                    
                    # Classification report expander
                    with st.expander("📋 Detailed Classification Report (Best Model)"):
                        best_result = max(results, key=lambda x: x['Test Accuracy'])
                        
                        # Load the best model to get predictions
                        best_model_obj = joblib.load(best_result['File'])
                        
                        # Get test features based on type
                        if 'Raw' in best_result['Features']:
                            X_test_eval = X_test_raw
                        else:
                            X_test_eval = X_test_eng
                        
                        y_pred_best = best_model_obj.predict(X_test_eval)
                        
                        target_names = [new_label_mapping[i] for i in sorted(new_label_mapping.keys())]
                        report = classification_report(
                            y_test,
                            y_pred_best,
                            target_names=target_names,
                            zero_division=0
                        )
                        st.text(report)
                
                gc.collect()

if __name__ == "__main__":
    main()