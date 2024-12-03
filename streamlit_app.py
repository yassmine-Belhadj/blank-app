
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the data
@st.cache_data
def load_data():
    file_path = 'data.csv'
    data = pd.read_csv(file_path)
    return data

# Preprocess the data
def preprocess_data(df):
    # Drop unnecessary columns ('id' and 'Unnamed: 32')
    df = df.drop(columns=['id', 'Unnamed: 32'])
    
    # Map 'diagnosis' column: 'M' -> 1 (Malignant), 'B' -> 0 (Benign)
    df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Separate features and target
    X = df.drop(columns=['diagnosis', 'target'])  # Drop 'diagnosis' and 'target' columns
    y = df['target']  # 'target' is the label
    
    # Handle missing values (numeric columns only)
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy="median")

    # Impute numeric columns (replace NaNs with the median of each column)
    X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

    # Ensure the shape is correct before proceeding
    st.write("Final X shape:", X.shape)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)
    # Map 'diagnosis' column: 'M' -> 1 (Malignant), 'B' -> 0 (Benign)
    df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Separate features and target
    X = df.drop(columns=['diagnosis', 'target'])
    y = df['target']

    # Check the initial shape and column names
    st.write("Initial X shape:", X.shape)
    st.write("Initial columns in X:", X.columns)

    # Handle missing values (numeric columns only)
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy="median")

    # Impute numeric columns
    X_numeric_imputed = pd.DataFrame(
        imputer.fit_transform(X[numeric_cols]),  # Impute only numeric columns
        columns=numeric_cols,                   # Use the same column names
        index=X.index                           # Preserve the original index
    )

    # Check shape after imputation
    st.write("Shape after imputation:", X_numeric_imputed.shape)

    # Replace the numeric columns in X with the imputed values
    X[numeric_cols] = X_numeric_imputed

    # Check the final shape and columns of X
    st.write("Final X shape after update:", X.shape)
    st.write("Final columns in X:", X.columns)

    return train_test_split(X, y, test_size=0.2, random_state=42)
    # Map 'diagnosis' column: 'M' -> 1 (Malignant), 'B' -> 0 (Benign)
    df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Separate features and target
    X = df.drop(columns=['diagnosis', 'target'])
    y = df['target']

    # Handle missing values (numeric columns only)
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy="median")

    # Impute numeric columns
    X_numeric_imputed = pd.DataFrame(
        imputer.fit_transform(X[numeric_cols]),  # Impute only numeric columns
        columns=numeric_cols,                   # Use the same column names
        index=X.index                           # Preserve the original index
    )
    
    # Replace the numeric columns in X with the imputed values
    X[numeric_cols] = X_numeric_imputed

    return train_test_split(X, y, test_size=0.2, random_state=42)
    # Map 'diagnosis' column: 'M' -> 1 (Malignant), 'B' -> 0 (Benign)
    df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Separate features and target
    X = df.drop(columns=['diagnosis', 'target'])
    y = df['target']

    # Handle missing values (numeric columns only)
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy="median")

    # Impute numeric columns
    X_numeric_imputed = pd.DataFrame(
        imputer.fit_transform(X[numeric_cols]),
        columns=numeric_cols,
        index=X.index
    )
    
    # Replace numeric columns in X with the imputed values
    X.update(X_numeric_imputed)

    return train_test_split(X, y, test_size=0.2, random_state=42)
    # Map 'diagnosis' column: 'M' -> 1 (Malignant), 'B' -> 0 (Benign)
    df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Separate features and target
    X = df.drop(columns=['diagnosis', 'target'])
    y = df['target']

    # Handle missing values (numeric columns only)
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy="median")
    
    # Transform numeric columns and reassign back to DataFrame
    X[numeric_cols] = pd.DataFrame(imputer.fit_transform(X[numeric_cols]), columns=numeric_cols, index=X.index)

    return train_test_split(X, y, test_size=0.2, random_state=42)
    # Map 'diagnosis' column: 'M' -> 1 (Malignant), 'B' -> 0 (Benign)
    df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Separate features and target
    X = df.drop(columns=['diagnosis', 'target'])
    y = df['target']

    # Handle missing values (numeric columns only)
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy="median")
    X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
@st.cache_data
def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = SVC(probability=True)
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Predict malignancy
def predict(model, scaler, input_data):
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    return prediction[0], probability[0]

# Main app
st.title("🎈 Tumor Malignancy Predictor")

st.write(
    "This app predicts whether a tumor is **malignant** or **benign** based on input features."
)

# Load and preprocess the data
data = load_data()
st.write("Data Preview:")
st.dataframe(data.head())

X_train, X_test, y_train, y_test = preprocess_data(data)
model, scaler = train_model(X_train, y_train)

# Display model accuracy
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: **{accuracy:.2%}**")

# Input form for prediction
st.header("Input Tumor Features")
input_data = []
columns = X_train.columns
for col in columns:
    value = st.number_input(f"Enter {col}:", key=col)
    input_data.append(value)

if st.button("Predict"):
    prediction, probability = predict(model, scaler, input_data)
    if prediction == 1:
        st.write("### The tumor is **Malignant**.")
    else:
        st.write("### The tumor is **Benign**.")
    st.write(f"Confidence: Malignant: {probability[1]:.2%}, Benign: {probability[0]:.2%}")

