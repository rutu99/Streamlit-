import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# ------------- Data prep and model training -------------

def convert_range(x: str):
    parts = str(x).split('-')
    if len(parts) == 2:
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except ValueError:
            return None
    try:
        return float(x)
    except ValueError:
        return None

def remove_outliers_sqft(df: pd.DataFrame) -> pd.DataFrame:
    df_output = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        gen_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_output = pd.concat([df_output, gen_df], ignore_index=True)
    return df_output

def bhk_outlier_remover(df: pd.DataFrame) -> pd.DataFrame:
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(
                    exclude_indices,
                    bhk_df[bhk_df.price_per_sqft < stats['mean']].index.values
                )
    return df.drop(exclude_indices, axis='index')

@st.cache_resource
def load_model():
    # Load original dataset (same as in notebook)
    data = pd.read_csv("Bengaluru_House_Data.csv")

    # Drop unnecessary columns
    data.drop(columns=['area_type', 'availability', 'society', 'balcony'], inplace=True)

    # Handle missing values
    data['location'] = data['location'].fillna('Sarjapur  Road')
    data['size'] = data['size'].fillna('2 BHK')
    data['bath'] = data['bath'].fillna(data['bath'].median())

    # Feature engineering: bhk
    data['bhk'] = data['size'].str.split().str.get(0).astype(int)

    # Convert total_sqft ranges to numbers
    data['total_sqft'] = data['total_sqft'].apply(convert_range)

    # Drop rows where total_sqft could not be converted
    data = data.dropna(subset=['total_sqft'])

    # Price per square feet
    data['price_per_sqft'] = data['price'] * 100000 / data['total_sqft']

    # Clean locations: strip spaces and group rare locations into "other"
    data['location'] = data['location'].apply(lambda x: x.strip())
    location_count = data['location'].value_counts()
    location_count_less_10 = location_count[location_count <= 10]
    data['location'] = data['location'].apply(
        lambda x: 'other' if x in location_count_less_10 else x
    )

    # Remove outliers (same logic as notebook)
    data = remove_outliers_sqft(data)
    data = bhk_outlier_remover(data)

    # Drop unused columns
    data.drop(columns=['size', 'price_per_sqft'], inplace=True)

    # One-hot encoding for location
    dummies = pd.get_dummies(data.location)
    data = pd.concat([data, dummies.drop('other', axis='columns')], axis='columns')
    data = data.drop('location', axis='columns')

    # Features and target
    X = data.drop(columns=['price'])
    y = data['price']

    # Train Linear Regression (same model as notebook)
    lr = LinearRegression()
    lr.fit(X, y)

    # Locations to show in UI (exclude "other")
    # We recompute them from the original cleaned location column
    all_locations = sorted([loc for loc in location_count.index if loc not in location_count_less_10.index])

    return lr, X.columns, all_locations

def price_predict(model, columns, location, sqft, bath, bhk):
    # Build input vector in the same order as X.columns
    x = np.zeros(len(columns))
    # Assuming first three columns are: total_sqft, bath, bhk
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    # Set the location dummy if it exists
    if location in columns:
        loc_index = list(columns).index(location)
        x[loc_index] = 1

    return model.predict([x])[0]

# ------------- Streamlit UI -------------

st.set_page_config(page_title="Bangalore House Price Prediction", layout="centered")

st.title("🏠 Bangalore House Price Prediction")
st.write("This app uses a Linear Regression model trained on Bengaluru house data to predict prices.")

with st.spinner("Loading model and data..."):
    model, feature_columns, locations = load_model()

st.subheader("Enter property details")

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Location", locations)

    bhk = st.number_input(
        "BHK (Number of Bedrooms)",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

with col2:
    sqft = st.number_input(
        "Total Area (Square Feet)",
        min_value=300.0,
        max_value=10000.0,
        value=1000.0,
        step=10.0
    )

    bath = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

if st.button("Predict Price"):
    price_lakhs = price_predict(model, feature_columns, location, sqft, bath, bhk)
    price_rupees = price_lakhs * 100000

    st.success(f"Estimated Price: **₹{price_rupees:,.0f}**  (≈ **{price_lakhs:.2f} Lakhs**)")

    st.caption("Note: Predictions are approximate and based on historical data.")