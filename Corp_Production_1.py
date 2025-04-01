import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from streamlit_folium import st_folium
import folium

# Load data
page_by_img_1 = """
    <style>
    [data-testid="stAppViewContainer"]{
    background: rgb(95, 226, 91);
    background: linear-gradient(159deg, rgb(151, 233, 187) 0%, rgb(98, 235, 80) 100%);
    [data-testid="stSidebar"]{
    background-color:rgb(81, 238, 42);
    background-image: linear-gradient(315deg,rgb(29, 110, 9) 0%,rgb(180, 201, 171) 74%);;
    }
    }
    </style>
    """
st.markdown(page_by_img_1, unsafe_allow_html=True)  
df = pd.read_excel('C:/Users/Admin/OneDrive/Desktop/Final/DS_Crop_Production_Prediction/FAOSTAT_data.xlsx')   # Replace with your actual dataset
st.title('Crop Yield Prediction')
st.write('Analyze trends and predict crop production')


# Display data preview
st.subheader('Data Preview')
st.write(df.head())


df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df.dropna(inplace=True)

###Analyze Trends:        
trends = df.groupby(['Area', 'Element', 'Year'])['Value'].sum().reset_index()
st.subheader('Analyze trends')
region = st.selectbox("Select Region", trends['Area'].unique())
filtered_data = trends[trends['Area'] == region]
plt.plot(filtered_data['Year'], filtered_data['Value'])
st.pyplot(plt)

###Predictive Modeling:
#Preprocessing the Data
le = LabelEncoder()
df['Element'] = le.fit_transform(df['Element'])
df['Item'] = le.fit_transform(df['Item'])
df.dropna(inplace=True)
#Splitting the Data
X = df[['Element', 'Item', 'Year']]
y = df['Value']  # Replace 'Value' with the column you'd like to predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Building a Model
model = RandomForestRegressor()
model.fit(X_train, y_train)
#Integrating with Streamlit
st.subheader('Predictive Modeling')
feature_values = [st.number_input(f"Enter value for {col}") for col in X.columns]
if st.button("Predict"):
    prediction = model.predict([feature_values])
    st.write(f"Predicted Production: {prediction[0]}")
    
#Evaluating the Model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
st.write(f"Model Evaluation:\nMean Squared Error: {mse}\nR2 Score: {r2}")


###Actionable Insights
#analysis to uncover trends and relationships between variables:
sns.pairplot(df, hue='Item')
plt.show()
#Modeling for Insights
#model1 = RandomForestClassifier()
model1 = RandomForestRegressor()
model1.fit(X_train, y_train)
y_train_binned = np.digitize(y_train, bins=[10, 20, 30])  # Adjust bins as per your data

def get_crop_recommendations(Area):
 return ["Rice", "Wheat", "Maize"] if Area else ["No recommendations"]
st.subheader("Agricultural Insights")

selected_region = st.selectbox("Select Region", df['Area'].unique())
if selected_region:
 crop_recommendations = get_crop_recommendations(selected_region)  # Define this function
 st.write(f"Recommended Crops for, {selected_region} : {crop_recommendations}")


# Define the optimize_resources function
def optimize_resources(Area):
    # Add your optimization logic here
    return {"Water": "Optimized", "Fertilizer": "Balanced"}  # Example allocation

# Streamlit button to optimize resources
if st.button("Optimize Resources"):
    #selected_region = "Area"  # Replace this with actual input for the region
    allocation = optimize_resources(selected_region)
    st.write(f"Optimized Resource Allocation:{selected_region}: {allocation}")

# Create a Folium map
latitude, longitude = 10.0, 78.0  # Replace with actual coordinates
m = folium.Map(location=[latitude, longitude], zoom_start=12)


# Display the map in Streamlit
st_folium(m)