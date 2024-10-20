import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset (Assuming you have it in the same directory or in Google Drive)
laptop_data = pd.read_csv('laptops.csv')

# Create Price Range column
def categorize_price(price):
    if price < 60000:
        return 'Low'
    elif 60000 <= price < 100000:
        return 'Medium'
    else:
        return 'High'

laptop_data['Price Range'] = laptop_data['Price'].apply(categorize_price)

# Prepare the user-item matrix for collaborative filtering
def prepare_collaborative_data(laptop_data):
    # Remove rows with NaN values in 'Name' or 'User Rating'
    ratings_data = laptop_data.dropna(subset=['Name', 'User Rating'])
    
    # Group by 'Name' and calculate the average 'User Rating'
    ratings_data = ratings_data.groupby('Name')['User Rating'].mean().reset_index()
    
    # Create a user-item matrix
    user_item_matrix = ratings_data.set_index('Name')
    
    return user_item_matrix

# Calculate cosine similarity
def calculate_cosine_similarity(user_item_matrix):
    # Ensure there are no NaN values
    user_item_matrix = user_item_matrix.fillna(0)
    
    cosine_sim = cosine_similarity(user_item_matrix)
    return pd.DataFrame(
        cosine_sim,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

# Get collaborative filtering recommendations
def get_collaborative_recommendations(laptop_name, cosine_sim_df, laptop_data, num_recommendations=5):
    if laptop_name not in cosine_sim_df.index:
        return pd.DataFrame()  # Return empty DataFrame if laptop not found
    
    similar_scores = cosine_sim_df[laptop_name]
    similar_laptops = similar_scores.sort_values(ascending=False)
    
    # Remove the input laptop from recommendations
    similar_laptops = similar_laptops.drop(laptop_name, errors='ignore')
    
    recommended_laptops = similar_laptops.head(num_recommendations).index.tolist()
    
    # Get full details of recommended laptops
    recommendations = laptop_data[laptop_data['Name'].isin(recommended_laptops)].drop_duplicates(subset=['Name'])
    
    return recommendations[['Name', 'Brand', 'Price', 'Link']]

# Streamlit app setup
st.title("Laptop Recommendation System")
st.header("Find the Best Laptop for You!")

# User inputs
processor = st.text_input("Search for Processor")
graphic_processor = st.text_input("Search for Graphic Processor")

# Modified dropdown inputs
ssd_options = ["Select"] + list(laptop_data['SSD Capacity'].unique())
ram_options = ["Select"] + list(laptop_data['RAM'].unique())
brand_options = ["Select"] + list(laptop_data['Brand'].unique())

ssd_capacity = st.selectbox("Select SSD Capacity", options=ssd_options)
ram = st.selectbox("Select RAM", options=ram_options)
color = st.text_input("Search for Color")
brand = st.selectbox("Select Brand", options=brand_options)
price_range = st.slider("Select Price Range", 0, 300000, (0, 300000))

# Filter laptops based on user inputs
filtered_laptops = laptop_data[
    (laptop_data['Price'] >= price_range[0]) &
    (laptop_data['Price'] <= price_range[1])
]

if processor:
    filtered_laptops = filtered_laptops[filtered_laptops['Processor'].str.contains(processor, case=False, na=False)]
if graphic_processor:
    filtered_laptops = filtered_laptops[filtered_laptops['Graphic Processor'].str.contains(graphic_processor, case=False, na=False)]
if color:
    filtered_laptops = filtered_laptops[filtered_laptops['Color'].str.contains(color, case=False, na=False)]
if ssd_capacity != "Select":
    filtered_laptops = filtered_laptops[filtered_laptops['SSD Capacity'] == ssd_capacity]
if ram != "Select":
    filtered_laptops = filtered_laptops[filtered_laptops['RAM'] == ram]
if brand != "Select":
    filtered_laptops = filtered_laptops[filtered_laptops['Brand'] == brand]

# Display recommendations
st.subheader("Recommended Laptops:")
if not filtered_laptops.empty:
    for i, row in filtered_laptops.iterrows():
        st.write(f"**{row['Name']}** - {row['Brand']} - Price: ${row['Price']}")
        st.write(f"[More Info]({row['Link']})")
else:
    st.write("No laptops found with the selected criteria.")

# Collaborative Filtering Section
st.subheader("Collaborative Filtering Recommendations")

# Prepare data for collaborative filtering
user_item_matrix = prepare_collaborative_data(laptop_data)
cosine_sim_df = calculate_cosine_similarity(user_item_matrix)

if st.button("Get Recommendations", key="collab_recommend_button"):
    if not filtered_laptops.empty:
        # Assume the first laptop in the filtered list for recommendations
        laptop_name = filtered_laptops.iloc[0]['Name']
        st.write(f"Selected Laptop for Recommendations: {laptop_name}")

        # Get recommendations
        recommended_laptops = get_collaborative_recommendations(laptop_name, cosine_sim_df, laptop_data)
        st.write("Recommended Laptops Based on Ratings:")
        
        if not recommended_laptops.empty:
            for _, rec_row in recommended_laptops.iterrows():
                st.write(f"**{rec_row['Name']}** - {rec_row['Brand']} - Price: ${rec_row['Price']}")
                st.write(f"[More Info]({rec_row['Link']})")
        else:
            st.write("No recommendations available for this laptop.")
    else:
        st.write("No laptops found with the selected criteria.")