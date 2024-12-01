import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
import json
from typing import List, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Constants
API_BASE_URL = "http://0.0.0.0:6027"
TIMEFRAMES = ["1T", "5T", "15T", "30T", "1H", "4H", "1D", "1W", "1M"]

# Helper Functions
def format_api_response(response: requests.Response) -> dict:
    """Format API response and handle errors"""
    try:
        return response.json()
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")
        return None

def get_categories() -> List[str]:
    """Fetch all categories from the API"""
    response = requests.get(f"{API_BASE_URL}/categories")
    if response.status_code == 200:
        data = response.json()
        return [cat["category"] for cat in data["categories"]]
    return []

def get_symbols(category: str) -> List[str]:
    """Fetch symbols for a specific category"""
    response = requests.get(f"{API_BASE_URL}/categories/{category}/symbols")
    if response.status_code == 200:
        return response.json()["symbols"]
    return []

def plot_candlestick(data: Dict[str, List], title: str):
    """Create a candlestick chart with volume"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, subplot_titles=(title, 'Volume'),
                       row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=[datetime.fromisoformat(d) for d in data['timestamp']],
                                open=data['open'],
                                high=data['high'],
                                low=data['low'],
                                close=data['close'],
                                name='OHLC'),
                 row=1, col=1)

    # Volume chart
    fig.add_trace(go.Bar(x=[datetime.fromisoformat(d) for d in data['timestamp']],
                        y=data['volume'],
                        name='Volume'),
                 row=2, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", 
    ["Category Management", "Data Upload", "Data Retrieval"]
)

# Category Management Page
if page == "Category Management":
    st.title("Category Management")
    
    # Create Category
    with st.expander("Create New Category"):
        new_category = st.text_input("Category Name")
        if st.button("Create Category"):
            response = requests.post(
                f"{API_BASE_URL}/categories",
                json={"category": new_category}
            )
            if response.status_code == 200:
                st.success(f"Category '{new_category}' created successfully!")
            else:
                st.error(f"Error creating category: {response.json()['detail']}")

    # Manage Existing Categories
    st.subheader("Manage Existing Categories")
    categories = get_categories()
    
    if categories:
        selected_category = st.selectbox("Select Category", categories)
        
        # Show category statistics
        if st.button("Show Category Stats"):
            response = requests.get(f"{API_BASE_URL}/category_stats/{selected_category}")
            if response.status_code == 200:
                stats = response.json()
                st.write(f"Total Symbols: {stats['total_symbols']}")
                st.write("Symbol Statistics:")
                stats_df = pd.DataFrame(stats['symbols'])
                st.dataframe(stats_df)

        # Add/Remove Symbols
        with st.expander("Add Symbol"):
            new_symbol = st.text_input("Symbol Name")
            if st.button("Add Symbol"):
                response = requests.post(
                    f"{API_BASE_URL}/categories/{selected_category}/symbols",
                    json={"symbol": new_symbol}
                )
                if response.status_code == 200:
                    st.success(f"Symbol '{new_symbol}' added successfully!")
                else:
                    st.error(f"Error adding symbol: {response.json()['detail']}")

        # Delete Category
        if st.button("Delete Category", key="delete_category"):
            if st.checkbox("Confirm deletion"):
                response = requests.delete(f"{API_BASE_URL}/categories/{selected_category}")
                if response.status_code == 200:
                    st.success(f"Category '{selected_category}' deleted successfully!")
                else:
                    st.error(f"Error deleting category: {response.json()['detail']}")

# Data Upload Page
elif page == "Data Upload":
    st.title("Data Upload")
    
    # Category and Symbol Selection
    categories = get_categories()
    category = st.selectbox("Select Category", categories)
    
    if category:
        symbols = get_symbols(category)
        symbol = st.selectbox("Select Symbol", symbols if symbols else [""])
        
        if symbol:
            # File Upload
            uploaded_files = st.file_uploader(
                "Upload Data Files (CSV/TXT)", 
                accept_multiple_files=True
            )
            
            if uploaded_files:
                chunk_size = st.number_input(
                    "Chunk Size", 
                    min_value=1000,
                    max_value=100000,
                    value=50000
                )
                
                max_workers = st.number_input(
                    "Max Workers",
                    min_value=1,
                    max_value=8,
                    value=4
                )
                
                if st.button("Upload Data"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    files = [
                        ("files", file)
                        for file in uploaded_files
                    ]
                    
                    response = requests.post(
                        f"{API_BASE_URL}/upload/{category}/{symbol}",
                        params={
                            "chunk_size": chunk_size,
                            "max_workers": max_workers
                        },
                        files=files
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(result["message"])
                        st.write("Upload Details:", result["details"])
                    else:
                        st.error(f"Error uploading data: {response.json()['detail']}")

# Data Retrieval Page
else:
    st.title("Data Retrieval")
    
    # Category and Symbol Selection
    categories = get_categories()
    category = st.selectbox("Select Category", categories)
    
    if category:
        symbols = get_symbols(category)
        symbol = st.selectbox("Select Symbol", symbols if symbols else [""])
        
        if symbol:
            # Date Range Selection
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    datetime.now() - timedelta(days=30)
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    datetime.now()
                )
            
            # Timeframe Selection
            timeframe = st.selectbox("Select Timeframe", TIMEFRAMES)
            
            if st.button("Retrieve Data"):
                response = requests.get(
                    f"{API_BASE_URL}/historical_data/{category}/{symbol}",
                    params={
                        "start_date": datetime.combine(start_date, datetime.min.time()).isoformat(),
                        "end_date": datetime.combine(end_date, datetime.max.time()).isoformat(),
                        "timeframe": timeframe
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()["data"]
                    
                    # Create DataFrame for display
                    df = pd.DataFrame({
                        'timestamp': [datetime.fromisoformat(ts) for ts in data['timestamp']],
                        'open': data['open'],
                        'high': data['high'],
                        'low': data['low'],
                        'close': data['close'],
                        'volume': data['volume']
                    })
                    
                    # Display candlestick chart
                    plot_candlestick(
                        data,
                        f"{symbol} ({timeframe}) - {start_date} to {end_date}"
                    )
                    
                    # Display data table
                    st.write("Raw Data:")
                    st.dataframe(df)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"{symbol}_{timeframe}_{start_date}_{end_date}.csv",
                        "text/csv",
                        key='download-csv'
                    )
                else:
                    st.error(f"Error retrieving data: {response.json()['detail']}")

# Display system info in sidebar
if st.sidebar.checkbox("Show System Info"):
    response = requests.get(f"{API_BASE_URL}/system_info")
    if response.status_code == 200:
        system_info = response.json()
        st.sidebar.write("System Information:")
        st.sidebar.json(system_info)