import streamlit as st
import json
import pandas as pd
import requests
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from concurrent.futures import ThreadPoolExecutor, as_completed
import altair as alt
import re
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore
from streamlit_tags import st_tags
import boto3
from io import StringIO


#api key below
api_token = st.secrets['API_KEY']
key_dict = json.loads(st.secrets['textkey'])
# Initialize Firestore if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(key_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()


real_sectors = ['XLK', 'XLC', 'XLV', 'XLF', 'XLP', 'XLI', 'XLE', 'XLY', 'XLB', 'XME', 'XLU', 'XLRE']
real_subsectors = ['GDX', 'UFO', 'KBE', 'AMLP', 'ITA', 'ITB', 'IAK', 'SMH', 'PINK', 'XBI', 'NLR', 'FXI', 'WGMI', 'JETS', 'PEJ', 'REMX']

sectors = ['XLK', 'XLC', 'XLV', 'XLF', 'XLP', 'XLI', 'XLE', 'XLY', 'XLB', 'XLU', 'XLRE', 'MAGS', 'SPY']
subsectors = ['GDX', 'UFO', 'KBE', 'KRE', 'AMLP', 'ITA', 'ITB', 'IAK', 'SMH','XME', 'PINK', 'XBI', 'NLR', 'BOAT', 'WGMI', 'JETS', 'PEJ', 'QTUM', 'HACK', 'SHLD', 'REMX']
ratecut_etfs = ['AAAU', 'COPX', 'CPER', 'URA', 'CANE', 'XOP', 'UNG', 'WOOD', 'LIT', 'PPLT', 'PALL', 'SLX', 'BNO', 'IBIT', 'SILJ', 'URNJ', 'SLV', 'ETHA', 'USCI', 'LITP']
macro_etfs = ['EWA', 'INDA', 'IDX', 'EWM', 'THD', 'EIS', 'FXI', 'ENZL', 'EZA', 'EWY','EWU', 'ARGT', 'EWJ', 'EWC', 'EWW', 'UAE', 'EWS', 'COLO', 'EWG', 'EPOL', 'EWD', 'VGK', 'EWO', 'EWP', 'QAT', 'EWK', 'EWT', 'GREK', 'EWH', 'EWN', 'ECH', 'EPU']

# Function to remove rows with any null values
def remove_nulls(df):
    return df.dropna()

def get_previous_business_day(date):
    """
    Adjust the date to the nearest previous business day if it falls on a non-business day (weekend or holiday).
    """
    holidays = USFederalHolidayCalendar().holidays()
    bday_range = pd.bdate_range(date, date, freq='C', holidays=holidays)
    
    while not len(bday_range):
        date -= BDay(1)
        bday_range = pd.bdate_range(date, date, freq='C', holidays=holidays)
    
    return date


# Function to load CSV from S3 and convert to DataFrame
# OPTIMIZED: Added caching with 1 hour TTL
@st.cache_data(ttl=3600)
def load_csv_from_s3(bucket, file_key):
    """
    Load CSV from S3 bucket and convert it to a pandas DataFrame
    Cached for 1 hour to reduce S3 API calls
    """
    aws_access_key = st.secrets['aws_access_key']
    aws_secret_key = st.secrets['aws_secret_key']
    region_name = st.secrets['region_name']
    
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region_name
    )
    
    obj = s3.get_object(Bucket=bucket, Key=file_key)
    csv_content = obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_content))
    return df
    
# Function to calculate consecutive appearances and track the first appearance date
def calculate_consecutive_appearances(df):
    # Extract date columns and sort them in descending order
    date_columns = sorted([col for col in df.columns if col != 'Rank'], reverse=True)
    
    # Get all unique symbols, excluding NaN
    symbols = pd.unique(df[date_columns].values.ravel('K'))
    symbols = [symbol for symbol in symbols if pd.notna(symbol)]
    
    # Initialize a dictionary to store counts and first appearance dates
    consecutive_counts = {}
    first_appearance_dates = {}

    # Iterate over each symbol
    for symbol in symbols:
        count = 0
        first_appearance = None
        for date in date_columns:
            # Check if the symbol appears on this date
            if symbol in df[date].values:
                count += 1
                first_appearance = date  # Track the first date it appears
            else:
                break  # Stop counting when the symbol does not appear on a date
        
        consecutive_counts[symbol] = count
        first_appearance_dates[symbol] = first_appearance if count > 0 else "N/A"

    # Convert the dictionary to a DataFrame
    consecutive_df = pd.DataFrame(list(consecutive_counts.items()), columns=['Symbol', 'consecutive_appearance'])
    consecutive_df['first_appearance_date'] = consecutive_df['Symbol'].map(first_appearance_dates)
    
    return consecutive_df

# Function to calculate the return using the adjusted close of one period before the signal date
def calculate_consecutive_returns(symbol, first_appearance_date, df, api_key):
    # Ensure the historical data is sorted by date in ascending order
    df = df.sort_values(by='date')
    
    # Find the index of the first_appearance_date
    signal_index = df.index[df['date'] == first_appearance_date].tolist()
    
    if signal_index:
        signal_index = signal_index[0]  # Get the index of the first appearance date
        
        # Ensure there's a row before the signal date for the price_at_signal
        if signal_index > 0:
            # Get the adjusted close price of one period before the signal date
            price_at_signal = df.iloc[signal_index - 1]['adjusted_close']
            
            # Fetch the real-time price using the API
            current_price = fetch_real_time_price(symbol, api_key)
            
            if current_price:
                # Calculate the return
                consecutive_day_return = ((current_price - price_at_signal) / price_at_signal) * 100

                # Collect the required data
                return {
                    'symbol': symbol,
                    'date_signaled': first_appearance_date,
                    'price_at_signal': price_at_signal,
                    'current_price': current_price,
                    'consecutive_day_return': consecutive_day_return
                }
    
    # If no valid signal date or no previous price available, return None
    return None

# Main process to calculate returns for consecutive appearances
def process_consecutive_returns(sector_df, subsector_df, api_key):
    # Merge and filter the consecutive appearance data
    combined_df = merge_consecutive_dfs(sector_df, subsector_df)
    
    results = []
    
    # Loop over each symbol and fetch the data
    for idx, row in combined_df.iterrows():
        symbol = row['Symbol']
        first_appearance_date = row['first_appearance_date']
        
        if first_appearance_date != "N/A":
            # Fetch historical data for the symbol
            _, historical_data = fetch_data(symbol, api_key)
            
            if not historical_data.empty:
                # Calculate the return based on the adjusted close one period before the signal date
                result = calculate_consecutive_returns(symbol, first_appearance_date, historical_data, api_key)
                if result:
                    results.append(result)
    
    # Convert results to DataFrame
    return pd.DataFrame(results)
    
# Merging the sector and subsector dataframes based on Symbol
def merge_consecutive_dfs(sector_df, subsector_df):
    # Filter rows where consecutive appearances > 0
    sector_df = sector_df[sector_df['consecutive_appearance'] > 0]
    subsector_df = subsector_df[subsector_df['consecutive_appearance'] > 0]
    
    # Merge the two dataframes
    combined_df = pd.concat([sector_df, subsector_df]).drop_duplicates(subset='Symbol').reset_index(drop=True)
    return combined_df


# Firestore data fetching functions
# OPTIMIZED: Added caching with 5 minute TTL
@st.cache_data(ttl=300)
def fetch_bbands_data_from_firestore(sector):
    collection_ref = db.collection('BBands_Results').document(sector).collection('Symbols')
    docs = collection_ref.stream()
    data = []
    for doc in docs:
        doc_dict = doc.to_dict()
        data.append({
            'Symbol': doc_dict.get('Symbol', ''),
            'Crossing Daily Band': doc_dict.get('Crossing Daily Band', ''),
            'Crossing Weekly Band': doc_dict.get('Crossing Weekly Band', ''),
            'Crossing Monthly Band': doc_dict.get('Crossing Monthly Band', '')
        })
    return pd.DataFrame(data)

# OPTIMIZED: Added caching with 5 minute TTL
@st.cache_data(ttl=300)
def fetch_roc_stddev_data_from_firestore(performance_type):
    collection_ref = db.collection('ROCSTDEV_Results').document(performance_type).collection('Top_Symbols')
    docs = collection_ref.stream()
    data = [doc.to_dict() for doc in docs]
    df = pd.DataFrame(data)
    
    # Reorder columns
    df = df[['Symbol', 'ROC/STDDEV', 'RSI', 'Sector']]
    
    # Sort by ROC/STDDEV in descending order
    df = df.sort_values(by='ROC/STDDEV', ascending=False)
    
    return df

# OPTIMIZED: Added caching with 5 minute TTL
@st.cache_data(ttl=300)
def fetch_z_score_data_from_firestore(score_type):
    collection_ref = db.collection('Z_score_results').document(score_type).collection('Records')
    docs = collection_ref.stream()
    data = [doc.to_dict() for doc in docs]

    return pd.DataFrame(data)

# OPTIMIZED: Added caching with 10 minute TTL to reduce redundant API calls
@st.cache_data(ttl=600)
def fetch_data(symbol, api_key):
    """Fetch historical data for a given symbol"""
    url = f'https://eodhistoricaldata.com/api/eod/{symbol}.US?api_token={api_key}&period=d&fmt=json'
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            return symbol, pd.DataFrame(data)
        except ValueError as e:
            print(f"Error decoding JSON for {symbol}: {e}")
            return symbol, pd.DataFrame()
    else:
        print(f"Failed to fetch data for {symbol}: {response.status_code}")
        return symbol, pd.DataFrame()

# OPTIMIZED: Added caching with 1 minute TTL for real-time price
@st.cache_data(ttl=60)
def fetch_real_time_price(symbol, api_key):
    """Fetch real-time data for a given symbol"""
    url = f'https://eodhd.com/api/real-time/{symbol}.US?api_token={api_key}&fmt=json'
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            return data['close']
        except ValueError as e:
            print(f"Error decoding real-time JSON for {symbol}: {e}")
            return None
    else:
        print(f"Failed to fetch real-time data for {symbol}: {response.status_code}")
        return None

# OPTIMIZED: Added caching with 1 minute TTL
@st.cache_data(ttl=60)
def fetch_current_price(symbol, api_token):
    url = f'https://eodhd.com/api/real-time/{symbol}.US?api_token={api_token}&fmt=json'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        current_price = data.get('close', None)
        if current_price is not None:
            current_price = pd.to_numeric(current_price, errors='coerce')
        return current_price
    else:
        print(f"Failed to fetch current price for {symbol}: {response.status_code}, {response.text}")
        return None

# OPTIMIZED: Reduced from 6 API calls to 2 API calls per symbol
@st.cache_data(ttl=300)
def analyze_symbol_optimized(symbol, api_token):
    """
    OPTIMIZED VERSION: Reduced from 6 API calls to 2 API calls
    - 1 call for current price
    - 1 call for historical data (fetches enough data for all calculations)
    """
    current_date = datetime.now()
    
    # Fetch current price (1 API call)
    current_price = fetch_current_price(symbol, api_token)
    
    if current_price is None:
        return symbol, None, None, None, None, None, None
    
    # Fetch historical data once - get 1 year of data (1 API call instead of 5)
    start_date = (current_date - timedelta(days=370)).strftime('%Y-%m-%d')
    end_date = current_date.strftime('%Y-%m-%d')
    df = fetch_historical_data_cached(symbol, api_token, start_date, end_date)
    
    if df.empty:
        return symbol, current_price, None, None, None, None, None
    
    df = df.sort_values('date')
    
    # Calculate all metrics from the single dataset
    try:
        # Previous close (yesterday)
        previous_close_price = df['adjusted_close'].iloc[-2] if len(df) >= 2 else None
        
        # 5 days ago
        start_5_days_price = df['adjusted_close'].iloc[-6] if len(df) >= 6 else None
        
        # Start of month
        start_of_month = get_previous_business_day(current_date.replace(day=1))
        df_month = df[df['date'] >= start_of_month]
        start_month_price = df_month['adjusted_close'].iloc[0] if not df_month.empty else None
        
        # Start of quarter
        start_of_quarter = get_previous_business_day(pd.Timestamp((current_date - pd.offsets.QuarterBegin(startingMonth=1)).strftime('%Y-%m-%d')))
        df_quarter = df[df['date'] >= start_of_quarter]
        start_quarter_price = df_quarter['adjusted_close'].iloc[0] if not df_quarter.empty else None
        
        # Start of year
        start_of_year = get_previous_business_day(current_date.replace(month=1, day=1))
        df_year = df[df['date'] >= start_of_year]
        start_year_price = df_year['adjusted_close'].iloc[0] if not df_year.empty else None
        
        # Calculate percentages
        today_percentage = round(((current_price - previous_close_price) / previous_close_price) * 100, 2) if previous_close_price else None
        five_day_percentage = round(((current_price - start_5_days_price) / start_5_days_price) * 100, 2) if start_5_days_price else None
        mtd_percentage = round(((current_price - start_month_price) / start_month_price) * 100, 2) if start_month_price else None
        qtd_percentage = round(((current_price - start_quarter_price) / start_quarter_price) * 100, 2) if start_quarter_price else None
        ytd_percentage = round(((current_price - start_year_price) / start_year_price) * 100, 2) if start_year_price else None
        
        return symbol, current_price, today_percentage, five_day_percentage, mtd_percentage, qtd_percentage, ytd_percentage
    
    except Exception as e:
        print(f"Error calculating metrics for {symbol}: {e}")
        return symbol, current_price, None, None, None, None, None

# OPTIMIZED: Added caching for historical data
@st.cache_data(ttl=300)
def fetch_historical_data_cached(symbol, api_token, start_date, end_date):
    """Cached version of fetch_historical_data"""
    url = f'https://eodhistoricaldata.com/api/eod/{symbol}.US?api_token={api_token}&from={start_date}&to={end_date}&fmt=json&adjusted=true'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        
        if data and isinstance(data, list) and len(data) > 0 and 'date' in data[0]:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df['adjusted_close'] = pd.to_numeric(df['adjusted_close'], errors='coerce')
            df.dropna(subset=['adjusted_close'], inplace=True)
            return df
        else:
            return pd.DataFrame()
    else:
        print(f"Failed to fetch data for {symbol}: {response.status_code}")
        return pd.DataFrame()

# OPTIMIZED: Use the optimized analyze function
@st.cache_data(ttl=300)
def create_dataframe_optimized(symbols, api_token):
    """
    OPTIMIZED VERSION: Uses analyze_symbol_optimized which makes 2 API calls instead of 6
    Total API calls reduced by ~67% per symbol
    """
    data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(analyze_symbol_optimized, symbol, api_token) for symbol in symbols]
        for future in as_completed(futures):
            result = future.result()
            if result[1] is not None:  # Skip if current_price is None
                data.append(result)
    
    df = pd.DataFrame(data, columns=['Symbol', 'Current Price', 'Today %', '5-Day %', 'MTD %', 'QTD %', 'YTD %'])
    return df

# OPTIMIZED: Added caching for correlation calculations
@st.cache_data(ttl=600)
def calculate_rolling_correlations(symbols, benchmarks, api_token, rolling_window):
    current_date = datetime.now()
    start_date = (current_date - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = current_date.strftime('%Y-%m-%d')
    
    all_symbols = symbols + benchmarks
    historical_data = {}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_historical_data_cached, symbol, api_token, start_date, end_date): symbol for symbol in all_symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    df.set_index('date', inplace=True)
                    historical_data[symbol] = df['adjusted_close']
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {e}")

    combined_df = pd.DataFrame(historical_data)
    rolling_correlations = combined_df.rolling(window=rolling_window).corr(pairwise=True)
    
    results = {}
    for symbol in symbols:
        results[symbol] = {}
        for benchmark in benchmarks:
            rolling_corr = rolling_correlations.loc[(slice(None), benchmark), symbol].unstack().dropna()
            rolling_corr = rolling_corr.rename(columns={benchmark: f'{benchmark}_correlation'})
            results[symbol][benchmark] = rolling_corr
    
    return results

# OPTIMIZED: Added caching for Firestore correlation fetches
@st.cache_data(ttl=300)
def fetch_correlations_from_firestore(ticker, etf_name):
    """
    Fetch the correlation data for a given ticker symbol from Firestore.
    OPTIMIZED: Added caching to reduce Firestore reads
    """
    try:
        doc_ref = db.collection('ETF_Correlations').document(etf_name)
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            return data.get(ticker, {}).get('Top Correlations', {})
        else:
            st.error(f"No correlation data found for {ticker} in {etf_name}")
            return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Extract the 5 highest and 5 lowest correlations
def extract_top_correlations(correlations):
    """
    Extract the '5 Highest' and '5 Lowest' correlations from the Firestore data.
    """
    if not correlations:
        return pd.DataFrame(), pd.DataFrame()

    highest_5 = correlations.get('5 Highest', {})
    lowest_5 = correlations.get('5 Lowest', {})

    highest_5_df = pd.DataFrame(list(highest_5.items()), columns=['Stock', 'Correlation'])
    lowest_5_df = pd.DataFrame(list(lowest_5.items()), columns=['Stock', 'Correlation'])

    return highest_5_df, lowest_5_df

def visualize_rolling_correlations(results):
    charts = []
    for symbol, benchmarks in results.items():
        df_list = []
        for benchmark, df in benchmarks.items():
            df = df.reset_index()
            df_list.append(df[['date', f'{benchmark}_correlation']].rename(columns={f'{benchmark}_correlation': 'correlation'}).assign(benchmark=benchmark))
        
        combined_df = pd.concat(df_list)
        
        highlight = alt.selection_point(fields=['benchmark'], bind='legend')
        
        base = alt.Chart(combined_df).mark_line().encode(
            x='date:T',
            y='correlation:Q',
            color='benchmark:N',
            opacity=alt.condition(highlight, alt.value(1), alt.value(0.2)),
            tooltip=['date:T', 'correlation:Q', 'benchmark:N']
        ).properties(
            title=f'Rolling Correlation of {symbol} with Benchmarks',
            width=800,
            height=400
        ).add_params(
            highlight
        )

        charts.append(base)
    
    final_chart = alt.vconcat(*charts).resolve_scale(
        y='shared'
    )
    
    st.altair_chart(final_chart, use_container_width=True)

# Custom CSS for a darker gray sidebar
st.markdown(
    """
    <style>
    /* Change the background color of the sidebar to a darker gray */
    [data-testid="stSidebar"] {
        background-color: #626770;
    }

    /* Change the text color in the sidebar to dark gray */
    [data-testid="stSidebar"] .css-1d391kg p,
    [data-testid="stSidebar"] .css-1d391kg label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #333333;
    }
    
    /* Customize the selectbox dropdown to match the darker gray theme */
    .stSelectbox label {
        color: #333333;
    }
    
    .stSelectbox .css-11unzgr {
        background-color: #c0c0c0;
        color: #333333;
    }

    /* Customize the dataframe header and cell colors to match the theme */
    .dataframe thead th {
        background-color: #c0c0c0;
        color: #333333;
    }

    .dataframe tbody tr td {
        background-color: #e6e6e6;
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the image from a URL or a local file
image_url = "momento_logo.png"
image = Image.open(image_url)

# Display the logo in the sidebar
st.sidebar.image(image, use_column_width=True)

# Sidebar for analysis selection
st.sidebar.title("Select Analysis Type")
selected_analysis = st.sidebar.radio("Analysis Type", ["BBands analysis", "Sector Overall Performance", "ROC/STDDEV analysis", "Z Score Analysis", "Trailing Correlation Analysis"])

# Sidebar for sector/subsector selection based on analysis type
if selected_analysis == "BBands analysis":
    st.sidebar.title("Select Sector")
    selected_sector = st.sidebar.radio("Sectors", real_sectors + real_subsectors)
    df = fetch_bbands_data_from_firestore(selected_sector)

elif selected_analysis == "ROC/STDDEV analysis":
    st.sidebar.title("Select Performance Type")
    
    performance_type = st.sidebar.radio("Performance Type", ["Sector_Performers", "Subsector_Performers", "Consecutive_Appearances"])
    
    if performance_type in ["Sector_Performers", "Subsector_Performers"]:
        df = fetch_roc_stddev_data_from_firestore(performance_type)
    
    elif performance_type == "Consecutive_Appearances":
        bucket_name = st.secrets['bucket_name']
        sector_csv_path = st.secrets['sector_csv_path']
        subsector_csv_path = st.secrets['subsector_csv_path']
        
        # OPTIMIZED: Now uses cached version
        sector_df = load_csv_from_s3(bucket_name, sector_csv_path)
        subsector_df = load_csv_from_s3(bucket_name, subsector_csv_path)
        
        sector_df_clean = remove_nulls(sector_df)
        sector_consecutive_df = calculate_consecutive_appearances(sector_df_clean)
        subsector_consecutive_df = calculate_consecutive_appearances(subsector_df)
        
        api_key = st.secrets['API_KEY']
        df = process_consecutive_returns(sector_consecutive_df, subsector_consecutive_df, api_key)
        
        df = df.sort_values(['consecutive_day_return'], ascending=[False])

    else:
        st.write("Invalid performance type selected.")


elif selected_analysis == "Z Score Analysis":
    st.sidebar.title("Select Score Type")
    score_type = st.sidebar.radio("Score Type", ["Top_Sectors", "Top_Subsectors"])
    df = fetch_z_score_data_from_firestore(score_type)


# Color mapping for different bands (for BBands analysis)
color_map = {
    'LBand 1STD': 'background-color: lightcoral',
    'LBand 2STD': 'background-color: red',
    'UBand 1STD': 'color: black; background-color: lightgreen',
    'UBand 2STD': 'background-color: green',
    'Mid Zone': '',
}

def highlight_cells(val):
    return color_map.get(val, '')

def prioritize_bands(df):
    band_priority = {
        'UBand 2STD': 1,
        'UBand 1STD': 2,
        'LBand 1STD': 3,
        'LBand 2STD': 4,
        'Mid Zone': 5
    }
    df['Priority'] = df[['Crossing Daily Band', 'Crossing Weekly Band', 'Crossing Monthly Band']].apply(
        lambda x: min(band_priority.get(x[0], 5), band_priority.get(x[1], 5), band_priority.get(x[2], 5)), axis=1
    )
    return df.sort_values('Priority').drop(columns='Priority')

def generate_tradingview_embed(ticker):
    return f"""
    <iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_c2a09&symbol={ticker}&interval=D&hidesidetoolbar=1&symboledit=1&saveimage=1&toolbarbg=f1f3f6&studies=[%7B%22id%22%3A%22BB%40tv-basicstudies%22%2C%22inputs%22%3A%5B20%2C2%5D%7D]&theme=Dark&style=1&timezone=exchange&withdateranges=1&hideideas=1&studies_overrides={{}}&overrides={{}}&enabled_features=[]&disabled_features=[]&locale=en&utm_source=www.tradingview.com&utm_medium=widget&utm_campaign=chart&utm_term={ticker}" width="100%" height="600" frameborder="0" allowfullscreen></iframe>
    """

# Color mapping for percentage columns
def color_percentages(val):
    if pd.isna(val):
        return ''
    elif val < 0:
        return 'background-color: lightcoral; color: black'
    else:
        return 'background-color: lightgreen; color: black'

# Main code
if selected_analysis == "BBands analysis":
    sorted_df = prioritize_bands(df)
    highlighted_df = sorted_df.style.applymap(highlight_cells, subset=['Crossing Daily Band', 'Crossing Weekly Band', 'Crossing Monthly Band'])

    st.title(f"{selected_sector} - Bollinger Bands Analysis")
    st.dataframe(highlighted_df, height=500, width=1000)

    selected_ticker = st.selectbox("Select Ticker to View Chart", sorted_df['Symbol'])

    col1, col2 = st.columns([3, 1])

    with col1:
        chart_html = generate_tradingview_embed(selected_ticker)
        st.components.v1.html(chart_html, height=600)

    # OPTIMIZED: Now uses cached analyze_symbol_optimized
    symbol, current_price, today_percentage, five_day_percentage, mtd_percentage, qtd_percentage, ytd_percentage = analyze_symbol_optimized(selected_ticker, api_token)

    if current_price is not None:
        with col2:
            st.subheader(f"{selected_ticker}")
            st.write(f"**Current Price:** {current_price}")
            st.write(f"**Today:** {today_percentage}%")
            st.write(f"**5-Day:** {five_day_percentage}%")
            st.write(f"**MTD:** {mtd_percentage}%")
            st.write(f"**QTD:** {qtd_percentage}%")
            st.write(f"**YTD:** {ytd_percentage}%")
    else:
        st.write(f"Could not fetch data for {selected_ticker}. Please try again later.")

    # OPTIMIZED: Now uses cached function
    correlations = fetch_correlations_from_firestore(selected_ticker, selected_sector)
    
    if correlations:
        lowest_5, highest_5 = extract_top_correlations(correlations)
        
        st.subheader(f"5 Highest Correlations for {selected_ticker}")
        st.dataframe(lowest_5)

        st.subheader(f"5 Lowest Correlations for {selected_ticker}")
        st.dataframe(highest_5)
        
    # Create the scatter plot for ROC/STDDEV vs RSI
    if 'ROC/STDDEV' in df.columns and 'RSI' in df.columns:
        x_min, x_max = df['ROC/STDDEV'].min(), df['ROC/STDDEV'].max()
        y_min, y_max = df['RSI'].min(), df['RSI'].max()
        
        scatter = alt.Chart(df).mark_circle(size=60).encode(
            x=alt.X('ROC/STDDEV', scale=alt.Scale(domain=[x_min, x_max]), title='ROC/STDDEV'),
            y=alt.Y('RSI', scale=alt.Scale(domain=[y_min, y_max]), title='RSI'),
            color=alt.Color('Symbol', legend=None),
            tooltip=['Symbol', 'ROC/STDDEV', 'RSI']
        ).interactive()
        
        text = scatter.mark_text(
            align='left',
            baseline='middle',
            dx=7,
            fontSize=10
        ).encode(
            text='Symbol'
        )
        
        chart = scatter + text
        
        chart = chart.properties(
            title='ROC/STDDEV vs RSI Scatter Plot'
        ).configure_axis(
            grid=True
        ).configure_title(
            fontSize=20
        ).configure_legend(
            labelFontSize=12,
            titleFontSize=14
        )
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("The dataframe does not contain the required columns for the scatter plot.")


# New Section: Sector and Subsector Performance
elif selected_analysis == "Sector Overall Performance":
    # OPTIMIZED: Now uses cached version, and refresh properly clears cache
    st.title("Sector and Subsector Performance")
    
    # Add info about cache
    st.info("📊 Data is cached for 5 minutes. Click 'Clear Cache & Refresh' to force update all data.")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Clear Cache & Refresh"):
            st.cache_data.clear()
            st.rerun()
    
    # OPTIMIZED: All these now use cached functions
    sector_df = create_dataframe_optimized(sectors, api_token)
    subsector_df = create_dataframe_optimized(subsectors, api_token)
    ratecut_etfs_df = create_dataframe_optimized(ratecut_etfs, api_token)
    macro_etfs_df = create_dataframe_optimized(macro_etfs, api_token)

    # Create tabs for different dataframes
    tab1, tab2, tab3, tab4 = st.tabs(["Sector Performance", "Subsector Performance", "Commodities & Metals Performance", "Macro Performance"])

    # Sector performance tab
    with tab1:
        st.subheader("Sector DataFrame")
        sector_df_styled = sector_df.style.applymap(color_percentages, subset=['Today %', '5-Day %', 'MTD %', 'QTD %', 'YTD %'])
        sector_df_styled = sector_df_styled.format({
            'Current Price': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'Today %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            '5-Day %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'MTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'QTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'YTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
        })
        st.dataframe(sector_df_styled, height=500, width=1000)
    
    # Subsector performance tab
    with tab2:
        st.subheader("Subsector DataFrame")
        subsector_df_styled = subsector_df.style.applymap(color_percentages, subset=['Today %', '5-Day %', 'MTD %', 'QTD %', 'YTD %'])
        subsector_df_styled = subsector_df_styled.format({
            'Current Price': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'Today %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            '5-Day %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'MTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'QTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'YTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
        })
        st.dataframe(subsector_df_styled, height=500, width=1000)
    
    # Bonds & Metals performance tab
    with tab3:
        st.subheader("Commodities & Metals DataFrame")
        ratecut_etfs_df_styled = ratecut_etfs_df.style.applymap(color_percentages, subset=['Today %', '5-Day %', 'MTD %', 'QTD %', 'YTD %'])
        ratecut_etfs_df_styled = ratecut_etfs_df_styled.format({
            'Current Price': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'Today %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            '5-Day %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'MTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'QTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'YTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
        })
        st.dataframe(ratecut_etfs_df_styled, height=500, width=1000)
    
    # Countries performance tab
    with tab4:
        st.subheader("Countries DataFrame")
        macro_etfs_df_styled = macro_etfs_df.style.applymap(color_percentages, subset=['Today %', '5-Day %', 'MTD %', 'QTD %', 'YTD %'])
        macro_etfs_df_styled = macro_etfs_df_styled.format({
            'Current Price': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'Today %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            '5-Day %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'MTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'QTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'YTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
        })
        st.dataframe(macro_etfs_df_styled, height=500, width=1000)
    

# Add Trailing Correlation Analysis
elif selected_analysis == "Trailing Correlation Analysis":

    st.write("Enter the symbols and benchmarks for analysis.")
    symbols = st_tags(label='### Symbols', text='Add symbols (e.g., AAPL)', suggestions=["AAPL", "GOOGL", "TSLA"], maxtags=10)
    benchmarks = st_tags(label='### Benchmarks', text='Add benchmarks (e.g., SPY)', suggestions=["SPY", "DIA", "QQQ"], maxtags=10)

    rolling_window = st.slider("Rolling Window (Days)", min_value=10, max_value=100, value=30)
    
    if st.button("Run Trailing Correlation Analysis"):
        # OPTIMIZED: Now uses cached version
        results = calculate_rolling_correlations(symbols, benchmarks, api_token, rolling_window)
        visualize_rolling_correlations(results)
