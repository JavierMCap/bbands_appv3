import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
from concurrent.futures import ThreadPoolExecutor, as_completed
import altair as alt
import re
from PIL import Image
#api key below
api_token = st.secrets['API_KEY']
sectors = ['XLK', 'XLC', 'XLV', 'XLF', 'XLP', 'XLI', 'XLE', 'XLY', 'XLB', 'XLU', 'XLRE', 'MAGS', 'SPY']
subsectors = ['GDX', 'UFO', 'KBE', 'KRE', 'AMLP', 'ITA', 'ITB', 'IAK', 'SMH', 'PINK', 'XBI', 'NLR']


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

def fetch_previous_close_price(symbol, api_token):
    end_date = datetime.now() - BDay(1)
    start_date = end_date - BDay(5)
    url = f'https://eodhistoricaldata.com/api/eod/{symbol}.US?api_token={api_token}&from={start_date.strftime("%Y-%m-%d")}&to={end_date.strftime("%Y-%m-%d")}&fmt=json&adjusted=true'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df['adjusted_close'] = pd.to_numeric(df['adjusted_close'], errors='coerce')
        df.dropna(subset=['adjusted_close'], inplace=True)
        previous_close_price = df['adjusted_close'].iloc[-1] if not df.empty else None
        return previous_close_price
    else:
        print(f"Failed to fetch previous close price for {symbol}: {response.status_code}, {response.text}")
        return None

def fetch_historical_data(symbol, api_token, start_date, end_date):
    url = f'https://eodhistoricaldata.com/api/eod/{symbol}.US?api_token={api_token}&from={start_date}&to={end_date}&fmt=json&adjusted=true'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df['adjusted_close'] = pd.to_numeric(df['adjusted_close'], errors='coerce')
        df.dropna(subset=['adjusted_close'], inplace=True)
        return df
    else:
        print(f"Failed to fetch data for {symbol}: {response.status_code}, {response.text}")
        return pd.DataFrame()


def analyze_symbol(symbol, api_token):
    current_date = datetime.now()
    start_of_month = current_date.replace(day=1)
    start_of_quarter = (current_date - pd.offsets.QuarterBegin(startingMonth=1)).strftime('%Y-%m-%d')
    start_of_year = current_date.replace(month=1, day=1)
    start_of_30_days = current_date - timedelta(days=30)
    start_of_5_days = current_date - BDay(5)

    with ThreadPoolExecutor() as executor:
        current_price_future = executor.submit(fetch_current_price, symbol, api_token)
        previous_close_price_future = executor.submit(fetch_previous_close_price, symbol, api_token)
        df_month_future = executor.submit(fetch_historical_data, symbol, api_token, start_of_month.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))
        df_quarter_future = executor.submit(fetch_historical_data, symbol, api_token, start_of_quarter, current_date.strftime('%Y-%m-%d'))
        df_year_future = executor.submit(fetch_historical_data, symbol, api_token, start_of_year.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))
        df_5_days_future = executor.submit(fetch_historical_data, symbol, api_token, start_of_5_days.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))

        current_price = current_price_future.result()
        previous_close_price = previous_close_price_future.result()
        df_month = df_month_future.result()
        df_quarter = df_quarter_future.result()
        df_year = df_year_future.result()
        df_5_days = df_5_days_future.result()

    if current_price is None:
        return symbol, None, None, None, None, None

    if previous_close_price is None:
        today_percentage = None
    else:
        today_percentage = round(((current_price - previous_close_price) / previous_close_price) * 100, 2)

    start_month_price = df_month['adjusted_close'].iloc[0] if not df_month.empty else None
    start_quarter_price = df_quarter['adjusted_close'].iloc[0] if not df_quarter.empty else None
    start_year_price = df_year['adjusted_close'].iloc[0] if not df_year.empty else None
    start_5_days_price = df_5_days['adjusted_close'].iloc[0] if not df_5_days.empty else None

    mtd_percentage = round(((current_price - start_month_price) / start_month_price) * 100, 2) if start_month_price is not None else None
    qtd_percentage = round(((current_price - start_quarter_price) / start_quarter_price) * 100, 2) if start_quarter_price is not None else None
    ytd_percentage = round(((current_price - start_year_price) / start_year_price) * 100, 2) if start_year_price is not None else None
    five_day_percentage = round(((current_price - start_5_days_price) / start_5_days_price) * 100, 2) if start_5_days_price is not None else None

    return symbol, current_price, today_percentage, five_day_percentage, mtd_percentage, qtd_percentage, ytd_percentage

# Function Definitions
def create_dataframe(symbols, api_token):
    data = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(analyze_symbol, symbol, api_token) for symbol in symbols]
        for future in as_completed(futures):
            result = future.result()
            if result[1] is not None:  # Skip if current_price is None
                data.append(result)
    
    df = pd.DataFrame(data, columns=['Symbol', 'Current Price', 'Today %', '5-Day %', 'MTD %', 'QTD %', 'YTD %'])
    return df


# Custom CSS for a darker gray sidebar
st.markdown(
    """
    <style>
    /* Change the background color of the sidebar to a darker gray */
    [data-testid="stSidebar"] {
        background-color: #626770; /* You can adjust this hex code for a lighter or darker shade */
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
        background-color: #c0c0c0; /* Slightly darker than the sidebar background */
        color: #333333;
    }

    /* Customize the dataframe header and cell colors to match the theme */
    .dataframe thead th {
        background-color: #c0c0c0; /* Slightly darker gray */
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

# Load the BBands Excel file
bbands_excel_file_path = 'BBands_ETFs_2024-08-30.xlsx'
bbands_sheets_dict = pd.read_excel(bbands_excel_file_path, sheet_name=None)

# Load the ROC/STDDEV Excel file
roc_stddev_excel_file_path = 'ROCSTDEV_ETF_Analysis_2024-08-30_sheets.xlsx'
roc_stddev_sheets_dict = pd.read_excel(roc_stddev_excel_file_path, sheet_name=None)

z_score_excel_file_path = 'Z_Score_Results_2024-08-30_GHub.xlsx'
z_score_sheets_dict = pd.read_excel(z_score_excel_file_path, sheet_name=None)

# Function to extract date from the filename
def extract_date_from_filename(filename):
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    if match:
        return datetime.strptime(match.group(), '%Y-%m-%d').strftime('%B %d, %Y')
    return 'Unknown Date'

bbands_date = extract_date_from_filename(bbands_excel_file_path)
roc_stddev_date = extract_date_from_filename(roc_stddev_excel_file_path)
zscore_date = extract_date_from_filename(z_score_excel_file_path)



# Load the image from a URL or a local file
image_url = "momento_logo.png"  # Update this URL
image = Image.open(image_url)

# Display the logo in the sidebar
st.sidebar.image(image, use_column_width=True)

# Sidebar for analysis selection
st.sidebar.title("Select Analysis Type")
selected_analysis = st.sidebar.radio("Analysis Type", ["BBands analysis", "Sector Overall Performance", "ROC/STDDEV analysis", "Z Score Analysis"])

# Sidebar for sector/subsector selection based on analysis type
if selected_analysis == "BBands analysis":
    st.sidebar.title("Select Sector")
    selected_sector = st.sidebar.radio("Sectors", list(bbands_sheets_dict.keys()))
    df = bbands_sheets_dict[selected_sector]
    
    #if selection_type == "Sector":
        #selected_sector = st.sidebar.radio("Sectors", sectors)
        #df = create_dataframe([selected_sector], api_token)
    #else:
        #selected_subsector = st.sidebar.radio("Subsectors", subsectors)
        #df = create_dataframe([selected_subsector], api_token)
# New Section: Z Score Analysis

elif selected_analysis == "ROC/STDDEV analysis":
    st.sidebar.title("Select Subsector")
    selected_subsector = st.sidebar.radio("Subsectors", list(roc_stddev_sheets_dict.keys()))
    df = roc_stddev_sheets_dict[selected_subsector]


# Color mapping for different bands (for BBands analysis)
color_map = {
    'LBand 1STD': 'background-color: lightcoral',  # light red
    'LBand 2STD': 'background-color: red',  # stronger red
    'UBand 1STD': 'color: black; background-color: lightgreen',  # light green with black text
    'UBand 2STD': 'background-color: green',  # stronger green
    'Mid Zone': '',  # no color
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

    st.title(f"{selected_sector} - Bollinger Bands Analysis - {bbands_date}")
    st.dataframe(highlighted_df, height=500, width=1000)

    # Display chart and data for selected symbol
    selected_ticker = st.selectbox("Select Ticker to View Chart", sorted_df['Symbol'])

    # Generate and display the TradingView chart
    col1, col2 = st.columns([3, 1])

    with col1:
        chart_html = generate_tradingview_embed(selected_ticker)
        st.components.v1.html(chart_html, height=600)

    # Perform and display the analysis for the selected ticker
    symbol, current_price, today_percentage, five_day_percentage, mtd_percentage, qtd_percentage, ytd_percentage = analyze_symbol(selected_ticker, api_token)

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

# Z Score Analysis Section
elif selected_analysis == "Z Score Analysis":
    st.sidebar.title(f"Select ETF for Z Score Analysis")
    selected_etf = st.sidebar.radio("ETFs", list(z_score_sheets_dict.keys()))
    df = z_score_sheets_dict[selected_etf]
    
    st.title(f"{selected_etf} - Z Score Analysis (52 W) - {zscore_date} ")
    st.dataframe(df, height=500, width=1000)

    # Display chart and data for selected ETF symbol
    selected_ticker = st.selectbox("Select Ticker to View Chart", df['Ticker'])

    # Generate and display the TradingView chart
    col1, col2 = st.columns([3, 1])

    with col1:
        chart_html = generate_tradingview_embed(selected_ticker)
        st.components.v1.html(chart_html, height=600)

    # Perform and display the analysis for the selected ticker
    symbol, current_price, today_percentage, five_day_percentage, mtd_percentage, qtd_percentage, ytd_percentage = analyze_symbol(selected_ticker, api_token)

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

    
elif selected_analysis == "ROC/STDDEV analysis":
    st.title(f"{selected_subsector} - ROC/STDDEV Analysis - {roc_stddev_date}")
    st.dataframe(df, height=500, width=1000)

    # Display chart and data for selected symbol
    selected_ticker = st.selectbox("Select Ticker to View Chart", df['Symbol'])

    # Generate and display the TradingView chart
    col1, col2 = st.columns([3, 1])

    with col1:
        chart_html = generate_tradingview_embed(selected_ticker)
        st.components.v1.html(chart_html, height=600)

    # Perform and display the analysis for the selected ticker
    symbol, current_price, today_percentage, five_day_percentage,  mtd_percentage, qtd_percentage, ytd_percentage = analyze_symbol(selected_ticker, api_token)

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

    # Create the scatter plot for ROC/STDDEV vs RSI
    if 'ROC/STDDEV' in df.columns and 'RSI' in df.columns:

        # Calculate min and max for x and y axes
        x_min, x_max = df['ROC/STDDEV'].min(), df['ROC/STDDEV'].max()
        y_min, y_max = df['RSI'].min(), df['RSI'].max()
        
        # Create scatter plot with dynamic domain
        scatter = alt.Chart(df).mark_circle(size=60).encode(
            x=alt.X('ROC/STDDEV', scale=alt.Scale(domain=[x_min, x_max]), title='ROC/STDDEV'),
            y=alt.Y('RSI', scale=alt.Scale(domain=[y_min, y_max]), title='RSI'),
            color=alt.Color('Symbol', legend=None),
            tooltip=['Symbol', 'ROC/STDDEV', 'RSI']
        ).interactive()
        
        # Add text labels to the points
        text = scatter.mark_text(
            align='left',
            baseline='middle',
            dx=7,
            fontSize=10
        ).encode(
            text='Symbol'
        )
        
        # Combine the scatter plot and text
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
    sector_df = create_dataframe(sectors, api_token)
    subsector_df = create_dataframe(subsectors, api_token)
    
    st.title("Sector and Subsector Performance")

    tab1, tab2 = st.tabs(["Sector Performance", "Subsector Performance"])
    # Round the percentage columns to two decimal places
    

    with tab1:
        st.subheader("Sector DataFrame")
        sector_df_styled = sector_df.style.applymap(color_percentages, subset=['Today %', '5-Day %', 'MTD %', 'QTD %', 'YTD %'])
        sector_df_styled = sector_df_styled.format({'Current Price': '{:.2f}', 'Today %': '{:.2f}', '5-Day %': '{:.2f}', 'MTD %': '{:.2f}', 'QTD %': '{:.2f}', 'YTD %': '{:.2f}'})
        st.dataframe(sector_df_styled, height=500, width=1000)

    with tab2:
        st.subheader("Subsector DataFrame")
        subsector_df_styled = subsector_df.style.applymap(color_percentages, subset=['Today %', '5-Day %', 'MTD %', 'QTD %', 'YTD %'])
        subsector_df_styled = subsector_df_styled.format({'Current Price': '{:.2f}', 'Today %': '{:.2f}', '5-Day %': '{:.2f}', 'MTD %': '{:.2f}', 'QTD %': '{:.2f}', 'YTD %': '{:.2f}'})
        st.dataframe(subsector_df_styled, height=500, width=1000)
