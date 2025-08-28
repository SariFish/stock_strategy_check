Streamlit app for earnings drift backtest

run local
pip install -r requirements.txt
streamlit run app.py

inputs
tickers list comma separated
benchmark
start and end date
three month signal threshold

output
summary table
equity curve plot
trades table and csv download

deploy on Streamlit Cloud
push this folder to a new GitHub repo
on share streamlit io create a new app and choose app py as the entry point
set Python version to 3 dot 11 or newer
