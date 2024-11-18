# from flask import Flask, request, render_template, send_file
from flask import Flask, render_template, request, send_file
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime
import matplotlib.pyplot as plt
import io
import numpy as np

app = Flask(__name__)

# Load and preprocess data
file_path = 'ref_escore.csv'
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'])
data['date_ordinal'] = data['date'].map(pd.Timestamp.toordinal)

# Extract company names from the columns
company_columns = [col for col in data.columns if col.endswith('_diff')]
companies = [col.replace('_diff', '').capitalize() for col in company_columns]

# Initialize models for each company
models = {}
for company in company_columns:
    X = data[['date_ordinal', 'industry_tone']]
    y = data[company]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model on validation set
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f'MSE for {company}: {mse}')
    
    models[company] = model

@app.route('/')
def index():
    return render_template('index.html', companies=companies)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form['company'].lower() + '_diff'
    date = request.form['date']
    period = int(request.form['period'])
    
    date_ordinal = datetime.strptime(date, '%Y-%m-%d').toordinal()
    
    # Prepare inputs for the specified period
    predictions = []
    for i in range(period):
        future_date_ordinal = date_ordinal + i
        X_new = pd.DataFrame({'date_ordinal': [future_date_ordinal], 'industry_tone': [0]})
        prediction = models[company].predict(X_new)[0]
        
        # Introduce some randomness to simulate zigzag pattern
        noise = np.random.normal(0, 0.001)
        prediction += noise
        
        predictions.append((datetime.fromordinal(future_date_ordinal).strftime('%Y-%m-%d'), prediction))
    
    # Generate the plot
    dates = [datetime.fromordinal(date_ordinal + i).strftime('%Y-%m-%d') for i in range(period)]
    values = [pred for _, pred in predictions]
    plt.figure(figsize=(10, 5))
    plt.plot(dates, values, marker='o', linestyle='-', color='b')
    plt.xlabel('Date')
    plt.ylabel('Prediction')
    plt.title(f'Predictions for {request.form["company"].capitalize()}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    return render_template('index.html', predictions=predictions, companies=companies, selected_company=request.form['company'], start_date=date, period=period, plot_url='/plot.png')

@app.route('/plot.png')
def plot_png():
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
