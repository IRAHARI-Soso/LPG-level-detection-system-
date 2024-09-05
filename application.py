from flask import Flask, request, render_template, redirect, url_for, session, jsonify, make_response, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
from datetime import datetime
import pandas as pd
from io import BytesIO
import threading
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your secret key

# MySQL connection setup
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',  # Add your MySQL password here
    'database': 'lpg_monitoring'
}

def get_db_connection():
    connection = mysql.connector.connect(**db_config)
    return connection

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Static files route
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

# Dashboard route (requires login)
@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        return render_template('dashboard.html', user=session['user'])
    return redirect(url_for('login'))  # Redirect to login if not logged in

# Route to fetch LPG data from the database
@app.route('/get_lpg_data')
def get_lpg_data():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    cursor.execute("SELECT * FROM lpg_data")
    lpg_data = cursor.fetchall()

    cursor.execute("SELECT * FROM prediction")
    prediction_data = cursor.fetchall()

    cursor.close()
    connection.close()

    return jsonify({'lpg_data': lpg_data, 'prediction_data': prediction_data})

# Route to insert data into the database
def insert_data(distance, gas_mass, lpg_used, lpg_remaining, gas_detected, decision, hours_remaining):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        sql = """INSERT INTO lpg_data 
                (distance, gas_mass, lpg_used, lpg_remaining, gas_detected, timestamp, decision, hours_remaining) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""

        values = (distance, gas_mass, lpg_used, lpg_remaining, gas_detected, timestamp, decision, hours_remaining)

        cursor.execute(sql, values)
        connection.commit()

        return True
    except mysql.connector.Error as error:
        print(f"Failed to insert data: {error}")
        return False
    finally:
        cursor.close()
        connection.close()

@app.route('/insert_data', methods=['POST'])
def insert_data_route():
    try:
        distance = float(request.form.get('distance'))
        gas_mass = float(request.form.get('gas_mass'))
        lpg_used = float(request.form.get('lpg_used'))
        lpg_remaining = float(request.form.get('lpg_remaining'))
        gas_detected = int(request.form.get('gas_detected'))
        decision = request.form.get('decision', 'Unknown')  # Default to 'Unknown' if not provided
        hours_remaining = float(request.form.get('hours_remaining'))

        # Insert the data into the database
        success = insert_data(distance, gas_mass, lpg_used, lpg_remaining, gas_detected, decision, hours_remaining)

        if not success:
            return jsonify({"status": "failure"}), 500
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Exception occurred: {e}")
        return jsonify({'status': 'error'}), 500

# Route to export LPG data as an Excel file
@app.route('/export_lpg_data')
def export_lpg_data():
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute("SELECT * FROM lpg_data")
        lpg_data = cursor.fetchall()

        cursor.close()
        connection.close()

        # Convert the data to a pandas DataFrame
        df = pd.DataFrame(lpg_data)

        # Convert DataFrame to Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='LPG Data')
        output.seek(0)  # Move the cursor to the beginning of the stream

        # Send the Excel file as a response
        response = make_response(output.read())
        response.headers['Content-Disposition'] = 'attachment; filename=lpg_data.xlsx'
        response.headers['Content-type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        return response

    except Exception as e:
        return f"Error: {e}", 500

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        cursor.close()
        connection.close()

        if user and check_password_hash(user['password'], password):
            session['user'] = user['name']
            return redirect(url_for('dashboard'))
        return 'Invalid email or password'
    return render_template('login.html')

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user:
            return 'Email address already in use'

        cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, password))
        connection.commit()

        cursor.close()
        connection.close()

        return redirect(url_for('login'))  # Redirect to login page after successful signup
    return render_template('signup.html')


@app.route('/delete_lpg_data/<int:id>', methods=['DELETE'])
def delete_lpg_data(id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # Delete the LPG data from the database
        cursor.execute("DELETE FROM lpg_data WHERE id = %s", (id,))
        connection.commit()

        cursor.close()
        connection.close()

        return jsonify({"status": "success"}), 200
    except mysql.connector.Error as error:
        print(f"Failed to delete data: {error}")
        return jsonify({"status": "failure"}), 500


@app.route('/delete_prediction_data/<int:id>', methods=['DELETE'])
def delete_prediction_data(id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # Delete the prediction data from the database
        cursor.execute("DELETE FROM prediction WHERE id = %s", (id,))
        connection.commit()

        cursor.close()
        connection.close()

        return jsonify({"status": "success"}), 200
    except mysql.connector.Error as error:
        print(f"Failed to delete data: {error}")
        return jsonify({"status": "failure"}), 500




# Logout route
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

def continuous_data_insertion():
    while True:
        # This function is not needed if data is inserted from Arduino directly
        # Removing it to avoid confusion
        time.sleep(10)  # Adjust the interval as needed

if __name__ == '__main__':
    # Start the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)
