# Create an environment in windows

- py -3 -m venv .venv

# Activate the environmentmoney laundering

- .venv\Scripts\activate

- Set-ExecutionPolicy Unrestricted -Scope Process (jika eror policy)

# Install Flask

- pip install Flask

# Men set app.py sebagai run utama

- set FLASK_APP=app.py

# Mengaktifkan Debug

- flask --app app run --debug
