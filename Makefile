# Define your virtual environment and flask app
VENV = venv
FLASK_APP = app.py

# Install dependencies
install:
	python3 -m venv $(VENV)
	./$(VENV)/bin/pip install -r requirements.txt

# Run the Flask application
run: 
	FLASK_APP=app.py FLASK_ENV=development flask run --host=0.0.0.0 --port=3000

# Clean up virtual environment
clean:
	find . -type f -name '*.pyc' -exec rm -f {} \;

# Reinstall all dependencies
reinstall: clean install