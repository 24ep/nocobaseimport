# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=flask_nocobase_importer.app
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
# Note: FLASK_UPLOAD_FOLDER will be set by environment variable when running the container,
# as it might be mapped to a volume. Default can be set in app.py.

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY flask_nocobase_importer/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application directory into the container at /app
COPY ./flask_nocobase_importer ./flask_nocobase_importer

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application
# The FLASK_APP, FLASK_RUN_HOST, FLASK_RUN_PORT env vars are not directly used by gunicorn CMD
# but can remain as they might be useful if running 'flask run' manually.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "flask_nocobase_importer.app:app"]
