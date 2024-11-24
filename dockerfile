# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 to allow external access to the container
EXPOSE 8080

# Define environment variable for Flask app
ENV FLASK_APP=app.py

# Run the application using Gunicorn (preferred for production)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
