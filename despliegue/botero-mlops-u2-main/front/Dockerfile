

# Use the slim Python base image
FROM python:3.10-slim as builder

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app/

# Expose the Flask app port
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]
