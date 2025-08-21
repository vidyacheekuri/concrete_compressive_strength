# Use a standard Python 3.11 image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all your project files into the container
COPY . .

# Install the Python libraries AND run your training script
RUN pip install --no-cache-dir -r requirements.txt && python train.py

# Expose the port the app will run on
EXPOSE 7860

# Command to start the web server when the container launches
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]