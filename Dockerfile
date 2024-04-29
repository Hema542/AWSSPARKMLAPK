FROM python:3.9

# Copy your Python script into the Docker image
COPY your_script.py /app/

# Set the working directory
WORKDIR /app

# Install dependencies
RUN pip install -r requirements.txt

# Specify the command to run your script
CMD ["python", "your_script.py"]

