# Use python as base image
FROM python:3.8

WORKDIR /app

# Install Python dependencies
ADD ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.27

# Copy the whole directory to the working directory
COPY . .

# Ingest the data
RUN python3 ingest_data.py

# Start the server
CMD ["python3", "retrieval_chatbot.py"]