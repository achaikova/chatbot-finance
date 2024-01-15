# Use python as base image
FROM python:3.8

WORKDIR /app

# Install Python dependencies
ADD ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt
RUN pip install llama-cpp-python

# Copy the whole directory to the working directory
COPY data/ ./data/
COPY model/ ./model/
COPY index/ ./index/
COPY script/ ./script/
COPY init_model.sh .
# Check the model
RUN ./init_model.sh

# Ingest the data
# RUN python3 ingest_data.py
WORKDIR script/

ENV HOST=0.0.0.0
ENV LISTEN_PORT 8000
EXPOSE 8000
# Start the server
CMD ["chainlit", "run", "retrieval_chatbot.py"]
