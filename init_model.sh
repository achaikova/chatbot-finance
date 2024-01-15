#!/bin/bash

 # Check if the MODEL environment variable is set
 if [ -z "$MODEL_PATH" ]
 then
     echo "Exporting MODEL_PATH variable"
     MODEL_PATH=$(python -c 'from configuration import MODEL_PATH; print(MODEL_PATH)')
 fi

 # Check if the MODEL_DOWNLOAD_URL environment variable is set
 if [ -z "$MODEL_DOWNLOAD_URL" ]
 then
     echo "Exporting MODEL_DOWNLOAD_URL variable"
     MODEL_DOWNLOAD_URL=$(python -c 'from configuration import MODEL_DOWNLOAD_URL; print(MODEL_DOWNLOAD_URL)')
 fi

 # Check if the model file exists
 if [ ! -f $MODEL_PATH ]; then
     echo "Model file not found. Downloading..."
     # Check if curl is installed
     if ! [ -x "$(command -v curl)" ]; then
         echo "curl is not installed. Installing..."
         apt-get update --yes --quiet
         apt-get install --yes --quiet curl
     fi
     # Download the model file
     curl -L -o $MODEL_PATH $MODEL_DOWNLOAD_URL
     if [ $? -ne 0 ]; then
         echo "Download failed. Trying with TLS 1.2..."
         curl -L --tlsv1.2 -o $MODEL_PATH $MODEL_DOWNLOAD_URL
     fi
 else
     echo "$MODEL_PATH model found."
 fi