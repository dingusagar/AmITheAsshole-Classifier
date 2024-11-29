# Use a lightweight Linux base image
FROM python:3.10-slim


# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, curl, and other dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama using curl
RUN curl -fsSL https://ollama.com/install.sh | sh

# Add a non-root user (optional for security)
RUN useradd -m -u 1000 user
USER user

# Set environment variables for Ollama
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    OLLAMA_HOST=0.0.0.0

# Set working directory
WORKDIR $HOME/app



# Install Python dependencies for your application
RUN pip3 install ollama

# Pull the required Ollama model
#RUN ollama serve & sleep 5 && ollama pull llama3.2:1b

RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install gradio transformers
# Copy your Python application
COPY app.py $HOME/app/app.py

# Expose the Ollama server port
EXPOSE 11434
EXPOSE 7860

# Set the entrypoint to bash (optional)
#ENTRYPOINT ["/bin/bash"]
ENV GRADIO_SERVER_NAME=0.0.0.0

RUN echo "Build Complete"

# Command to run your Python application
#CMD ["sh", "-c", "ollama run llama3.2:1b > ollama.log 2>&1 & python3 app.py"]
CMD ["python3", "app.py"]