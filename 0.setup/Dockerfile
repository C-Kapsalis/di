# Start from the Gitpod workspace full image
FROM gitpod/workspace-full-vnc

USER root

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    apt-transport-https \
    ca-certificates \
    software-properties-common \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Add Google Chrome repository and install Google Chrome
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && \
    apt-get install -y google-chrome-stable



# Download and install the ChromeDriver
#RUN wget https://chromedriver.storage.googleapis.com/$(wget -q -O - https://chromedriver.storage.googleapis.com/LATEST_RELEASE)/chromedriver_linux64.zip && \
#    unzip chromedriver_linux64.zip && \
#    mv chromedriver /usr/local/bin/chromedriver && \
#    chmod +x /usr/local/bin/chromedriver && \
#    rm chromedriver_linux64.zip

# Download and install the specific version of ChromeDriver for Chrome 125
RUN wget https://storage.googleapis.com/chrome-for-testing-public/125.0.6422.141/linux64/chromedriver-linux64.zip && \
    unzip chromedriver-linux64.zip && \
    mv chromedriver-linux64/chromedriver /usr/local/bin/chromedriver && \
    chmod +x /usr/local/bin/chromedriver && \
    rm -rf chromedriver-linux64 chromedriver-linux64.zip


USER gitpod

# Check if Google Chrome and ChromeDriver were installed successfully
RUN google-chrome --version && chromedriver --version


# Install necessary Python packages
COPY requirements.txt /workspace/
RUN pip install -r /workspace/requirements.txt
