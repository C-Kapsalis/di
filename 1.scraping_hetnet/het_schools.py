from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd

# Setup WebDriver options
options = Options()
options.headless = True  # Run headless Chrome
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-gpu')
options.add_argument('--remote-debugging-port=9222')  # Required for running inside Gitpod

# Set the path to the Chrome binary
options.binary_location = '/usr/bin/google-chrome'  # Update this path if needed

# Setup WebDriver
service = Service('/usr/local/bin/chromedriver')  # Path to the ChromeDriver
driver = webdriver.Chrome(service=service, options=options)

# Define the URL of the page to scrape
url = "https://www.hetwebsite.net/het/thought.htm"

# Load the page
driver.get(url)

# Find all 'li' elements
li_elements = driver.find_elements(By.TAG_NAME, 'li')

# Data structure to store the extracted information
names = []
details = []
hyperlinks = []

# Iterate over each 'li' element
for li in li_elements:
    name = li.text
    if name in ['Home', 'Alphabetical Index', 'Schools of Thought', 'Essays & Surveys', 'Contact', 'Search', 'Schools of Political Economy', 'Neoclassical Schools', 'Alternative Schools', 'Thematic Schools', 'Institutions', 'Pre-Classical', 'Classical', 'Anglo-American', 'Continental', 'Heterodox', 'Keynesian', 'Empirical', 'Theoretical', 'Universities', 'Societies']: 
        continue
    print(name)
    try:
        a_tag = li.find_element(By.TAG_NAME, 'a')
        hyperlink = a_tag.get_attribute('href')

        # Visit the hyperlink
        driver.get(hyperlink)

        # Allow time for the page to load
        time.sleep(2)  # Adjust as necessary

        # Extract text from 'p' elements
        p_elements = driver.find_elements(By.TAG_NAME, 'p')
        details.append(' '.join([p.text for p in p_elements]))
        print(details[:10])
        # Store the extracted information
        names.append(name)
        print(name)
        hyperlinks.append(hyperlink)
        print(hyperlink)

        # Navigate back to the original page
        driver.back()
        time.sleep(1)  # Adjust as necessary
    except Exception as e:
        print(f"An error occurred: {e}")
        continue

# Quit the WebDriver
driver.quit()

# Print or process the extracted data
data = pd.DataFrame({'school': names, 'details': details, 'hyperlink': hyperlinks})
data.to_csv('het_schools.csv', index=False, header=True)

import os
import subprocess

# List files in the current directory (example of using os module)
files = os.listdir('.')
print(files)

# Run Git commands using subprocess
subprocess.run(["git", "add", "."])
subprocess.run(["git", "commit", "-m", "ran scraping"])
subprocess.run(["git", "push", "origin"])
