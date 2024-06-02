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
url = "https://www.hetwebsite.net/het/essays.htm#micro"

# Load the page
driver.get(url)

# Find all 'li' elements that link to essays
li_elements = driver.find_elements(By.TAG_NAME, 'li')

# Data structure to store the extracted information
essay_names = []
chapter_texts = []
hyperlinks = []

# Iterate over each 'li' element
for li in li_elements:
    essay_name = li.text
    if essay_name in ['Home', 'Alphabetical Index', 'Schools of Thought', 'Essays & Surveys', 'Contact', 'Search', 'Theory of Value', 'Macroeconomics', 'Microeconomics', 'Other Items']:
        continue
    print(essay_name)
    try:
        a_tag = li.find_element(By.TAG_NAME, 'a')
        essay_link = a_tag.get_attribute('href')
        print(essay_link)
        # Visit the essay page
        driver.get(essay_link)

        # Allow time for the page to load
        time.sleep(2)  # Adjust as necessary

        # Check if the page contains <p> elements
        p_elements = driver.find_elements(By.TAG_NAME, 'p')
        if p_elements:
            # If <p> elements are found, extract the text
            essay_text = ' '.join([p.text for p in p_elements])
        else:
            # If no <p> elements are found, look for links to chapters
            chapter_links = driver.find_elements(By.TAG_NAME, 'a')
            chapter_texts_list = []
            for chapter_link in chapter_links:
                chapter_href = chapter_link.get_attribute('href')
                if chapter_href and 'essays' in chapter_href:  # Ensure it's a valid chapter link
                    driver.get(chapter_href)
                    time.sleep(2)
                    chapter_p_elements = driver.find_elements(By.TAG_NAME, 'p')
                    chapter_text = ' '.join([p.text for p in chapter_p_elements])
                    chapter_texts_list.append(chapter_text)
                    driver.back()
                    time.sleep(1)
            essay_text = ' '.join(chapter_texts_list)

        # Store the extracted information
        essay_names.append(essay_name)
        chapter_texts.append(essay_text)
        print(essay_text)
        hyperlinks.append(essay_link)

        # Navigate back to the initial essays list
        driver.back()
        time.sleep(1)  # Adjust as necessary
    except Exception as e:
        print(f"An error occurred: {e}")
        continue

# Quit the WebDriver
driver.quit()

# Print or process the extracted data
data = pd.DataFrame({'essay_name': essay_names, 'essay_text': chapter_texts, 'hyperlink': hyperlinks})
data.to_csv('het_essays.csv', index=False, header=True)

import os
import subprocess

# List files in the current directory (example of using os module)
files = os.listdir('.')
print(files)

# Run Git commands using subprocess
subprocess.run(["git", "add", "."])
subprocess.run(["git", "commit", "-m", "ran scraping"])
subprocess.run(["git", "push", "origin"])
