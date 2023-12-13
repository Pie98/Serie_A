import time
import pandas as pd
import re
import csv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def save_page_html(url, save_path):
    chrome_options = Options()
    chrome_options.add_argument("--incognito")

    driver = webdriver.Chrome(options=chrome_options)

    driver.get(url)

    # Handle cookies if there is a popup
    try:
        # Wait for the cookie popup to appear (adjust the timeout as needed)
        cookie_popup = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "cookie-popup"))
        )

        # Click the "X" button to close the cookie popup
        close_button = cookie_popup.find_element(By.XPATH, "//button[contains(@class, 'close-button')]")
        close_button.click()
    except Exception as e:
        print(f"Cookie popup not found or encountered an error: {e}")

    # Wait for some time after closing the cookie popup (adjust as needed)
    time.sleep(5)

    # Get the HTML of the page
    page_html = driver.page_source

    # Save the HTML to a file
    with open(save_path, "w", encoding="utf-8") as file:
        file.write(page_html)

    print(f"HTML saved to {save_path}")

    # Close the browser window
    driver.quit()


#Filtering the parts of the html file that we are interested in 
def filter_words(input_file, output_file, start_date, end_date):
    with open(input_file, 'r', encoding='utf-8') as infile:
        words = infile.read().split()

    start_index = next((i for i, word in enumerate(words) if word == start_date), None)
    end_index = next((i for i, word in enumerate(words) if word == end_date), None)

    # Verifica se le parole '15/12' e '18/12' sono presenti nel file
    if start_index is None or end_index is None:
        raise ValueError("Le date di inizio o fine non sono presenti nel file.")

    # Filtra le parole tra '15/12' e '18/12' inclusi
    filtered_words = words[start_index:end_index + 20644]

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(' '.join(filtered_words))

def find_numbers_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Definisci il pattern regex per trovare le stringhe '\d\d,\d\d' o '\d,\d\d'
    pattern = re.compile(r'\b\d{1,2},\d{1,2}\b')

    # Trova tutte le corrispondenze nel testo
    matches = pattern.findall(content)

    return matches


### MAIN

def refresh_odds(start_date_filter, end_date_filter, num_matches):
    # saving the previous odds in a csv file
    column_types = {'1': float, 'x': float, '2': float}
    pd.read_csv(r'Data_scraping/last_odds.csv', dtype=column_types).to_csv('Data_scraping/min_odds.csv',index=False)

    # Saving the html file
    website_url = "https://www.snai.it/sport/CALCIO/SERIE%20A"
    output_path = "Data_scraping/last_odds.html"
    save_page_html(website_url, output_path)

    # filtering the part of the html file i need
    input_file_path = "Data_scraping/last_odds.html"

    filter_words(input_file_path, output_path, start_date_filter, end_date_filter)


    # finding the odds i need
    result = find_numbers_in_file(output_path)

    if len(result) != num_matches * 7:
        raise ValueError(' WARNING, NOT 10 MATCHES SELECTED ')

    #saving the odds in a csv 
    with open('Data_scraping/last_odds.csv', mode='w', newline='', encoding='utf-8',) as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['1','x','2'])
        reset=0
        temp=[]
        for numero in result:
            temp.append(numero.replace(',', '.'))
            reset=reset+1
            if (reset % 7 ==0):
                csv_writer.writerow(temp[::-1][:3])
                print(temp)
                temp=[]
                reset=0