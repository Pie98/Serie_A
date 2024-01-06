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


#############################################################

# ------------------- save_page_html -----------------------#

#############################################################


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


#############################################################

# --------------------- filter_words -----------------------#

#############################################################
    

#Filtering the parts of the html file that we are interested in 
def filter_words(input_file, output_file, start_match, end_match, other_matches):
    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read()

    start_index = content.find(start_match)
    end_index = content.find(end_match)

    # verifies that the words are in the file
    if start_index == -1 or end_index == -1:
        raise ValueError("Le parole di inizio o fine non sono presenti nel file.")

    # filters the html file from start_match to end_match included 
    if other_matches:
        filtered_content = content[start_index:end_index]
    else:
        filtered_content = content[start_index:end_index + 21644]

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(filtered_content)

# Finding the odds in my html file
def find_numbers_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # define pattern regex 
    pattern = re.compile(r'\b\d{1,2},\d{1,2}\b|\s\s\d{1,2}\s')

    # find patterns
    matches = pattern.findall(content)
    matches = [s.replace(" ", "") for s in matches]

    return matches


#############################################################

# --------------------- refresh_odds -----------------------#

#############################################################


def refresh_odds(start_filter, end_filter, num_matches, other_matches, prima_iterazione):
    # saving the previous odds in a csv file
    column_types = {'1': float, 'x': float, '2': float}
    if (prima_iterazione==False):
        pd.read_csv(r'Refreshing_odds/last_odds.csv', dtype=column_types).to_csv('Refreshing_odds/previous_odds.csv',index=False)

    # Saving the html file
    website_url = "https://www.*******.it/sport/CALCIO/SERIE%20A"
    output_path = "Refreshing_odds/last_odds.html"
    save_page_html(website_url, output_path)

    # filtering the part of the html file i need
    input_file_path = "Refreshing_odds/last_odds.html"

    filter_words(input_file_path, output_path, start_filter, end_filter, other_matches)


    # finding the odds i need
    result = find_numbers_in_file(output_path)
    print(result)
    if len(result) != num_matches * 7:
        raise ValueError(' WARNING, NOT 10 MATCHES SELECTED ')

    #saving the odds in a csv 
    with open('Refreshing_odds/last_odds.csv', mode='w', newline='', encoding='utf-8',) as file:
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
    if (prima_iterazione==True):
        pd.read_csv(r'Refreshing_odds/last_odds.csv', dtype=column_types).to_csv('Refreshing_odds/previous_odds.csv',index=False)