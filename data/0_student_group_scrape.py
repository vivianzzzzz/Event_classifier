from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import datetime

# This file scrape data from duke student group. It generates files with all the event ids. in the 0_event_id folder.
# The files are used in event_info_scraper.py to scrape event information.


def scrape_past_data():
    # The earliest date of the event is in October 2019
    start_date = datetime.date(2019, 10, 1)
    end_date = datetime.date(2023, 12, 6)

    # Loop through the dates in intervals
    delta = datetime.timedelta(days=200)
    while start_date <= end_date:
        current_end_date = start_date + delta

        # Construct the URL for the current date range
        # 7334 Health & Wellness
        # 7338 Social
        # 7335 Workshops
        # 7331 Lectures
        # 7336 Panel
        url = f"https://dukegroups.com/events?show=past&event_type=7336&from_date={start_date.strftime('%d+%b+%Y')}&to_date={current_end_date.strftime('%d+%b+%Y')}"

        # Set up a Selenium WebDriver and Navigate to the URL
        driver = webdriver.Firefox()
        driver.get(url)

        # Scroll down until no more new data is loaded
        while True:
            # Scroll down to the bottom of the page
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            # Wait for any new data to load
            WebDriverWait(driver, 20).until(
                EC.invisibility_of_element_located((By.CSS_SELECTOR, "#LoadingEvents"))
            )
            # Wait for the load button to show up
            time.sleep(10)
            # Check if there's still more data to load
            if not driver.find_element_by_css_selector(
                "button#lnkLoadMore"
            ).is_displayed():
                break

        # Parse the page source
        document = BeautifulSoup(driver.page_source, "html.parser")

        # Get the event Ids
        ids = [
            li["id"].replace("event_", "")
            for li in document.find_all("li", {"class": "list-group-item"})
            if "id" in li.attrs
        ]

        # Close the browser
        time.sleep(5)
        driver.quit()

        # Specify the file path to store the event Ids
        file_path = "student_events_id.txt"

        # Open the file in append mode and write each event Id on a new line
        with open(file_path, "a") as file:
            file.write(
                f"Summary: Events from {start_date} to {current_end_date}, total number {len(ids)} \n"
            )
            file.write(url + "\n")
            for item in ids:
                file.write(item + "\n")

        # Update Start Date
        start_date = current_end_date + datetime.timedelta(days=1)
    exit()


def main():
    scrape_past_data()


if __name__ == "__main__":
    main()
