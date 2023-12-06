from typing import List
import sqlite3
import requests
from bs4 import BeautifulSoup
from event import Event


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# This file scrape data from duke calendar. But we decided only use event from duke student group. So the file is not used.


def scrape_data_selenium() -> List[Event]:
    # Set up a Selenium WebDriver
    driver = webdriver.Firefox()  # Or use another browser driver if you prefer

    # Navigate to the URL
    url = "https://calendar.duke.edu/index?prev_rows=40&rows=60&user_date=11%2F10%2F2018#current"
    driver.get(url)

    # Scroll down until no more new data is loaded
    while True:
        # Scroll down to the bottom of the page
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # Wait for any new data to load
        WebDriverWait(driver, 10).until_not(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".loading"))
        )
        # Check if there's still more data to load
        if not driver.find_element_by_css_selector(".load-more").is_displayed():
            break

    # Parse the page source
    document = BeautifulSoup(driver.page_source, "html.parser")

    # Extract the data
    html_product_selector = ".views-row"
    html_products = document.select(html_product_selector)

    # Store the extracted data
    for html_product in html_products:
        link = html_product.select("a").pop()["href"]
        # Extract other data as needed...

    # Close the browser
    driver.quit()


def scrape_data() -> List[Event]:
    # url = "https://math.duke.edu/events"
    """
        resp = requests.get(url)
    resp.raise_for_status()
    body = resp.text
    document = BeautifulSoup(body, "html.parser")
    with open("document.html", "w") as f:
        f.write(str(document))
    """
    url = "https://calendar.duke.edu/index?prev_rows=3000&rows=3020&user_date=11%2F10%2F2018#current"
    resp = requests.get(url)
    resp.raise_for_status()
    body = resp.text
    document = BeautifulSoup(body, "html.parser")
    with open("document.html", "w") as f:
        f.write(str(document))

    # print(document)
    exit()

    html_product_selector = ".views-row"
    html_products = document.select(html_product_selector)

    # initialize the list that will store the scraped data
    events: List[Event] = []

    for html_product in html_products:
        link = html_product.select("a").pop()["href"]
        title = (
            html_product.select(".views-field-field-display-title").pop().text.strip()
        )
        time = html_product.select(".views-field-field-event-date").pop().text.strip()
        location = (
            html_product.select(".views-field-field-event-location").pop().text.strip()
        )
        series = html_product.select(".event-series").pop().text.strip()
        speaker = (
            html_product.select(".views-field-field-event-speakers").pop().text.strip()
        )
        detail = html_product.select(".views-field-nothing").pop().text.strip()
        category = "Seminar"
        organizer = "Duke Mathematics Department"

        # instanciate a new event product
        # with the scraped data and add it to the list
        event = Event(
            title=title,
            organizer=organizer,
            category=category,
            time=time,
            location=location,
            link=link,
            series=series,
            speaker=speaker,
            detail=detail,
        )
        events.append(event)

    # print the list of products
    print(events)
    return events


def store_data(events: List[Event]):
    """connect to a sqlite database and write events in a event table"""
    conn = sqlite3.connect("events.db")
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            organizer TEXT NOT NULL,
            category TEXT NOT NULL,
            time TEXT NOT NULL,
            location TEXT NOT NULL,
            link TEXT NOT NULL UNIQUE,
            series TEXT,
            speaker TEXT,
            detail TEXT
        )"""
    )

    for event in events:
        c.execute(
            """INSERT INTO events (title, organizer, category, time, location, link, series, speaker, detail)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.title,
                event.organizer,
                event.category,
                event.time,
                event.location,
                event.link,
                event.series,
                event.speaker,
                event.detail,
            ),
        )
    conn.commit()
    conn.close()


def main():
    events = scrape_data()
    print(events)
    store_data(events)


if __name__ == "__main__":
    main()
