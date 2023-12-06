import requests
from bs4 import BeautifulSoup
import csv
import re


def scrape_event_info(url_file):
    event_info = []
    with open(url_file, "r") as f:
        urls = [line.strip() for line in f]
    i = 0
    while i < 50:
        for url in urls:
            i = i + 1
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            name = soup.find("h1", {"class": "rsvp__event-name"})
            if name is not None and name.text:
                # Extracting event name
                name = name.text.strip()

                # Extracting event details
                event_details_element = soup.find("div", {"id": "event_details"})
                elements_to_exclude = event_details_element.find_all(["h2", "a"])
                for element in elements_to_exclude:
                    element.decompose()
                event_details_text = event_details_element.get_text(strip=True)
                # event_tags = [
                #     span.text.strip()
                #     for span in soup.select("span.label.label-default.label-tag > span")
                # ]

                # event_tags = [
                #     span.text.strip()
                #     for span in soup.find_all(
                #         "span", {"class": "label label-default label-tag"}
                #     )
                # ]
                # event_tags = [
                #     span.find("span").text.strip()
                #     for span in soup.find_all(
                #         "span", {"class": "label label-default label-tag"}
                #     )
                # ]
                event_org = soup.find("p", {"class": "rsvp__event-org"}).find("a").text
                # registered = soup.find("span", {"class": "number"}).text

                # target_div = soup.find("div", id="event_main_card")

                # Extracting location
                location_text = soup.select_one(".col-md-5 p").text.strip()

                # Extracting date, time, and timezone
                target_div = soup.find("div", class_="col-md-4_5")
                date_text = target_div.find("p", style="margin:0;").text.strip()
                time_text = (
                    target_div.find("p", style="margin:0;").find_next("p").text.strip()
                )
                timezone_text = target_div.find("span", id="timezone").text.strip()

                print(location_text)
                print(time_text)
                print(timezone_text)

                event_info.append(
                    {
                        "name": name,
                        "details": event_details_text,
                        "location": location_text,
                        "time": time_text,
                    }
                )

    with open("event_info.csv", "w", neline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "details"])
        writer.writeheader()
        writer.writerows(event_info)

    return event_info


def main():
    output_file = "student_events_id_0_filtered.txt"
    event_info = scrape_event_info(output_file)
    # event_info_str = "\n".join(event_info)
    # with open(input_file, "w") as f:
    #     f.write(event_info_str)


if __name__ == "__main__":
    main()
