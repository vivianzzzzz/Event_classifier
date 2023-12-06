import requests
from bs4 import BeautifulSoup
import csv
import os


def scrape_event_info(url_file, type_name, type_id, output_file):
    event_info = []
    with open(url_file, "r") as f:
        urls = [line.strip() for line in f]

    for url in urls:
        try:
            response = requests.get(url)
        except requests.exceptions.RequestException:
            continue
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

            # Extracting event host
            event_org = soup.find("p", {"class": "rsvp__event-org"}).find("a").text

            # Extracting location
            location_text = soup.select_one(".col-md-5 p").text.strip()

            # Extracting date, time, and timezone
            target_div = soup.find("div", class_="col-md-4_5")
            date_text = target_div.find("p", style="margin:0;").text.strip()
            time_text = (
                target_div.find("p", style="margin:0;").find_next("p").text.strip()
            )
            timezone_text = target_div.find("span", id="timezone").text.strip()

            event_info.append(
                {
                    "title": name,
                    "details": event_details_text,
                    "type": type_name,
                    "type_id": type_id,
                    "host": event_org,
                    "date": date_text,
                    "time": time_text,
                    "location": location_text,
                    "link": url,
                }
            )
            print(url)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "title",
                "details",
                "type",
                "type_id",
                "host",
                "date",
                "time",
                "location",
                "link",
            ],
        )
        writer.writeheader()
        writer.writerows(event_info)

    return event_info


def main():
    folder_path = "1_event_url"
    output_folder_path = "2_event_info"

    filenames = sorted(os.listdir(folder_path))
    output_filenames = [
        filename.replace("_url.txt", "_info.csv") for filename in filenames
    ]
    url_file = [os.path.join(folder_path, filename) for filename in filenames]
    output_file = [
        os.path.join(output_folder_path, output_filename)
        for output_filename in output_filenames
    ]
    event_types = [
        "Health/Wellness",
        "Social",
        "Workshop/Short Course",
        "Lecture/Talk",
        "Panel/Seminar/Colloquim",
    ]
    for i in range(len(filenames)):
        scrape_event_info(url_file[i], event_types[i], i + 1, output_file[i])


if __name__ == "__main__":
    main()


# # Extracting event tags?
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

# registered = soup.find("span", {"class": "number"}).text

# target_div = soup.find("div", id="event_main_card")
