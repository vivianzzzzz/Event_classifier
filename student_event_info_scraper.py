import requests
from bs4 import BeautifulSoup
import csv


def scrape_event_info(url_file):
    event_info = []
    with open(url_file, "r") as f:
        urls = [line.strip() for line in f]
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        name = soup.find("h1", {"class": "rsvp__event-name"})
        if name is not None and name.text:
            name = name.text.strip()

            event_details_element = soup.find("div", {"id": "event_details"})
            elements_to_exclude = event_details_element.find_all(["h2", "a"])
            for element in elements_to_exclude:
                element.decompose()
            event_details_text = event_details_element.get_text(strip=True)

            event_info.append({"name": name, "details": event_details_text})

    with open("event_info.csv", "w", neline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "details"])
        writer.writeheader()
        writer.writerows(event_info)

    return event_info


def remove_duplicates_and_summary(input_file, output_file, main_url):
    unique_lines = set()
    suburls = []
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if (
                line not in unique_lines
                and not line.startswith("Summary")
                and not line.startswith("[eventId]")
            ):
                url = main_url + line
                f_out.write(url + "\n")
                unique_lines.add(line)
                # suburl = line.strip()


def main():
    """
    main_url = "https://dukegroups.com/events/rsvp?id="
    input_file = "student_events_id_0.txt"
    output_file = "student_events_id_0_filtered.txt"

    remove_duplicates_and_summary(input_file, output_file, main_url)
    """
    output_file = "student_events_id_0_filtered.txt"
    event_info = scrape_event_info(output_file)
    # event_info_str = "\n".join(event_info)
    # with open(input_file, "w") as f:
    #     f.write(event_info_str)


if __name__ == "__main__":
    main()
