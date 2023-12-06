def compare_event_id():
    # After scraping the events with different types. Compare the event ids from the two files and see if there are any duplicates.
    with open("workshop_event_id.txt", "r") as f:
        ids = [line.strip() for line in f]
    with open("lecture_event_id.txt", "r") as f:
        ids_1 = [line.strip() for line in f]
    print(len(set(ids)))
    print(len(set(ids_1)))
    print(len(set(ids) & set(ids_1)))
    print(len(set(ids) | set(ids_1)))
    duplicates = set(ids) & set(ids_1)
    print("Duplicate Events:")
    for event_id in duplicates:
        print(event_id)


compare_event_id()


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


main_url = "https://dukegroups.com/events/rsvp?id="
input_file = "student_events_id_0.txt"
output_file = "student_events_id_0_filtered.txt"

remove_duplicates_and_summary(input_file, output_file, main_url)
