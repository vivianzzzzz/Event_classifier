# Duke Student Groups Event Data Documentation and Retrieval Process

## Data Source
- **Duke Student Group:** [Duke Groups](https://dukegroups.com/events)
- **Description:** Events posted by Duke student groups.
- **Timeline:** The earliest event data is from October 2019, with a total of 19,065 events (as of November 2023).
- **Note:** Unlike Duke Calendar, some event locations are private and require login to view, so private location data is not collected.

## Data Description

- **Event Types:**
  - we selecte the below five event types and hardcoded the type into integer
  <img width="243" alt="Screenshot 2023-12-06 at 2 34 41 AM" src="https://github.com/nogibjj/Event_classifier/assets/46847817/a4798088-214c-40a3-8767-19a0a5dd1d89">
  - Health/Wellness: 1
  - Social: 2
  - Workshop/Short Course: 3
  - Lecture/Talk: 4
  - Panel/Seminar/Colloquim: 5
 
- **Observation Numbers for Each Type:**
  - Health/Wellness: 1736
  - Social: 1937
  - Workshop/Short Course: 978
  - Lecture/Talk: 756
  - Panel/Seminar/Colloquim: 445

- **Final Data Columns:**
  - Title (str)
  - Details (str)
  - Title_details (str) - Concatenation of title and details
  - Type (name)
  - Type_id (int)
  - Host (str)
  - Date (str)
  - Time (str)
  - Location (str)
  - Link (str) - Source link to the event

  **Note:** Date, time, and location may require additional data cleaning for research purposes.

## Data Retrieval

### Step 1: Scrape Event IDs Based on Event Type
- Use `0_student_group_scrape.py` to scrape event IDs and store them in the `0_event_id` folder.

### Step 2: Data Cleaning
- Run `1_data_cleaning.py` to ensure events are not duplicated and generate a list of event URLs, stored in the `1_event_url` folder.

### Step 3: Scrape Event Information Based on Event List
- Utilize `2_event_info_scraper.py` to scrape data from the event list and store it in the `2_event_info` folder.

### Step 4: Merge into One Dataset
- Execute `3_data_merging` to merge data from the `2_event_info` folder, add a concatenated column, and create the final dataset.
