import requests
from bs4 import BeautifulSoup
import re
import random
import csv

url_backend = "https://www.booking.com/reviewlist.ar.html?aid=304142&label=gen173nr-1FCAEoggI46AdIM1gEaMQBiAEBmAEBuAEHyAEM2AEB6AEB-AEMiAIBqAIDuAKik8WtBsACAdICJDBmMzFiMDM2LTk3ODYtNGJjNi04YWVkLThlNGEyYTUzMDRiMdgCBuACAQ&sid=fd9339f48eb8b6dd864393b84a3de966&cc1=sa&dist=1&pagename=crowne-plaza-riyadh-rdc-convention&srpvid=2eeb7b5f2ee3028c&type=total&rows=10&offset="

original_header = {'User-Agent': 'MyApp/(put the new value here).0 Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36'}

headers_list = []

num_headers = 15

# generate headers with the new structure
for i in range(1, num_headers + 1):
    new_header = original_header.copy()
    new_header['User-Agent'] = new_header['User-Agent'].replace('(put the new value here)', str(i))
    headers_list.append(new_header)

random.shuffle(headers_list)

for i, header in enumerate(headers_list, 1):
    print(f"Header: {header}")


review_counter = 0  # global variable for the number of reviews

def get_random_header():

    return random.choice(headers_list)

def check_language_if_english(text, threshold=0.2):

    # calc the ratio of English characters to total characters
    english_chars = sum(char.isascii() for char in text)   # calc the ratio of English characters
    total_chars = len(text)  # calc length
    ratio = english_chars / total_chars   # calc english characters to total characters in the text

    # check if the ratio exceeds the threshold
    return ratio > threshold

def scrape_booking_reviews(current_url):
    global review_counter  # access the global variable

    response = requests.get(current_url, headers=get_random_header())


    if response.status_code == 200:  # if is it 200(success) or 403(fail)
        # parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        page_elements = soup.find_all("div", class_="bui-pagination__item")

        num_pages = len(page_elements)

        for i in range(num_pages):
            url = current_url + str(i * 10)

            response = requests.get(url, headers=get_random_header())

            if response.status_code == 200:

                soup = BeautifulSoup(response.content, 'html.parser')

                review_div_tags = soup.find_all("div", class_="bui-grid__column-11")

                # loop through each review div
                for index, review_div_tag in enumerate(review_div_tags, 1):
                    review_counter += 1

                    # this review_title_tag var will store the whole tag (<h3>..(data)..</h3>)
                    review_title_tag = review_div_tag.find("h3", class_="c-review-block__title c-review__title--rtl")


                    # check if the tag was found
                    if review_title_tag:
                        # this review_text var will extract the review text only in text format and then store it
                        review_text = review_title_tag.get_text(strip=True)

                        if not check_language_if_english(review_text):
                            print(f"Review {review_counter}: {review_text}")
                    else:
                        break
            else:
                break

def main():
    print("----------------------------------------------------\n")
    input_city = input("Enter the name of city in arabic: ")
    print("----------------------------------------------------\n")

    city_hotels_url = f"https://www.booking.com/searchresults.ar.html?ss={input_city}"

    print(city_hotels_url)
    print("----------------------------------------------------\n")

    response = requests.get(city_hotels_url, headers=get_random_header())

    if response.status_code == 200:

        soup = BeautifulSoup(response.content, 'html.parser')

        hotels = soup.find_all("div", class_="d6767e681c")

        for hotel in hotels:
            link = hotel.find("a").get("href")
            pattern = r'hotel/sa/(.*?).ar.html?'
            match = re.search(pattern, link)
            new_url = re.sub(r'(pagename=).*?(&)', fr'\1{match.group(1)}\2', url_backend)
            scrape_booking_reviews(new_url)

if __name__ == "__main__":
    main()
