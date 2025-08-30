import asyncio
import json
import schedule
import time
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin

URL = "https://pureportal.coventry.ac.uk/en/organisations/fbl-school-of-economics-finance-and-accounting/publications/"
USERAGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36 Edg/134.0.0.0'

async def scrape():
    async with async_playwright() as p:
        try:
            browser = await p.firefox.launch(headless=True)
            context = await browser.new_context(user_agent=USERAGENT)
            page = await context.new_page()

            all_publications_data = []
            current_page_url = URL
            page_number = 1

            while True:
                print(f"Navigating to page {page_number}: {current_page_url}")
                await page.goto(current_page_url, wait_until="domcontentloaded")
                await page.wait_for_selector('li.list-result-item', state='visible', timeout=30000)

                html_content = await page.content()
                doc = BeautifulSoup(html_content, "html.parser")

                page_links = [
                    a['href'] for a in doc.select('h3.title a[href^="https://pureportal.coventry.ac.uk/en/publications/"]')
                ]

                for link in page_links:
                    print(f"Scraping detail page: {link}")
                    await asyncio.sleep(2)
                    await page.goto(link, wait_until="domcontentloaded")
                    await page.wait_for_selector('main', state='visible', timeout=300000)

                    detail_html = await page.content()
                    detail_doc = BeautifulSoup(detail_html, "html.parser")

                    title = detail_doc.select_one('h1').get_text(strip=True) if detail_doc.select_one('h1') else 'N/A'
                    title_link = link
                    published_date = detail_doc.select_one('span.date').get_text(strip=True) if detail_doc.select_one('span.date') else 'N/A'
                    abstract = detail_doc.select_one('div.rendering_abstractportal div.textblock').get_text(strip=True) if detail_doc.select_one('div.rendering_abstractportal div.textblock') else 'N/A'

                    authors_data = []
                    author_container_with_link = detail_doc.select_one('p.relations.persons a')
                    if author_container_with_link:
                        author_name = author_container_with_link.get_text(strip=True)
                        author_link = author_container_with_link.get('href', 'N/A')
                        authors_data.append({'name': author_name, 'link': author_link})
                    else:
                        author_container_no_link = detail_doc.select_one('p.relations.persons')
                        if author_container_no_link:
                            author_names_text = author_container_no_link.get_text(strip=True)
                            author_names_list = author_names_text.replace(" and ", ",").split(',')
                            for name in [n.strip() for n in author_names_list]:
                                if name:
                                    authors_data.append({'name': name, 'link': 'N/A'})

                    keywords_section = detail_doc.find('h2', string='Keywords')
                    keywords = 'N/A'
                    if keywords_section and keywords_section.find_next_sibling('ul'):
                        keywords_list = [li.get_text(strip=True) for li in keywords_section.find_next_sibling('ul').find_all('li')]
                        keywords = keywords_list

                    subject_areas_section = detail_doc.find('h2', string='ASJC Scopus subject areas')
                    subject_areas = 'N/A'
                    if subject_areas_section and subject_areas_section.find_next_sibling('ul'):
                        subject_areas_list = [li.get_text(strip=True) for li in subject_areas_section.find_next_sibling('ul').find_all('li')]
                        subject_areas = subject_areas_list

                    publication_data = {
                        'title': title, 'title_link': title_link, 'published_date': published_date, 'abstract': abstract,
                        'authors': authors_data, 'keywords': keywords, 'subject_areas': subject_areas
                    }
                    all_publications_data.append(publication_data)

                next_button = doc.select_one('li.next a')
                if next_button and 'href' in next_button.attrs:
                    current_page_url = urljoin(URL, next_button['href'])
                    page_number += 1
                else:
                    print("No more pages found. Scraping complete.")
                    break

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if 'browser' in locals():
                await browser.close()

        with open('../data/publications_data.json', 'w') as f:
            json.dump(all_publications_data, f, indent=4)
        print("Data saved to publications_data.json")

def run_scrape():
    asyncio.run(scrape())

if __name__ == "__main__":
    schedule.every().day.at("02:00").do(run_scrape)
    print("Scheduler started. Press Ctrl+C to exit.")
    while True:
        schedule.run_pending()
        time.sleep(1)
