# Data acquisition

Examples of true human-generated speeches are needed to properly train this
project. To do so, we are primarily using
[American Rhetoric](https://www.americanrhetoric.com/)'s large speech bank.

The speech bank itself is avalable at: <https://www.americanrhetoric.com/speechbank.htm>

At the moment, we only attempt to scrape all speech transcripts already stored
directly on the American Rhetoric website. This is to avoid dealing with
multiple formats to scrape from, if possible.

## Web scraping

Currently, we have decided to use the [Web Scraper IO](https://webscraper.io/)
tool, which is available for free local use via a
[Chrome extension](https://chrome.google.com/webstore/detail/web-scraper/jnhgnonknehpejjnehehllkliplmbmhn?hl=en).

To learn more about Web Scraper IO, see the
[video tutorials](https://webscraper.io/tutorials) and
[documentation](https://webscraper.io/documentation).

Specifically, we have generated a
[sitemap for American Rhetoric](americanrhetoric-sitemap-1.json) that is intended
for use in the Web Scraper IO tool. Note that this is a work in progress, and
may need updates to function properly.

Note on the Web Scraper IO scrape parameters:

- **Request interval** sets how long the scraper will wait until it opens the
next page, after the moment it opened the current page.
Currently using the minimum of 2000 ms.
- **Page load delay** sets how long the scraper will wait to scrape the
currently opened page, from the moment it was opened.
Currently using the minimum of 500 ms.

## American Rhetoric output

The current [sitemap for American Rhetoric](americanrhetoric-sitemap-1.json)
produces multiple pieces of data, including:

- Speech name
- Speaker name
- Subtitle from American Rhetoric (may contain speech date)
- Speech text, in a JSON list. Each entry in the list corresponds to a
paragraph in the actual speech transcript.
**This will require post-processing to join the list, modify formatting symbols, etc.**
- Raw HTML of the speech page, for reference if needed.

