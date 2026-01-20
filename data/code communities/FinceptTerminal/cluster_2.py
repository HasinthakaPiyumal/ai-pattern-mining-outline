# Cluster 2

def scrape_forex_calendar(date_str='oct15.2025'):
    """
    Scrapes Forex Factory calendar for given date
    Args:
        date_str: Date in format like 'oct15.2025'
    Returns:
        JSON with structured data
    """
    url = f'https://www.forexfactory.com/calendar?day={date_str}'
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    driver = webdriver.Chrome(options=options)
    events = []
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'calendar__table')))
        time.sleep(2)
        rows = driver.find_elements(By.CSS_SELECTOR, 'tr.calendar__row')
        for row in rows:
            try:
                event = {}
                time_elem = row.find_elements(By.CLASS_NAME, 'calendar__time')
                event['time'] = time_elem[0].text if time_elem else ''
                currency_elem = row.find_elements(By.CLASS_NAME, 'calendar__currency')
                event['currency'] = currency_elem[0].text if currency_elem else ''
                impact_elem = row.find_elements(By.CLASS_NAME, 'calendar__impact')
                if impact_elem:
                    impact_spans = impact_elem[0].find_elements(By.TAG_NAME, 'span')
                    event['impact'] = len([s for s in impact_spans if 'icon--ff-impact-red' in s.get_attribute('class') or 'icon--ff-impact-ora' in s.get_attribute('class') or 'icon--ff-impact-yel' in s.get_attribute('class')])
                else:
                    event['impact'] = 0
                event_elem = row.find_elements(By.CLASS_NAME, 'calendar__event')
                event['event'] = event_elem[0].text.strip() if event_elem else ''
                actual_elem = row.find_elements(By.CLASS_NAME, 'calendar__actual')
                event['actual'] = actual_elem[0].text if actual_elem else ''
                forecast_elem = row.find_elements(By.CLASS_NAME, 'calendar__forecast')
                event['forecast'] = forecast_elem[0].text if forecast_elem else ''
                previous_elem = row.find_elements(By.CLASS_NAME, 'calendar__previous')
                event['previous'] = previous_elem[0].text if previous_elem else ''
                if event['event'] and event['event'] not in ['', 'All Day']:
                    events.append(event)
            except Exception as e:
                continue
    finally:
        driver.quit()
    return {'date': date_str, 'url': url, 'events_count': len(events), 'events': events}

