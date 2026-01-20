# Cluster 13

@requests_mock.Mocker(kw='mock')
def test_parse_results_amz(**kwargs):
    mock_file = open('tests/transfer/mocks/mock_parse_results_amz', 'rb')
    mock_body = mock_file.read()
    mock_file.close()
    mock_query = 'red basketball shoes'
    query = mock_query.replace(' ', '+')
    kwargs['mock'].get(f'https://www.amazon.com/s?k={query}&page=1', content=mock_body)
    output = parse_results_amz(mock_query, 1)
    expected = [{'Price': '59.49', 'Title': 'High Top Mens Basketball Shoes Lou Williams Streetball Master Breathable Non Slip Outdoor Sneakers Cushioning Workout Shoes for Fitness', 'asin': 'B083QCWF61'}, {'Price': '45.99', 'Title': 'Kids Basketball Shoes High-top Sports Shoes Sneakers Durable Lace-up Non-Slip Running Shoes Secure for Little Kids Big Kids and Boys Girls', 'asin': 'B08FWWWQ11'}, {'Price': '64.99', 'Title': 'Unisex-Adult Lockdown 5 Basketball Shoe', 'asin': 'B0817BFNC4'}, {'Price': '63.75', 'Title': 'Unisex-Child Team Hustle D 9 (Gs) Sneaker', 'asin': 'B07HHTS79M'}, {'Price': '74.64', 'Title': 'Unisex-Adult D.O.N. Issue 3 Basketball Shoe', 'asin': 'B08N8DQLS2'}, {'Price': '104.90', 'Title': "Men's Lebron Witness IV Basketball Shoes", 'asin': 'B07TKMMHVB'}, {'Price': '36.68', 'Title': "Unisex-Child Pre-School Jet '21 Basketball Shoe", 'asin': 'B08N6VRHV4'}, {'Price': '59.98', 'Title': "Men's Triple Basketball Shoe", 'asin': 'B08QCL8VKM'}, {'Price': '45.98', 'Title': 'Unisex-Child Pre School Lockdown 4 Basketball Shoe', 'asin': 'B07HKP12DH'}, {'Price': '143.72', 'Title': "Men's Basketball Shoes", 'asin': 'B07SNR7HRF'}]
    assert output == expected

def parse_results_amz(query, page_num=None, verbose=True):
    url = 'https://www.amazon.com/s?k=' + query.replace(' ', '+')
    if page_num is not None:
        url += '&page=' + str(page_num)
    if verbose:
        print(f'Search Results URL: {url}')
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    soup = BeautifulSoup(webpage.content, 'html.parser')
    products = soup.findAll('div', {'data-component-type': 's-search-result'})
    if products is None:
        temp = open(DEBUG_HTML, 'w')
        temp.write(str(soup))
        temp.close()
        raise Exception("Couldn't find search results page, outputted html for inspection")
    results = []
    for product in products[:NUM_PROD_LIMIT]:
        asin = product['data-asin']
        title = product.find('h2', {'class': 'a-size-mini'})
        price_div = product.find('div', {'class': 's-price-instructions-style'})
        price = price_div.find('span', {'class': 'a-offscreen'})
        result = {'asin': asin, 'Title': title.text.strip(), 'Price': price.text.strip().strip('$')}
        results.append(result)
    if verbose:
        print('Scraped', len(results), 'products')
    return results

