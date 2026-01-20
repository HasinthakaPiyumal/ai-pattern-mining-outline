# Cluster 11

@requests_mock.Mocker(kw='mock')
def test_parse_results_ebay(**kwargs):
    mock_file = open('tests/transfer/mocks/mock_parse_results_ebay', 'rb')
    mock_body = mock_file.read()
    mock_file.close()
    mock_query = 'red basketball shoes'
    query = mock_query.replace(' ', '+')
    kwargs['mock'].get(f'https://www.ebay.com/sch/i.html?_nkw={query}&_pgn=1', content=mock_body)
    output = parse_results_ebay(mock_query, 1)
    expected = [{'Price': ['100.00', '150.00'], 'Title': "Reebok Answer IV Men's Basketball Shoes", 'asin': '175065123030'}, {'Price': '$119.90', 'Title': "Air Jordan Stay Loyal Shoes Black Red White DB2884-001 Men's Multi Size NEW", 'asin': '265672133690'}, {'Price': '$100.00', 'Title': "Fila Men's Stackhouse Spaghetti Basketball Shoes Black Red White 1BM01788-113", 'asin': '175282509234'}, {'Price': ['61.99', '85.99'], 'Title': 'Puma Disc Rebirth 19481203 Mens Black Red Synthetic Athletic Basketball Shoes', 'asin': '313944854658'}, {'Price': '$0.01', 'Title': "Puma RS-Dreamer J. Cole Basketball Shoes Red 193990-16 Men's Size 10.0", 'asin': '403760625150'}, {'Price': '$45.00', 'Title': 'Nike Mens 9.5 PG 5  Maroon Red White Basketball Shoes Sneaker DM 5045–601￼ Flaw', 'asin': '115456853186'}, {'Price': ['114.90', '119.90'], 'Title': "Air Jordan Stay Loyal Shoes White Black Red DB2884-106 Men's Multi Size NEW", 'asin': '155046831159'}, {'Price': '$8.99', 'Title': "Harden Volume 3 Men's Basketball Shoes Size 9.5", 'asin': '175342407862'}, {'Price': '$59.97', 'Title': "Men's Nike Precision 5 Basketball Shoes Gym Red Black Grey Bred Multi Size NEW", 'asin': '134149634710'}]
    assert output == expected

def parse_results_ebay(query, page_num=None, verbose=True):
    query_string = '+'.join(query.split())
    page_num = 1 if page_num is None else page_num
    url = f'https://www.ebay.com/sch/i.html?_nkw={query_string}&_pgn={page_num}'
    if verbose:
        print(f'Search Results URL: {url}')
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    soup = BeautifulSoup(webpage.text, 'html.parser')
    products = soup.select('.s-item__wrapper.clearfix')
    results = []
    for item in products[:NUM_PROD_LIMIT]:
        title = item.select_one('.s-item__title').text.strip()
        if 'shop on ebay' in title.lower():
            continue
        link = item.select_one('.s-item__link')['href']
        asin = link.split('?')[0][len('https://www.ebay.com/itm/'):]
        try:
            price = item.select_one('.s-item__price').text
            if 'to' in price:
                prices = price.split(' to ')
                price = [p.strip('$') for p in prices]
        except:
            price = None
        results.append({'asin': asin, 'Title': title, 'Price': price})
    if verbose:
        print(f'Scraped {len(results)} products')
    return results

