# Cluster 15

@requests_mock.Mocker(kw='mock')
def test_parse_results_ws(**kwargs):
    mock_file = open('tests/transfer/mocks/mock_parse_results_ws', 'rb')
    mock_body = mock_file.read()
    mock_file.close()
    mock_query = 'red basketball shoes'
    query_str = mock_query.replace(' ', '+')
    url = f'{WEBSHOP_URL}/search_results/{WEBSHOP_SESSION}/{query_str}/1'
    kwargs['mock'].get(url, content=mock_body)
    output = parse_results_ws(mock_query, 1)
    expected = [{'Price': [24.49, 39.99], 'Title': "BinGoDug Men's Basketball Shoes, Men's Fashion Sneakers, Air Basketball Shoes for Men, Womens Basketball Shoes, Mens Basketball Shoes, Boys Basketball Shoes, Youth Basketball Shoes Men Women", 'asin': 'B09GKFNQWT'}, {'Price': [1.89, 7.58], 'Title': "RQWEIN Comfortable Mesh Sneakers Men's Roading Running Shoes Tennis Shoes Casual Fashion Sneakers Outdoor Non Slip Gym Athletic Sport Shoes", 'asin': 'B09BFY2R3R'}, {'Price': 100.0, 'Title': 'PMUYBHF Womens Fashion Flat Shoes Comfortable Running Shoes Sneakers Tennis Athletic Shoe Casual Walking Shoes', 'asin': 'B09P87V3LZ'}, {'Price': 100.0, 'Title': 'PMUYBHF Fashion Travel Shoes Jogging Walking Sneakers Air Cushion Platform Loafers Air Cushion Mesh Shoes Walking Dance Shoes', 'asin': 'B09N6SNKC1'}, {'Price': 100.0, 'Title': "PMUYBHF Women's Ballet Flats Walking Flats Shoes Dressy Work Low Wedge Arch Suport Flats Shoes Slip On Dress Shoes", 'asin': 'B09N6X5S74'}, {'Price': 100.0, 'Title': "PWKSELW High-top Men's Basketball Shoes Outdoor Sports Shoes Cushioning Training Shoes Casual Running Shoes", 'asin': 'B09MDB9V5W'}, {'Price': 100.0, 'Title': "Women's Flat Shoes Classic Round Toe Slip Office Black Ballet Flats Walking Flats Shoes Casual Ballet Flats", 'asin': 'B09N6PDFRF'}, {'Price': 100.0, 'Title': "Women's Mid-Calf Boots Wide Calf Boots for Women Fashion Zipper Womens Shoes Pu Leather Casual Boots Womens Slip-On Womens Flat Shoes Med Heel Womens' Boots Winter Snow Boot Comfy Boots(,5.5)", 'asin': 'B09N8ZHFNM'}, {'Price': 100.0, 'Title': 'PMUYBHF Womens Leisure Fitness Running Sport Warm Sneakers Shoes Slip-On Mule Sneakers Womens Mules', 'asin': 'B09P87DWGR'}, {'Price': 100.0, 'Title': 'Men Dress Shoes Leather Modern Classic Business Shoes Lace Up Classic Office Shoes Business Formal Shoes for Men', 'asin': 'B09R9MMTKR'}]
    assert output == expected

def parse_results_ws(query, page_num=None, verbose=True):
    query_string = '+'.join(query.split())
    page_num = 1 if page_num is None else page_num
    url = f'{WEBSHOP_URL}/search_results/{WEBSHOP_SESSION}/{query_string}/{page_num}'
    if verbose:
        print(f'Search Results URL: {url}')
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    soup = BeautifulSoup(webpage.content, 'html.parser')
    products = soup.findAll('div', {'class': 'list-group-item'})
    results = []
    for product in products:
        asin = product.find('a', {'class': 'product-link'})
        title = product.find('h4', {'class': 'product-title'})
        price = product.find('h5', {'class': 'product-price'})
        if '\n' in title:
            title = title.text.split('\n')[0].strip()
        else:
            title = title.text.strip().strip('\n')
        if 'to' in price.text:
            prices = price.text.split(' to ')
            price = [float(p.strip().strip('\n$')) for p in prices]
        else:
            price = float(price.text.strip().strip('\n$'))
        results.append({'asin': asin.text, 'Title': title, 'Price': price})
    if verbose:
        print(f'Scraped {len(results)} products')
    return results

