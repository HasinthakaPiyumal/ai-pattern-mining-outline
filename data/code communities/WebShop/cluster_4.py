# Cluster 4

@requests_mock.Mocker(kw='mock')
def test_parse_item_page_ws(**kwargs):
    mock_file = open('tests/transfer/mocks/mock_parse_item_page_ws', 'rb')
    mock_body = mock_file.read()
    mock_file.close()
    mock_desc_file = open('tests/transfer/mocks/mock_parse_item_page_ws_desc', 'rb')
    mock_desc_body = mock_desc_file.read()
    mock_desc_file.close()
    mock_feat_file = open('tests/transfer/mocks/mock_parse_item_page_ws_feat', 'rb')
    mock_feat_body = mock_feat_file.read()
    mock_feat_file.close()
    mock_asin = 'B09P87V3LZ'
    mock_query = 'red basketball shoes'
    mock_options = {}
    query_str = '+'.join(mock_query.split())
    options_str = json.dumps(mock_options)
    url = f'{WEBSHOP_URL}/item_page/{WEBSHOP_SESSION}/{mock_asin}/{query_str}/1/{options_str}'
    url_desc = f'{WEBSHOP_URL}/item_sub_page/{WEBSHOP_SESSION}/{mock_asin}/{query_str}/1/Description/{options_str}'
    url_feat = f'{WEBSHOP_URL}/item_sub_page/{WEBSHOP_SESSION}/{mock_asin}/{query_str}/1/Features/{options_str}'
    print(f'Item Page URL: {url}')
    print(f'Item Description URL: {url_desc}')
    print(f'Item Features URL: {url_feat}')
    kwargs['mock'].get(url, content=mock_body)
    kwargs['mock'].get(url_desc, content=mock_desc_body)
    kwargs['mock'].get(url_feat, content=mock_feat_body)
    output = parse_item_page_ws(mock_asin, mock_query, 1, mock_options)
    expected = {'MainImage': 'https://m.media-amazon.com/images/I/51ltvkzGhGL.jpg', 'Price': '100.0', 'Rating': 'N.A.', 'Title': 'PMUYBHF Womens Fashion Flat Shoes Comfortable Running Shoes Sneakers Tennis Athletic Shoe Casual Walking Shoes', 'asin': mock_asin, 'option_to_image': {'6.5': 'http://3.83.245.205:3000/item_page/abc/B09P87V3LZ/%5B%27red%27%2C%20%27basketball%27%2C%20%27shoes%27%5D/1/%7B%27size%27:%20%276.5%27%7D', '7.5': 'http://3.83.245.205:3000/item_page/abc/B09P87V3LZ/%5B%27red%27%2C%20%27basketball%27%2C%20%27shoes%27%5D/1/%7B%27size%27:%20%277.5%27%7D', '8': 'http://3.83.245.205:3000/item_page/abc/B09P87V3LZ/%5B%27red%27%2C%20%27basketball%27%2C%20%27shoes%27%5D/1/%7B%27size%27:%20%278%27%7D', '8.5': 'http://3.83.245.205:3000/item_page/abc/B09P87V3LZ/%5B%27red%27%2C%20%27basketball%27%2C%20%27shoes%27%5D/1/%7B%27size%27:%20%278.5%27%7D', '9': 'http://3.83.245.205:3000/item_page/abc/B09P87V3LZ/%5B%27red%27%2C%20%27basketball%27%2C%20%27shoes%27%5D/1/%7B%27size%27:%20%279%27%7D', 'black': 'http://3.83.245.205:3000/item_page/abc/B09P87V3LZ/%5B%27red%27%2C%20%27basketball%27%2C%20%27shoes%27%5D/1/%7B%27color%27:%20%27black%27%7D', 'purple': 'http://3.83.245.205:3000/item_page/abc/B09P87V3LZ/%5B%27red%27%2C%20%27basketball%27%2C%20%27shoes%27%5D/1/%7B%27color%27:%20%27purple%27%7D', 'red': 'http://3.83.245.205:3000/item_page/abc/B09P87V3LZ/%5B%27red%27%2C%20%27basketball%27%2C%20%27shoes%27%5D/1/%7B%27color%27:%20%27red%27%7D'}, 'options': {'color': ['black', 'purple', 'red'], 'size': ['6.5', '7.5', '8', '8.5', '9']}, 'BulletPoints': 'Pure Running Shoe\nComfort Flat Sneakers\n[FEATURES]: Soles with unique non-slip pattern, it has great abrasion resistant and provide protection when you walking or running. (Pure Running Shoe Mesh Walking Shoes Fashion Sneakers Slip On Sneakers Wedge Platform Loafers Modern Walking Shoes Sock Sneakers Platform Loafers Shoes Non Slip Running Shoes Athletic Tennis Shoes Blade Type Sneakers Lace-up Sneaker) sole\n[WIDE ANKLE DESIGN]: Perfect accord with human body engineering, green, healthy concept design make the walking shoes wear more comfortable, wide width wlking shoes. (Low Top Walking Shoes Fashion Canvas Sneakers Slip On Shoes Casual Walking Shoes Hidden Wedge Sneaker Low Top Canvas Sneakers Lace-up Classic Casual Shoes Walking Tennis Shoes Lightweight Casual Sneakers Slip on Sock Sneakers Air Cushion Platform Loafers Slip-On Mule Sneaker )\n[CUSHION WITH ARCH SUPPORT]: Gives you a comfort for all day long. Wear these lightweight walking shoes, let every step of moving on a comfortable feeling. (Fashion Casual Shoes Athletic Workout Shoes Fitness Sneaker Athletic Running Shoes Air Cushion Sneakers Stylish Athletic Shoes Lace Up Canvas Shoes Slip on Walking Shoe Fashion Sneakers Low Top Classic Sneakers Comfort Fall Shoes Memory Foam Slip On Sneakers Air Cushion Sneakers Running Walking Shoes)\n[NON-SLIP SOLE]: Made from ultra soft and lightweight RUBBER material,with the function of shock absorbing and cushioning,offering the best durability and traction. (Wedge Sneakers Walking Tennis Shoes Slip On Running Shoes Lightweight Fashion Sneakers Fashion Travel Shoes Walking Running Shoes Non Slip Running Shoes Athletic Tennis Sneakers Sports Walking Shoes Platform Fashion Sneaker Memory Foam Tennis Sneakers Running Jogging Shoes Sock Sneakers Canvas Fashion Sneakers)\n[OCCASIONS]: Ultra lightweight design provides actual feelings of being barefooted and like walking on the feather, perfect for walking, hiking, bike riding, working, shopping, indoor, outdoor, casual, sports, travel, exercise, vacation, and etc. (Flat Fashion Sneakers Lightweight Walking Sneakers Platform Loafers Sport Running Shoes Casual Flat Loafers Slip-On Sneaker Casual Walking Shoes High Top Canvas Sneakers Lace up Sneakers Workout Walking Shoes Tennis Fitness Sneaker)\n[Customers Are Our Priority]: We follow the principle of customer first, so if you encounter any problems after buying shoes, we will try our best to solve them for you. (Breathable Air Cushion Sneakers Walking Tennis Shoes Air Athletic Running Shoes Air Cushion Shoes Mesh Sneakers Fashion Tennis Shoes Jogging Walking Sneakers Breathable Casual Sneakers Fashion Walking Shoes Athletic Running Sneakers Walking Work Shoes Air Running Shoes Slip on Sneakers Mesh Walking Shoes)', 'Description': 'Here Are The Things You Want To Knowa─=≡Σ(((つ̀ώ)つSTORE INTRODUCTION:>>>>Our store helps our customers improve their quality of life~As a distributor, we value quality and service.Focus on the high quality and durability of the product.Committed to creating a store that satisfies and reassures our customers.TIPS:>>>>1. Please allow minor errors in the data due to manual measurements.2. Due to the color settings of the display, the actual color may be slightly different from the online image.QUALITY PROMISE:>>>>Our goal is to continuously provide a range of quality products.We place a huge emphasis on the values of quality and reliability.We have always insisted on fulfilling this commitment.In short, we want our customers to have the same great product experience every time and be trusted to deliver on this commitment.Please give us a chance to serve you.OTHER:>>>>athletic sneaker laces athletic sneakers white athletic sneakers for women clearance leather Sneaker leather sneakers women leather sneakers for menleather sneaker laces leather sneaker platform basketball shoes basketball shoes for men basketball shoe laces basketball shoe grip basketball shoes for women fitness shoes for men fitness shoes women workout fitness shoes women fitness shoes women size 5 fitness shoes men workout fitness shoes for men high top sneakers for women walking shoes sneakers with arch support for women'}
    assert output == expected

def parse_item_page_ws(asin, query, page_num, options, verbose=True):
    product_dict = {}
    product_dict['asin'] = asin
    query_string = '+'.join(query.split())
    options_string = json.dumps(options)
    url = f'{WEBSHOP_URL}/item_page/{WEBSHOP_SESSION}/{asin}/{query_string}/{page_num}/{options_string}'
    if verbose:
        print(f'Item Page URL: {url}')
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    soup = BeautifulSoup(webpage.content, 'html.parser')
    product_dict['Title'] = soup.find('h2').text
    h4_headers = soup.findAll('h4')
    for header in h4_headers:
        text = header.text
        if 'Price' in text:
            product_dict['Price'] = text.split(':')[1].strip().strip('$')
        elif 'Rating' in text:
            product_dict['Rating'] = text.split(':')[1].strip()
    product_dict['MainImage'] = soup.find('img')['src']
    options, options_to_image = ({}, {})
    option_blocks = soup.findAll('div', {'class': 'radio-toolbar'})
    for block in option_blocks:
        name = block.find('input')['name']
        labels = block.findAll('label')
        inputs = block.findAll('input')
        opt_list = []
        for label, input in zip(labels, inputs):
            opt = label.text
            opt_img_path = input['onclick'].split('href=')[1].strip("';")
            opt_img_url = f'{WEBSHOP_URL}{opt_img_path}'
            opt_list.append(opt)
            options_to_image[opt] = opt_img_url
        options[name] = opt_list
    product_dict['options'] = options
    product_dict['option_to_image'] = options_to_image
    url = f'{WEBSHOP_URL}/item_sub_page/{WEBSHOP_SESSION}/{asin}/{query_string}/{page_num}/Description/{options_string}'
    if verbose:
        print(f'Item Description URL: {url}')
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    soup = BeautifulSoup(webpage.content, 'html.parser')
    product_dict['Description'] = soup.find(name='p', attrs={'class': 'product-info'}).text.strip()
    url = f'{WEBSHOP_URL}/item_sub_page/{WEBSHOP_SESSION}/{asin}/{query_string}/{page_num}/Features/{options_string}'
    if verbose:
        print(f'Item Features URL: {url}')
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    soup = BeautifulSoup(webpage.content, 'html.parser')
    bullets = soup.find(name='ul').findAll(name='li')
    product_dict['BulletPoints'] = '\n'.join([b.text.strip() for b in bullets])
    return product_dict

