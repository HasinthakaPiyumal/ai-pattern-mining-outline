# Cluster 7

@requests_mock.Mocker(kw='mock')
def test_parse_item_page_amz(**kwargs):
    mock_file = open('tests/transfer/mocks/mock_parse_item_page_amz', 'rb')
    mock_body = mock_file.read()
    mock_file.close()
    mock_asin = 'B073WRF565'
    kwargs['mock'].get(f'https://www.amazon.com/dp/{mock_asin}', content=mock_body)
    output = parse_item_page_amz(mock_asin)
    expected = {'asin': 'B073WRF565', 'Title': 'Amazon Basics Foldable 14" Black Metal Platform Bed Frame with Tool-Free Assembly No Box Spring Needed - Full', 'Price': 'N/A', 'Rating': '4.8 out of 5 stars', 'BulletPoints': ' \n About this item    Product dimensions: 75" L x 54" W x 14" H | Weight: 41.4 pounds    Designed for sleepers up to 250 pounds    Full size platform bed frame offers a quiet, noise-free, supportive foundation for a mattress. No box spring needed    Folding mechanism makes the frame easy to store and move in tight spaces    Provides extra under-the-bed storage space with a vertical clearance of about 13 inches    \n › See more product details ', 'Description': 'Amazon Basics Foldable, 14" Black Metal Platform Bed Frame with Tool-Free Assembly, No Box Spring Needed - Full   Amazon Basics', 'MainImage': 'https://images-na.ssl-images-amazon.com/images/I/41WIGwt-asL.__AC_SY300_SX300_QL70_FMwebp_.jpg', 'options': {'size': ['Twin', 'Full', 'Queen', 'King'], 'style': ['14-Inch', '18-Inch']}, 'option_to_image': {}}
    assert output == expected

def parse_item_page_amz(asin, verbose=True):
    product_dict = {}
    product_dict['asin'] = asin
    url = f'https://www.amazon.com/dp/{asin}'
    if verbose:
        print('Item Page URL:', url)
    begin = time.time()
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    end = time.time()
    if verbose:
        print(f'Item page scraping took {end - begin} seconds')
    soup = BeautifulSoup(webpage.content, 'html.parser')
    try:
        title = soup.find('span', attrs={'id': 'productTitle'})
        title = title.string.strip().replace(',', '')
    except AttributeError:
        title = 'N/A'
    product_dict['Title'] = title
    try:
        parent_price_span = soup.find(name='span', class_='apexPriceToPay')
        price_span = parent_price_span.find(name='span', class_='a-offscreen')
        price = float(price_span.getText().replace('$', ''))
    except AttributeError:
        price = 'N/A'
    product_dict['Price'] = price
    try:
        rating = soup.find(name='span', attrs={'id': 'acrPopover'})
        if rating is None:
            rating = 'N/A'
        else:
            rating = rating.text
    except AttributeError:
        rating = 'N/A'
    product_dict['Rating'] = rating.strip('\n').strip()
    try:
        features = soup.find(name='div', attrs={'id': 'feature-bullets'}).text
    except AttributeError:
        features = 'N/A'
    product_dict['BulletPoints'] = features
    try:
        desc_body = soup.find(name='div', attrs={'id': 'productDescription_feature_div'})
        desc_div = desc_body.find(name='div', attrs={'id': 'productDescription'})
        desc_ps = desc_div.findAll(name='p')
        desc = ' '.join([p.text for p in desc_ps])
    except AttributeError:
        desc = 'N/A'
    product_dict['Description'] = desc.strip()
    try:
        imgtag = soup.find('img', {'id': 'landingImage'})
        imageurl = dict(imgtag.attrs)['src']
    except AttributeError:
        imageurl = ''
    product_dict['MainImage'] = imageurl
    options, options_to_image = ({}, {})
    try:
        option_body = soup.find(name='div', attrs={'id': 'softlinesTwister_feature_div'})
        if option_body is None:
            option_body = soup.find(name='div', attrs={'id': 'twister_feature_div'})
        option_blocks = option_body.findAll(name='ul')
        for block in option_blocks:
            name = json.loads(block['data-a-button-group'])['name']
            opt_list = []
            for li in block.findAll('li'):
                img = li.find(name='img')
                if img is not None:
                    opt = img['alt'].strip()
                    opt_img = img['src']
                    if len(opt) > 0:
                        options_to_image[opt] = opt_img
                else:
                    opt = li.text.strip()
                if len(opt) > 0:
                    opt_list.append(opt)
            options[name.replace('_name', '').replace('twister_', '')] = opt_list
    except AttributeError:
        options = {}
    product_dict['options'], product_dict['option_to_image'] = (options, options_to_image)
    return product_dict

