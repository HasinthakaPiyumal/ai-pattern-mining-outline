# Cluster 5

@requests_mock.Mocker(kw='mock')
def test_parse_item_page_ebay(**kwargs):
    mock_file = open('tests/transfer/mocks/mock_parse_item_page_ebay', 'rb')
    mock_body = mock_file.read()
    mock_file.close()
    mock_asin = '403760625150'
    kwargs['mock'].get(f'https://www.ebay.com/itm/{mock_asin}', content=mock_body)
    output = parse_item_page_ebay(mock_asin)
    expected = {'BulletPoints': 'Item specifics Condition:New without box: A brand-new, unused, and unworn item (including handmade items) that is not in ...  Read moreabout the conditionNew without box: A brand-new, unused, and unworn item (including handmade items) that is not in original packaging or may be missing original packaging materials (such as the original box or bag). The original tags may not be attached. For example, new shoes (with absolutely no signs of wear) that are no longer in their original box fall into this category.  See all condition definitionsopens in a new window or tab  Closure:Lace Up US Shoe Size:10 Occasion:Activewear, Casual Silhouette:Puma Fabric Type:Mesh Vintage:No Cushioning Level:Moderate Department:Men Style:Sneaker Outsole Material:Rubber Features:Breathable, Comfort, Cushioned, Performance Season:Fall, Spring, Summer, Winter Idset_Mpn:193990-21 Shoe Shaft Style:Low Top Style Code:193990-16 Pattern:Solid Character:J. Cole Lining Material:Synthetic Color:Red Brand:PUMA Type:Athletic Customized:No Model:RS-Dreamer Theme:Sports Shoe Width:Standard Upper Material:Textile Insole Material:Synthetic Performance/Activity:Basketball Product Line:Puma Dreamer', 'Description': 'N/A', 'MainImage': 'https://i.ebayimg.com/images/g/4ggAAOSwpk1ioTWz/s-l500.jpg', 'Price': 'N/A', 'Rating': None, 'Title': "Puma RS-Dreamer J. Cole Basketball Shoes Red 193990-16 Men's Size 10.0", 'asin': '403760625150', 'option_to_image': {}, 'options': {}}
    assert output == expected

def parse_item_page_ebay(asin, verbose=True):
    product_dict = {}
    product_dict['asin'] = asin
    url = f'https://www.ebay.com/itm/{asin}'
    if verbose:
        print(f'Item Page URL: {url}')
    begin = time.time()
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    end = time.time()
    if verbose:
        print(f'Item page scraping took {end - begin} seconds')
    soup = BeautifulSoup(webpage.content, 'html.parser')
    try:
        product_dict['Title'] = soup.find('h1', {'class': 'x-item-title__mainTitle'}).text.strip()
    except:
        product_dict['Title'] = 'N/A'
    try:
        price_str = soup.find('div', {'class': 'mainPrice'}).text
        prices = re.findall('\\d*\\.?\\d+', price_str)
        product_dict['Price'] = prices[0]
    except:
        product_dict['Price'] = 'N/A'
    try:
        img_div = soup.find('div', {'id': 'mainImgHldr'})
        img_link = img_div.find('img', {'id': 'icImg'})['src']
        product_dict['MainImage'] = img_link
    except:
        product_dict['MainImage'] = ''
    try:
        rating = soup.find('span', {'class': 'reviews-star-rating'})['title'].split()[0]
    except:
        rating = None
    product_dict['Rating'] = rating
    options, options_to_images = ({}, {})
    try:
        option_blocks = soup.findAll('select', {'class': 'msku-sel'})
        for block in option_blocks:
            name = block['name'].strip().strip(':')
            option_tags = block.findAll('option')
            opt_list = []
            for option_tag in option_tags:
                if 'select' not in option_tag.text.lower():
                    opt_list.append(option_tag.text)
            options[name] = opt_list
    except:
        options = {}
    product_dict['options'], product_dict['option_to_image'] = (options, options_to_images)
    desc = None
    try:
        desc_link = soup.find('iframe', {'id': 'desc_ifr'})['src']
        desc_webpage = requests.get(desc_link, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
        desc_soup = BeautifulSoup(desc_webpage.content, 'html.parser')
        desc = ' '.join(desc_soup.text.split())
    except:
        desc = 'N/A'
    product_dict['Description'] = desc
    features = None
    try:
        features = soup.find('div', {'class': 'x-about-this-item'}).text
    except:
        features = 'N/A'
    product_dict['BulletPoints'] = features
    return product_dict

