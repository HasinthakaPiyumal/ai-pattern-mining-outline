# Cluster 17

def test_convert_dict_to_actions():
    asin = '334490012932'
    page_num = 2
    products = [{'asin': '125331076844', 'Title': 'Modern Tall Torchiere Floor Lamp Brushed Nickel Chrome Metal Decor Living Room', 'Price': '$129.95'}, {'asin': '125109985453', 'Title': 'Floor Lamps Set of 2 Polished Steel Crystal Glass for Living Room Bedroom', 'Price': '$179.99'}, {'asin': '125265434055', 'Title': 'Floor Lamp Nickel/Polished Concrete Finish with Off-White Linen Fabric Shade', 'Price': '$130.68'}, {'asin': '195197281169', 'Title': 'New ListingVintage Mid Century Modern Glass Amber Globe Tension Pole Floor Lamp Light', 'Price': '$165.00'}, {'asin': '195197512929', 'Title': 'New ListingVTG Brass Floor Lamp Glass Shade 63.5" Tall 12" Diameter Glass Shade Original', 'Price': '$279.45'}, {'asin': '304550250934', 'Title': 'Vintage Mid Century Modern 3 Light Tension Pole Floor Lamp glass shades atomic a', 'Price': '$149.99'}, {'asin': '175338033811', 'Title': 'Antique FOSTORIA Ornate Brass Piano  Adjustable Floor Oil Lamp up to 76" Tall !!', 'Price': '$1,995.00'}, {'asin': '334490012932', 'Title': 'Vintage Mid Century Glass Shade Amber Globe 3 Tension Pole Floor Lamp Light MCM', 'Price': '$128.00'}, {'asin': '185433933521', 'Title': 'Brass & Pink Glass Lotus 6 Petal Lamp Shades Set Of Two Replacement Parts As Is', 'Price': '$90.00'}]
    actions = convert_dict_to_actions(Page.RESULTS, products, asin, page_num)
    assert actions['valid'] == ['click[back to search]', 'click[< prev]', 'click[item - Modern Tall Torchiere Floor Lamp Brushed Nickel Chrome Metal Decor Living Room]', 'click[item - Floor Lamps Set of 2 Polished Steel Crystal Glass for Living Room Bedroom]', 'click[item - Floor Lamp Nickel/Polished Concrete Finish with Off-White Linen Fabric Shade]', 'click[item - New ListingVintage Mid Century Modern Glass Amber Globe Tension Pole Floor Lamp Light]', 'click[item - New ListingVTG Brass Floor Lamp Glass Shade 63.5" Tall 12" Diameter Glass Shade Original]', 'click[item - Vintage Mid Century Modern 3 Light Tension Pole Floor Lamp glass shades atomic a]', 'click[item - Antique FOSTORIA Ornate Brass Piano  Adjustable Floor Oil Lamp up to 76" Tall !!]', 'click[item - Vintage Mid Century Glass Shade Amber Globe 3 Tension Pole Floor Lamp Light MCM]', 'click[item - Brass & Pink Glass Lotus 6 Petal Lamp Shades Set Of Two Replacement Parts As Is]']
    asin = '224636269803'
    products = {'224636269803': {'asin': '224636269803', 'Title': 'Sony SRS-XB01 EXTRA BASS Portable Water-Resistant  Wireless Bluetooth Speaker', 'Price': '24.99', 'MainImage': 'https://i.ebayimg.com/images/g/jVEAAOSwCLBhXLuD/s-l500.jpg', 'Rating': None, 'options': {'Color': ['Black', 'White', 'Red', 'Blue']}, 'option_to_image': {}, 'Description': "eBay Sony EXTRA BASS Portable Water-Resistant Wireless Bluetooth SpeakerBRAND NEW ITEMFREE SHIPPING WITHIN USA30 DAY RETURN POLICYKey FeaturesEXTRA BASS for deep, punchy soundCompact portable designUp to 6 hours of battery lifeWater resistant for worry-free useSupplied with color-coordinated strap What's in the Box?Sony EXTRA BASS Portable Bluetooth SpeakerPower supplyUser manual HIGHLIGHTSMUSIC THAT TRAVELSSmall size but mighty in volume to deliver powerful beats wherever you travelHANDS FREE CALLINGWith the built-in microphone, taking calls from your smartphone is easy. SPLASHPROOF CASINGTake to the pool or beach without worrying about water damaging the speaker unit UPGRADE THE AUDIOWirelessly connects 2 speakers and achieve stereo sound with speaker add function LONGER BATTERY LIFELonger Virtual Happy Hours with this rechargeable speaker's 6 hour battery life Technical SpecsFeatureValueBrandSonyTypePortable speakerModel NumberSRSXB01BluetoothYesFrequency range2.4 GHzMax. Communication Range32 ftBattery LifeApprox. 6 hrsWater ProtectionIPX5Input and Output TerminalsStereo Mini Jack (IN)Dimensions (W x H x D)Approx. 3 1/4 × 2 3/8 × 2 1/4 inWeightApprox. 5.65 oz", 'BulletPoints': "Item specifics Condition:New: A brand-new, unused, unopened, undamaged item in its original packaging (where packaging is ...  Read moreabout the conditionNew: A brand-new, unused, unopened, undamaged item in its original packaging (where packaging is applicable). Packaging should be the same as what is found in a retail store, unless the item is handmade or was packaged by the manufacturer in non-retail packaging, such as an unprinted box or plastic bag. See the seller's listing for full details. See all condition definitionsopens in a new window or tab  Model:EXTRA BASS Connectivity:Bluetooth, Wireless Type:Portable Speaker System Compatible Model:EXTRA BASS, Portable Water-Resistant Features:Bluetooth, Water-Resistant MPN:SRS-XB01/B, SRS-XB01/L, SRS-XB01/R, SRS-XB01/W Brand:Sony"}}
    actions = convert_dict_to_actions(Page.ITEM_PAGE, products, asin, 1)
    assert actions['valid'] == ['click[back to search]', 'click[< prev]', 'click[description]', 'click[features]', 'click[buy now]', 'click[Black]', 'click[White]', 'click[Red]', 'click[Blue]']
    actions = convert_dict_to_actions(Page.SUB_PAGE, {}, '12345', 1)
    assert actions['valid'] == ['click[back to search]', 'click[< prev]']

def convert_dict_to_actions(page_type, products=None, asin=None, page_num=None) -> dict:
    info = {'valid': []}
    if page_type == Page.RESULTS:
        info['valid'] = ['click[back to search]']
        if products is None or page_num is None:
            print(page_num)
            print(products)
            raise Exception('Provide `products`, `page_num` to get `results` valid actions')
        if len(products) > 10:
            info['valid'].append('click[next >]')
        if page_num > 1:
            info['valid'].append('click[< prev]')
        for product in products:
            info['valid'].append('click[item - ' + product['Title'] + ']')
    if page_type == Page.ITEM_PAGE:
        if products is None or asin is None:
            raise Exception('Provide `products` and `asin` to get `item_page` valid actions')
        info['valid'] = ['click[back to search]', 'click[< prev]', 'click[description]', 'click[features]', 'click[buy now]']
        if 'options' in products[asin]:
            for key, values in products[asin]['options'].items():
                for value in values:
                    info['valid'].append('click[' + value + ']')
    if page_type == Page.SUB_PAGE:
        info['valid'] = ['click[back to search]', 'click[< prev]']
    info['image_feat'] = torch.zeros(512)
    return info

