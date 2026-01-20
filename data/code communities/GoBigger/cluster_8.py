# Cluster 8

def test_save_screen_data_to_img():
    screen_data = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    img_path = './temp.jpg'
    save_screen_data_to_img(screen_data, img_path=None)
    assert True

def save_screen_data_to_img(screen_data, img_path=None):
    """
    Overview:
        Save the numpy screen data as the corresponding picture
    """
    img = cv2.cvtColor(screen_data, cv2.COLOR_RGB2BGR)
    img = np.fliplr(img)
    img = np.rot90(img)
    if img_path is not None:
        cv2.imwrite(img_path, img)

