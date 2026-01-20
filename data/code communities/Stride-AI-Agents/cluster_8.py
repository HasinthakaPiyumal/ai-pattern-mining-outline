# Cluster 8

def create_database():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('\n        CREATE TABLE IF NOT EXISTS Users (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            user_id INTEGER,\n            first_name TEXT,\n            last_name TEXT,\n            email TEXT UNIQUE,\n            phone TEXT\n        )\n    ')
    cursor.execute('\n        CREATE TABLE IF NOT EXISTS PurchaseHistory (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            user_id INTEGER,\n            date_of_purchase TEXT,\n            item_id INTEGER,\n            amount REAL,\n            FOREIGN KEY (user_id) REFERENCES Users(user_id)\n        )\n    ')
    cursor.execute('\n        CREATE TABLE IF NOT EXISTS Products (\n            product_id INTEGER PRIMARY KEY,\n            product_name TEXT NOT NULL,\n            price REAL NOT NULL\n        );\n        ')
    conn.commit()

def get_connection():
    global conn
    if conn is None:
        conn = sqlite3.connect('application.db')
    return conn

def add_user(user_id, first_name, last_name, email, phone):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Users WHERE user_id = ?', (user_id,))
    if cursor.fetchone():
        return
    try:
        cursor.execute('\n            INSERT INTO Users (user_id, first_name, last_name, email, phone)\n            VALUES (?, ?, ?, ?, ?)\n        ', (user_id, first_name, last_name, email, phone))
        conn.commit()
    except sqlite3.Error as e:
        print(f'Database Error: {e}')

def add_purchase(user_id, date_of_purchase, item_id, amount):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('\n        SELECT * FROM PurchaseHistory\n        WHERE user_id = ? AND item_id = ? AND date_of_purchase = ?\n    ', (user_id, item_id, date_of_purchase))
    if cursor.fetchone():
        return
    try:
        cursor.execute('\n            INSERT INTO PurchaseHistory (user_id, date_of_purchase, item_id, amount)\n            VALUES (?, ?, ?, ?)\n        ', (user_id, date_of_purchase, item_id, amount))
        conn.commit()
    except sqlite3.Error as e:
        print(f'Database Error: {e}')

def add_product(product_id, product_name, price):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('\n        INSERT INTO Products (product_id, product_name, price)\n        VALUES (?, ?, ?);\n        ', (product_id, product_name, price))
        conn.commit()
    except sqlite3.Error as e:
        print(f'Database Error: {e}')

def initialize_database():
    global conn
    create_database()
    initial_users = [(1, 'Alice', 'Smith', 'alice@test.com', '123-456-7890'), (2, 'Bob', 'Johnson', 'bob@test.com', '234-567-8901'), (3, 'Sarah', 'Brown', 'sarah@test.com', '555-567-8901')]
    for user in initial_users:
        add_user(*user)
    initial_purchases = [(1, '2024-01-01', 101, 99.99), (2, '2023-12-25', 100, 39.99), (3, '2023-11-14', 307, 49.99)]
    for purchase in initial_purchases:
        add_purchase(*purchase)
    initial_products = [(7, 'Hat', 19.99), (8, 'Wool socks', 29.99), (9, 'Shoes', 39.99)]
    for product in initial_products:
        add_product(*product)

