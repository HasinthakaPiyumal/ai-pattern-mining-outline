# Cluster 6

def display_google_sheets_button(data, unique_key):
    if not os.path.exists('client_secret.json'):
        st.warning('Google Sheets integration is not set up.')
        st.markdown('To enable Google Sheets integration, please follow the setup guide in the [CyberScraper-2077 README](https://github.com/itsOwen/CyberScraper-2077/blob/main/README.md#setup-google-sheets-authentication).')
        return
    creds = get_google_sheets_credentials()
    if not creds:
        auth_button = '🔑 Authorize Google Sheets'
        if st.button(auth_button, key=f'auth_sheets_{unique_key}', help='Authorize access to Google Sheets'):
            initiate_google_auth()
    else:
        upload_button = '✅ Upload to Google Sheets'
        if st.button(upload_button, key=f'upload_{unique_key}', help='Upload data to Google Sheets'):
            with st.spinner('Uploading to Google Sheets...'):
                spreadsheet_id = upload_to_google_sheets(data)
                if spreadsheet_id:
                    st.success(f'Hey Choom! Data uploaded successfully. Spreadsheet ID: {spreadsheet_id}')
                    st.markdown(f'[Open Spreadsheet](https://docs.google.com/spreadsheets/d/{spreadsheet_id})')
                else:
                    st.error('Failed to upload data to Google Sheets. Check the console for error details.')

def upload_to_google_sheets(data):
    creds = get_google_sheets_credentials()
    if not creds:
        return None
    try:
        service = build('sheets', 'v4', credentials=creds)
        spreadsheet = {'properties': {'title': f'CyberScraper Data {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}'}}
        spreadsheet = service.spreadsheets().create(body=spreadsheet, fields='spreadsheetId').execute()
        spreadsheet_id = spreadsheet.get('spreadsheetId')
        if isinstance(data, pd.DataFrame):
            df = clean_data_for_sheets(data)
        else:
            return None
        values = [df.columns.tolist()] + df.values.tolist()
        body = {'values': values}
        result = service.spreadsheets().values().update(spreadsheetId=spreadsheet_id, range='Sheet1', valueInputOption='RAW', body=body).execute()
        return spreadsheet_id
    except HttpError as error:
        print(f'An HTTP error occurred: {error}')
        return None
    except Exception as e:
        print(f'An error occurred: {str(e)}')
        return None

def get_google_sheets_credentials():
    if not os.path.exists('client_secret.json'):
        return None
    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception as e:
            print(f'Error loading credentials from file: {str(e)}')
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                save_credentials(creds)
            except Exception as e:
                print(f'Error refreshing credentials: {str(e)}')
                creds = None
        else:
            creds = None
    if not creds:
        if 'google_auth_token' in st.session_state:
            try:
                creds = Credentials.from_authorized_user_info(json.loads(st.session_state['google_auth_token']), SCOPES)
                save_credentials(creds)
            except Exception as e:
                print(f'Error creating credentials from session state: {str(e)}')
    return creds

def clean_data_for_sheets(df):

    def clean_value(val):
        if pd.isna(val):
            return ''
        if isinstance(val, (int, float)):
            return str(val)
        return str(val).replace('\n', ' ').replace('\r', '')
    for col in df.columns:
        df[col] = df[col].map(clean_value)
    if 'comments' in df.columns:
        df['comments'] = df['comments'].astype(str)
    return df

