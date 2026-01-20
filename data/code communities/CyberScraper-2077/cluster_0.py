# Cluster 0

def handle_oauth_callback():
    if 'code' in st.query_params:
        try:
            flow = Flow.from_client_secrets_file('client_secret.json', scopes=SCOPES, redirect_uri=get_redirect_uri())
            flow.fetch_token(code=st.query_params['code'])
            st.session_state['google_auth_token'] = flow.credentials.to_json()
            st.success('Successfully authenticated with Google!')
            st.query_params.clear()
        except Exception as e:
            st.error(f'Error during OAuth callback: {str(e)}')

def get_redirect_uri():
    return st.get_option('server.baseUrlPath') or 'http://localhost:8501'

def initiate_google_auth():
    if not os.path.exists('client_secret.json'):
        st.error('Google Sheets integration is not set up correctly.')
        st.markdown('Please follow the setup guide for Google Sheets integration in the [CyberScraper-2077 README](https://github.com/itsOwen/CyberScraper-2077/blob/main/README.md#setup-google-sheets-authentication).')
        st.info("Once you've completed the setup, restart the application.")
        return
    flow = Flow.from_client_secrets_file('client_secret.json', scopes=SCOPES, redirect_uri=get_redirect_uri())
    authorization_url, state = flow.authorization_url(prompt='consent')
    st.session_state['oauth_state'] = state
    st.markdown(f'Please visit this URL to authorize the application: [Auth URL]({authorization_url})')
    st.info("After authorizing, you'll be redirected back to this app. Then you can proceed with uploading.")

