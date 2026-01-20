# Cluster 1

def initialize_web_scraper_chat(url=None):
    if st.session_state.selected_model.startswith('ollama:'):
        model = st.session_state.selected_model
    else:
        model = st.session_state.selected_model
    scraper_config = ScraperConfig(use_current_browser=st.session_state.use_current_browser, headless=not st.session_state.use_current_browser, max_retries=3, delay_after_load=5, debug=True, wait_for='domcontentloaded')
    web_scraper_chat = StreamlitWebScraperChat(model_name=model, scraper_config=scraper_config)
    if url:
        web_scraper_chat.process_message(url)
        website_name = get_website_name(url)
        st.session_state.chat_history[st.session_state.current_chat_id]['name'] = website_name
    return web_scraper_chat

