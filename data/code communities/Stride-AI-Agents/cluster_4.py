# Cluster 4

def basic_usage(crawler):
    cprint('🛠️ [bold cyan]Basic Usage: Simply provide a URL and let Crawl4ai do the magic![/bold cyan]')
    result = crawler.run(url='https://www.nbcnews.com/business', only_text=True)
    cprint('[LOG] 📦 [bold yellow]Basic crawl result:[/bold yellow]')
    print_result(result)

def cprint(message, press_any_key=False):
    console.print(message)
    if press_any_key:
        console.print('Press any key to continue...', style='')
        input()

def print_result(result):
    console.print(f'\t[bold]Result:[/bold]')
    for key, value in result.model_dump().items():
        if isinstance(value, str) and value:
            console.print(f'\t{key}: [green]{value[:20]}...[/green]')
    if result.extracted_content:
        items = json.loads(result.extracted_content)
        print(f'\t[bold]{len(items)} blocks is extracted![/bold]')

def basic_usage_some_params(crawler):
    cprint('🛠️ [bold cyan]Basic Usage: Simply provide a URL and let Crawl4ai do the magic![/bold cyan]')
    result = crawler.run(url='https://www.nbcnews.com/business', word_count_threshold=1, only_text=True)
    cprint('[LOG] 📦 [bold yellow]Basic crawl result:[/bold yellow]')
    print_result(result)

def understanding_parameters(crawler):
    cprint("\n🧠 [bold cyan]Understanding 'bypass_cache' and 'include_raw_html' parameters:[/bold cyan]")
    cprint("By default, Crawl4ai caches the results of your crawls. This means that subsequent crawls of the same URL will be much faster! Let's see this in action.")
    cprint('1️⃣ First crawl (caches the result):', True)
    start_time = time.time()
    result = crawler.run(url='https://www.nbcnews.com/business')
    end_time = time.time()
    cprint(f'[LOG] 📦 [bold yellow]First crawl took {end_time - start_time} seconds and result (from cache):[/bold yellow]')
    print_result(result)
    cprint('2️⃣ Second crawl (Force to crawl again):', True)
    start_time = time.time()
    result = crawler.run(url='https://www.nbcnews.com/business', bypass_cache=True)
    end_time = time.time()
    cprint(f'[LOG] 📦 [bold yellow]Second crawl took {end_time - start_time} seconds and result (forced to crawl):[/bold yellow]')
    print_result(result)

def add_llm_extraction_strategy(crawler):
    cprint('\n🤖 [bold cyan]Time to bring in the big guns: LLMExtractionStrategy without instructions![/bold cyan]', True)
    cprint("LLMExtractionStrategy uses a large language model to extract relevant information from the web page. Let's see it in action!")
    result = crawler.run(url='https://www.nbcnews.com/business', extraction_strategy=LLMExtractionStrategy(provider='openai/gpt-4o', api_token=os.getenv('OPENAI_API_KEY')))
    cprint('[LOG] 📦 [bold yellow]LLMExtractionStrategy (no instructions) result:[/bold yellow]')
    print_result(result)
    cprint("\n📜 [bold cyan]Let's make it even more interesting: LLMExtractionStrategy with instructions![/bold cyan]", True)
    cprint("Let's say we are only interested in financial news. Let's see how LLMExtractionStrategy performs with instructions!")
    result = crawler.run(url='https://www.nbcnews.com/business', extraction_strategy=LLMExtractionStrategy(provider='openai/gpt-4o', api_token=os.getenv('OPENAI_API_KEY'), instruction='I am interested in only financial news'))
    cprint('[LOG] 📦 [bold yellow]LLMExtractionStrategy (with instructions) result:[/bold yellow]')
    print_result(result)
    result = crawler.run(url='https://www.nbcnews.com/business', extraction_strategy=LLMExtractionStrategy(provider='openai/gpt-4o', api_token=os.getenv('OPENAI_API_KEY'), instruction='Extract only content related to technology'))
    cprint('[LOG] 📦 [bold yellow]LLMExtractionStrategy (with technology instruction) result:[/bold yellow]')
    print_result(result)

def targeted_extraction(crawler):
    cprint("\n🎯 [bold cyan]Targeted extraction: Let's use a CSS selector to extract only H2 tags![/bold cyan]", True)
    result = crawler.run(url='https://www.nbcnews.com/business', css_selector='h2')
    cprint('[LOG] 📦 [bold yellow]CSS Selector (H2 tags) result:[/bold yellow]')
    print_result(result)

def interactive_extraction(crawler):
    cprint("\n🖱️ [bold cyan]Let's get interactive: Passing JavaScript code to click 'Load More' button![/bold cyan]", True)
    cprint("In this example we try to click the 'Load More' button on the page using JavaScript code.")
    js_code = "\n    const loadMoreButton = Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('Load More'));\n    loadMoreButton && loadMoreButton.click();\n    "
    result = crawler.run(url='https://www.nbcnews.com/business', js=js_code)
    cprint('[LOG] 📦 [bold yellow]JavaScript Code (Load More button) result:[/bold yellow]')
    print_result(result)

def multiple_scrip(crawler):
    cprint("\n🖱️ [bold cyan]Let's get interactive: Passing JavaScript code to click 'Load More' button![/bold cyan]", True)
    cprint("In this example we try to click the 'Load More' button on the page using JavaScript code.")
    js_code = ["\n    const loadMoreButton = Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('Load More'));\n    loadMoreButton && loadMoreButton.click();\n    "] * 2
    result = crawler.run(url='https://www.nbcnews.com/business', js=js_code)
    cprint('[LOG] 📦 [bold yellow]JavaScript Code (Load More button) result:[/bold yellow]')
    print_result(result)

def using_crawler_hooks(crawler):

    def on_driver_created(driver):
        print('[HOOK] on_driver_created')
        driver.maximize_window()
        driver.get('https://example.com/login')
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, 'username')))
        driver.find_element(By.NAME, 'username').send_keys('testuser')
        driver.find_element(By.NAME, 'password').send_keys('password123')
        driver.find_element(By.NAME, 'login').click()
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'welcome')))
        driver.add_cookie({'name': 'test_cookie', 'value': 'cookie_value'})
        return driver

    def before_get_url(driver):
        print('[HOOK] before_get_url')
        driver.execute_cdp_cmd('Network.enable', {})
        driver.execute_cdp_cmd('Network.setExtraHTTPHeaders', {'headers': {'X-Test-Header': 'test'}})
        return driver

    def after_get_url(driver):
        print('[HOOK] after_get_url')
        print(driver.current_url)
        return driver

    def before_return_html(driver, html):
        print('[HOOK] before_return_html')
        print(len(html))
        return driver
    cprint("\n🔗 [bold cyan]Using Crawler Hooks: Let's see how we can customize the crawler using hooks![/bold cyan]", True)
    crawler_strategy = LocalSeleniumCrawlerStrategy(verbose=True)
    crawler_strategy.set_hook('on_driver_created', on_driver_created)
    crawler_strategy.set_hook('before_get_url', before_get_url)
    crawler_strategy.set_hook('after_get_url', after_get_url)
    crawler_strategy.set_hook('before_return_html', before_return_html)
    crawler = WebCrawler(verbose=True, crawler_strategy=crawler_strategy)
    crawler.warmup()
    result = crawler.run(url='https://example.com')
    cprint('[LOG] 📦 [bold yellow]Crawler Hooks result:[/bold yellow]')
    print_result(result=result)

