# Cluster 6

def using_crawler_hooks_dleay_example(crawler):

    def delay(driver):
        print('Delaying for 5 seconds...')
        time.sleep(5)
        print('Resuming...')

    def create_crawler():
        crawler_strategy = LocalSeleniumCrawlerStrategy(verbose=True)
        crawler_strategy.set_hook('after_get_url', delay)
        crawler = WebCrawler(verbose=True, crawler_strategy=crawler_strategy)
        crawler.warmup()
        return crawler
    cprint("\n🔗 [bold cyan]Using Crawler Hooks: Let's add a delay after fetching the url to make sure entire page is fetched.[/bold cyan]")
    crawler = create_crawler()
    result = crawler.run(url='https://google.com', bypass_cache=True)
    cprint('[LOG] 📦 [bold yellow]Crawler Hooks result:[/bold yellow]')
    print_result(result)

def create_crawler():
    crawler_strategy = LocalSeleniumCrawlerStrategy(verbose=True)
    crawler_strategy.set_hook('after_get_url', delay)
    crawler = WebCrawler(verbose=True, crawler_strategy=crawler_strategy)
    crawler.warmup()
    return crawler

