# Cluster 5

def screenshot_usage(crawler):
    cprint("\n📸 [bold cyan]Let's take a screenshot of the page![/bold cyan]")
    result = crawler.run(url='https://www.nbcnews.com/business', screenshot=True)
    cprint('[LOG] 📦 [bold yellow]Screenshot result:[/bold yellow]')
    with open('screenshot.png', 'wb') as f:
        f.write(base64.b64decode(result.screenshot))
    cprint("Screenshot saved to 'screenshot.png'!")
    print_result(result)

def add_chunking_strategy(crawler):
    cprint("\n🧩 [bold cyan]Let's add a chunking strategy: RegexChunking![/bold cyan]", True)
    cprint("RegexChunking is a simple chunking strategy that splits the text based on a given regex pattern. Let's see it in action!")
    result = crawler.run(url='https://www.nbcnews.com/business', chunking_strategy=RegexChunking(patterns=['\n\n']))
    cprint('[LOG] 📦 [bold yellow]RegexChunking result:[/bold yellow]')
    print_result(result)
    cprint('\n🔍 [bold cyan]Time to explore another chunking strategy: NlpSentenceChunking![/bold cyan]', True)
    cprint("NlpSentenceChunking uses NLP techniques to split the text into sentences. Let's see how it performs!")
    result = crawler.run(url='https://www.nbcnews.com/business', chunking_strategy=NlpSentenceChunking())
    cprint('[LOG] 📦 [bold yellow]NlpSentenceChunking result:[/bold yellow]')
    print_result(result)

def add_extraction_strategy(crawler):
    cprint("\n🧠 [bold cyan]Let's get smarter with an extraction strategy: CosineStrategy![/bold cyan]", True)
    cprint("CosineStrategy uses cosine similarity to extract semantically similar blocks of text. Let's see it in action!")
    result = crawler.run(url='https://www.nbcnews.com/business', extraction_strategy=CosineStrategy(word_count_threshold=10, max_dist=0.2, linkage_method='ward', top_k=3, sim_threshold=0.3, verbose=True))
    cprint('[LOG] 📦 [bold yellow]CosineStrategy result:[/bold yellow]')
    print_result(result)
    cprint("You can pass other parameters like 'semantic_filter' to the CosineStrategy to extract semantically similar blocks of text. Let's see it in action!")
    result = crawler.run(url='https://www.nbcnews.com/business', extraction_strategy=CosineStrategy(semantic_filter='inflation rent prices'))
    cprint('[LOG] 📦 [bold yellow]CosineStrategy result with semantic filter:[/bold yellow]')
    print_result(result)

def main():
    cprint("🌟 [bold green]Welcome to the Crawl4ai Quickstart Guide! Let's dive into some web crawling fun! 🌐[/bold green]")
    cprint('⛳️ [bold cyan]First Step: Create an instance of WebCrawler and call the `warmup()` function.[/bold cyan]')
    cprint("If this is the first time you're running Crawl4ai, this might take a few seconds to load required model files.")
    crawler = create_crawler()
    crawler.always_by_pass_cache = True
    basic_usage(crawler)
    understanding_parameters(crawler)
    crawler.always_by_pass_cache = True
    screenshot_usage(crawler)
    add_chunking_strategy(crawler)
    add_extraction_strategy(crawler)
    add_llm_extraction_strategy(crawler)
    targeted_extraction(crawler)
    interactive_extraction(crawler)
    multiple_scrip(crawler)
    cprint("\n🎉 [bold green]Congratulations! You've made it through the Crawl4ai Quickstart Guide! Now go forth and crawl the web like a pro! 🕸️[/bold green]")

