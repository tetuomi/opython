from icrawler.builtin import GoogleImageCrawler

crawler = GoogleImageCrawler(storage={"root_dir": "images/hina"})
crawler.crawl(keyword="河田陽菜", max_num=300)
