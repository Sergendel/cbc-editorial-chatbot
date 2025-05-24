from config import Config

config = Config("config.yml")

print(config.raw_news_path)  # explicitly outputs: data/raw/news-dataset.json
print(config.guidelines_url)  # explicitly outputs URL to CBC guidelines
