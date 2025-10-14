# config.py
# Free API Keys Configuration
API_KEYS = {
    # Get free API key from https://serpapi.com/ (100 searches/month free)
    'SERPAPI_KEY': 'd7eae1e00815db555c2490242a523d64e8e1c42b12da4db4aba91e9b31f71edb',
    
    # Get free API key from https://newsapi.org/ (100 requests/day free)
    'NEWSAPI_KEY': '70939ad0cb48470ab0834dc1454feed2',
    
    # No API key required for these:
    # - DuckDuckGo API
    # - Wikipedia API
    # - PubMed API
}

# Research Configuration
RESEARCH_CONFIG = {
    'MAX_RESULTS_PER_SOURCE': 3,
    'REQUEST_TIMEOUT': 10,
    'CACHE_DURATION': 3600,  # 1 hour
    'RATE_LIMIT_DELAY': 0.5  # seconds between requests
}

#Dementia risk increases with age