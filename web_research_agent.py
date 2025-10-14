import requests
import json
import time
from datetime import datetime
import re
from typing import List, Dict, Optional
from collections import defaultdict

class WebResearchAgent:
    def __init__(self):
        self.supported_apis = {
            'google_scholar': self.search_google_scholar,
            'pubmed': self.search_pubmed,
            'who': self.search_who_health_topics,
            'medical_news': self.search_medical_news
        }
        
        # API endpoints and configurations
        self.api_config = {
            'google_scholar_base': 'https://serpapi.com/search',
            'pubmed_base': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',
            'pubmed_summary_base': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi',
            'who_base': 'https://www.who.int/api/hubs',
            'news_api_base': 'https://newsapi.org/v2/everything'
        }
        
        # You'll need to get free API keys from these services
        self.api_keys = {
            'serpapi': 'your_serpapi_key_here',  # Get from https://serpapi.com/
            'newsapi': 'your_newsapi_key_here',  # Get from https://newsapi.org/
        }
        
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
        
    def search_google_scholar(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Google Scholar for scientific papers"""
        try:
            params = {
                'engine': 'google_scholar',
                'q': f'dementia {query}',
                'api_key': self.api_keys['serpapi'],
                'num': max_results
            }
            
            response = requests.get(self.api_config['google_scholar_base'], params=params)
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for article in data.get('organic_results', [])[:max_results]:
                    result = {
                        'title': article.get('title', ''),
                        'snippet': article.get('snippet', ''),
                        'link': article.get('link', ''),
                        'source': 'Google Scholar',
                        'type': 'research_paper',
                        'authors': article.get('publication_info', {}).get('authors', []),
                        'year': article.get('publication_info', {}).get('year'),
                        'citation_count': article.get('inline_links', {}).get('cited_by', {}).get('total', 0)
                    }
                    results.append(result)
                
                return results
        except Exception as e:
            print(f"Google Scholar search error: {e}")
        
        return []
    
    def search_pubmed(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search PubMed for medical research"""
        try:
            # Search for articles
            search_params = {
                'db': 'pubmed',
                'term': f'dementia {query}',
                'retmax': max_results,
                'retmode': 'json',
                'sort': 'relevance'
            }
            
            search_response = requests.get(self.api_config['pubmed_base'], params=search_params)
            if search_response.status_code == 200:
                search_data = search_response.json()
                article_ids = search_data.get('esearchresult', {}).get('idlist', [])
                
                if not article_ids:
                    return []
                
                # Get article details
                summary_params = {
                    'db': 'pubmed',
                    'id': ','.join(article_ids),
                    'retmode': 'json'
                }
                
                summary_response = requests.get(self.api_config['pubmed_summary_base'], params=summary_params)
                if summary_response.status_code == 200:
                    summary_data = summary_response.json()
                    results = []
                    
                    for article_id in article_ids:
                        article_data = summary_data.get('result', {}).get(article_id, {})
                        results.append({
                            'title': article_data.get('title', ''),
                            'snippet': f"PubMed ID: {article_id}",
                            'link': f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/",
                            'source': 'PubMed',
                            'type': 'medical_research',
                            'authors': article_data.get('authors', []),
                            'publication_date': article_data.get('pubdate', ''),
                            'journal': article_data.get('source', '')
                        })
                    
                    return results
                    
        except Exception as e:
            print(f"PubMed search error: {e}")
        
        return []
    
    def search_who_health_topics(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search WHO health topics and guidelines"""
        try:
            # This is a simplified implementation - WHO API requires authentication
            # Using a mock implementation for demonstration
            who_knowledge_base = {
                'dementia': [
                    "Dementia is a syndrome in which there is deterioration in cognitive function beyond what might be expected from the usual consequences of biological aging.",
                    "Worldwide, around 55 million people have dementia, with over 60% living in low- and middle-income countries.",
                    "Alzheimer's disease is the most common form of dementia and may contribute to 60–70% of cases."
                ],
                'prevention': [
                    "Physical activity, not smoking, avoiding harmful use of alcohol, controlling weight, eating a healthy diet, and maintaining healthy blood pressure, cholesterol and blood sugar levels may reduce dementia risk.",
                    "Cognitive stimulation throughout lifespan is associated with reduced dementia risk."
                ],
                'treatment': [
                    "There is no treatment currently available to cure dementia. Anti-dementia medicines and disease-modifying therapies developed to date have limited efficacy.",
                    "Supportive environments and person-centered care are crucial for maintaining quality of life."
                ]
            }
            
            results = []
            query_lower = query.lower()
            
            for category, facts in who_knowledge_base.items():
                if any(word in query_lower for word in [category, 'who', 'world health']):
                    for fact in facts[:2]:
                        results.append({
                            'title': f'WHO Dementia Information - {category.title()}',
                            'snippet': fact,
                            'link': 'https://www.who.int/health-topics/dementia',
                            'source': 'World Health Organization',
                            'type': 'health_guidelines',
                            'publication_date': '2024',
                            'reliability_score': 0.95
                        })
            
            return results[:max_results]
            
        except Exception as e:
            print(f"WHO search error: {e}")
        
        return []
    
    def search_medical_news(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search recent medical news and updates"""
        try:
            params = {
                'q': f'dementia {query}',
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': max_results,
                'apiKey': self.api_keys['newsapi']
            }
            
            response = requests.get(self.api_config['news_api_base'], params=params)
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for article in data.get('articles', [])[:max_results]:
                    # Filter for reputable medical sources
                    reputable_sources = ['medical news today', 'healthline', 'webmd', 'mayo clinic', 'nih', 'cdc']
                    source_name = article.get('source', {}).get('name', '').lower()
                    
                    if any(reputable in source_name for reputable in reputable_sources):
                        results.append({
                            'title': article.get('title', ''),
                            'snippet': article.get('description', ''),
                            'link': article.get('url', ''),
                            'source': article.get('source', {}).get('name', ''),
                            'type': 'medical_news',
                            'publication_date': article.get('publishedAt', ''),
                            'reliability_score': 0.85
                        })
                
                return results
                
        except Exception as e:
            print(f"Medical news search error: {e}")
        
        return []
    
    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search using DuckDuckGo Instant Answer API (no API key required)"""
        try:
            params = {
                'q': f'dementia {query}',
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }
            
            response = requests.get('https://api.duckduckgo.com/', params=params)
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Extract from Abstract
                if data.get('Abstract'):
                    results.append({
                        'title': data.get('Heading', 'Dementia Information'),
                        'snippet': data.get('AbstractText', ''),
                        'link': data.get('AbstractURL', ''),
                        'source': 'DuckDuckGo Instant Answers',
                        'type': 'general_knowledge',
                        'reliability_score': 0.80
                    })
                
                # Extract from RelatedTopics
                for topic in data.get('RelatedTopics', [])[:max_results-1]:
                    if 'Text' in topic and 'FirstURL' in topic:
                        results.append({
                            'title': topic.get('Text', '').split('.')[0],
                            'snippet': topic.get('Text', ''),
                            'link': topic.get('FirstURL', ''),
                            'source': 'DuckDuckGo',
                            'type': 'related_topic',
                            'reliability_score': 0.75
                        })
                
                return results
                
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
        
        return []
    
    def search_wikipedia(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search Wikipedia for dementia-related information"""
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': f'dementia {query}',
                'srlimit': max_results
            }
            
            response = requests.get('https://en.wikipedia.org/w/api.php', params=params)
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for page in data.get('query', {}).get('search', [])[:max_results]:
                    # Get page extract
                    extract_params = {
                        'action': 'query',
                        'format': 'json',
                        'prop': 'extracts',
                        'exintro': True,
                        'explaintext': True,
                        'pageids': page['pageid']
                    }
                    
                    extract_response = requests.get('https://en.wikipedia.org/w/api.php', params=extract_params)
                    if extract_response.status_code == 200:
                        extract_data = extract_response.json()
                        page_data = extract_data.get('query', {}).get('pages', {}).get(str(page['pageid']), {})
                        
                        results.append({
                            'title': page_data.get('title', ''),
                            'snippet': page_data.get('extract', '')[:300] + '...',
                            'link': f"https://en.wikipedia.org/?curid={page['pageid']}",
                            'source': 'Wikipedia',
                            'type': 'encyclopedic',
                            'reliability_score': 0.85
                        })
                
                return results
                
        except Exception as e:
            print(f"Wikipedia search error: {e}")
        
        return []
    
    def analyze_claim_with_web_research(self, claim: str, max_results_per_source: int = 3) -> Dict:
        """
        Comprehensive web research for a claim
        Returns aggregated results from multiple sources
        """
        print(f"🔍 Conducting web research for: {claim}")
        
        # Cache check
        cache_key = f"research_{hash(claim)}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return cached_data
        
        all_results = {
            'research_papers': [],
            'medical_guidelines': [],
            'news_articles': [],
            'general_knowledge': [],
            'summary_analysis': {}
        }
        
        # Search across all sources
        sources = [
            ('pubmed', self.search_pubmed),
            ('google_scholar', self.search_google_scholar),
            ('who', self.search_who_health_topics),
            ('medical_news', self.search_medical_news),
            ('wikipedia', self.search_wikipedia),
            ('duckduckgo', self.search_duckduckgo)
        ]
        
        for source_name, search_func in sources:
            try:
                results = search_func(claim, max_results_per_source)
                
                # Categorize results
                for result in results:
                    result_type = result.get('type', 'general_knowledge')
                    
                    if result_type in ['research_paper', 'medical_research']:
                        all_results['research_papers'].append(result)
                    elif result_type in ['health_guidelines', 'encyclopedic']:
                        all_results['medical_guidelines'].append(result)
                    elif result_type == 'medical_news':
                        all_results['news_articles'].append(result)
                    else:
                        all_results['general_knowledge'].append(result)
                        
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"Error searching {source_name}: {e}")
                continue
        
        # Generate summary analysis
        all_results['summary_analysis'] = self.generate_research_summary(claim, all_results)
        
        # Cache the results
        self.cache[cache_key] = (time.time(), all_results)
        
        return all_results
    
    def generate_research_summary(self, claim: str, research_results: Dict) -> Dict:
        """Generate a summary analysis based on research findings"""
        total_sources = sum(len(results) for results in research_results.values() if isinstance(results, list))
        
        # Analyze consensus
        supporting_evidence = 0
        conflicting_evidence = 0
        neutral_evidence = 0
        
        # Simple keyword-based analysis (in real implementation, use more sophisticated NLP)
        claim_lower = claim.lower()
        misinformation_keywords = ['cure', 'miracle', 'secret', 'conspiracy', 'big pharma', 'cover up']
        factual_keywords = ['study shows', 'research indicates', 'clinical trial', 'evidence-based']
        
        has_misinfo_pattern = any(keyword in claim_lower for keyword in misinformation_keywords)
        has_factual_pattern = any(keyword in claim_lower for keyword in factual_keywords)
        
        summary = {
            'total_sources_found': total_sources,
            'research_paper_count': len(research_results['research_papers']),
            'guideline_count': len(research_results['medical_guidelines']),
            'news_count': len(research_results['news_articles']),
            'consensus_level': 'high' if total_sources > 5 else 'medium' if total_sources > 2 else 'low',
            'has_research_backing': len(research_results['research_papers']) > 0,
            'has_medical_guidelines': len(research_results['medical_guidelines']) > 0,
            'misinformation_indicators': has_misinfo_pattern,
            'factual_indicators': has_factual_pattern,
            'research_quality_score': min(1.0, total_sources / 10.0)
        }
        
        return summary
    
    def get_web_evidence_for_claim(self, claim: str) -> List[Dict]:
        """Get formatted web evidence for integration with main system"""
        research_data = self.analyze_claim_with_web_research(claim)
        
        evidence_items = []
        
        # Convert research papers to evidence format
        for paper in research_data['research_papers'][:3]:
            evidence_items.append({
                'text': f"Research: {paper['title']}. {paper['snippet']}",
                'source': paper['source'],
                'type': 'scientific_research',
                'rag_type': 'Web Research',
                'reliability': paper.get('reliability_score', 0.8),
                'link': paper.get('link', '')
            })
        
        # Convert medical guidelines to evidence format
        for guideline in research_data['medical_guidelines'][:2]:
            evidence_items.append({
                'text': f"Medical Guideline: {guideline['snippet']}",
                'source': guideline['source'],
                'type': 'medical_guideline',
                'rag_type': 'Web Research',
                'reliability': guideline.get('reliability_score', 0.9),
                'link': guideline.get('link', '')
            })
        
        # Convert news articles to evidence format
        for news in research_data['news_articles'][:2]:
            evidence_items.append({
                'text': f"Recent News: {news['title']}. {news['snippet']}",
                'source': news['source'],
                'type': 'current_news',
                'rag_type': 'Web Research',
                'reliability': news.get('reliability_score', 0.7),
                'link': news.get('link', '')
            })
        
        # Add summary as additional evidence
        summary = research_data['summary_analysis']
        if summary['total_sources_found'] > 0:
            evidence_items.append({
                'text': f"Web research found {summary['total_sources_found']} sources including {summary['research_paper_count']} research papers and {summary['guideline_count']} medical guidelines.",
                'source': 'Research Summary',
                'type': 'analysis_summary',
                'rag_type': 'Web Research',
                'reliability': summary['research_quality_score']
            })
        
        return evidence_items

# Singleton instance
web_research_agent = WebResearchAgent()