"""Literature review utilities for Agent Laboratory.

This module provides utilities for searching and retrieving scientific papers.
"""

import time
import logging
import requests
from typing import List, Dict, Any, Optional

class ArxivSearcher:
    """Class for searching and retrieving papers from arXiv."""
    
    def __init__(self):
        """Initialize the arXiv searcher."""
        self.base_url = "http://export.arxiv.org/api/query"
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """Search for papers on arXiv.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of papers with title, summary, authors, etc.
        """
        # Process query string to fit within limits
        max_query_length = 300
        if len(query) > max_query_length:
            query = query[:max_query_length]
        
        # Prepare the request parameters
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        # Make the request with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                
                # Parse the response
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)
                
                # Extract the papers
                papers = []
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    # Skip the first entry, which is the opensearch description
                    if entry.find('{http://www.w3.org/2005/Atom}title') is None:
                        continue
                    
                    # Get the title
                    title = entry.find('{http://www.w3.org/2005/Atom}title').text
                    
                    # Get the summary
                    summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
                    
                    # Get the authors
                    authors = []
                    for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                        name = author.find('{http://www.w3.org/2005/Atom}name').text
                        authors.append(name)
                    
                    # Get the published date
                    published = entry.find('{http://www.w3.org/2005/Atom}published').text.split('T')[0]
                    
                    # Get the PDF URL
                    pdf_url = None
                    for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
                        if link.get('title') == 'pdf':
                            pdf_url = link.get('href')
                            break
                    
                    # Get the arXiv ID
                    id_url = entry.find('{http://www.w3.org/2005/Atom}id').text
                    arxiv_id = id_url.split('/abs/')[-1]
                    
                    papers.append({
                        'title': title,
                        'summary': summary,
                        'authors': authors,
                        'published': published,
                        'pdf_url': pdf_url,
                        'arxiv_id': arxiv_id
                    })
                
                return papers
            
            except Exception as e:
                logging.warning(f"Error searching arXiv (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    logging.error(f"Failed to search arXiv after {max_retries} attempts")
                    return []
    
    def format_papers(self, papers: List[Dict[str, str]]) -> str:
        """Format papers into a string for display.
        
        Args:
            papers: List of papers
            
        Returns:
            Formatted string
        """
        if not papers:
            return "No papers found"
        
        paper_strings = []
        for i, paper in enumerate(papers, 1):
            paper_str = f"Paper {i}:\n"
            paper_str += f"Title: {paper['title']}\n"
            paper_str += f"Authors: {', '.join(paper['authors'])}\n"
            paper_str += f"Published: {paper['published']}\n"
            paper_str += f"Summary: {paper['summary']}\n"
            paper_str += f"PDF URL: {paper['pdf_url']}\n"
            paper_str += f"arXiv ID: {paper['arxiv_id']}\n"
            paper_strings.append(paper_str)
        
        return "\n\n".join(paper_strings)


class SemanticScholarSearcher:
    """Class for searching and retrieving papers from Semantic Scholar."""
    
    def __init__(self):
        """Initialize the Semantic Scholar searcher."""
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for papers on Semantic Scholar.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of papers with title, abstract, authors, etc.
        """
        # Prepare the request parameters
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,authors,year,venue,url,citationCount"
        }
        
        # Make the request with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                
                # Parse the response
                data = response.json()
                
                # Extract the papers
                papers = []
                for paper in data.get("data", []):
                    papers.append({
                        'title': paper.get('title', ''),
                        'abstract': paper.get('abstract', ''),
                        'authors': [author.get('name', '') for author in paper.get('authors', [])],
                        'year': paper.get('year', ''),
                        'venue': paper.get('venue', ''),
                        'url': paper.get('url', ''),
                        'citation_count': paper.get('citationCount', 0)
                    })
                
                return papers
            
            except Exception as e:
                logging.warning(f"Error searching Semantic Scholar (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    logging.error(f"Failed to search Semantic Scholar after {max_retries} attempts")
                    return []
    
    def format_papers(self, papers: List[Dict[str, Any]]) -> str:
        """Format papers into a string for display.
        
        Args:
            papers: List of papers
            
        Returns:
            Formatted string
        """
        if not papers:
            return "No papers found"
        
        paper_strings = []
        for i, paper in enumerate(papers, 1):
            paper_str = f"Paper {i}:\n"
            paper_str += f"Title: {paper['title']}\n"
            paper_str += f"Authors: {', '.join(paper['authors'])}\n"
            paper_str += f"Year: {paper['year']}\n"
            paper_str += f"Venue: {paper['venue']}\n"
            paper_str += f"Citations: {paper['citation_count']}\n"
            paper_str += f"Abstract: {paper['abstract']}\n"
            paper_str += f"URL: {paper['url']}\n"
            paper_strings.append(paper_str)
        
        return "\n\n".join(paper_strings) 