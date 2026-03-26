import urllib.request
import xml.etree.ElementTree as ET
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

url = 'http://export.arxiv.org/api/query?search_query=all:GraphRAG&sortBy=submittedDate&sortOrder=descending&max_results=10'
req = urllib.request.Request(url)
try:
    response = urllib.request.urlopen(req)
    data = response.read()
    root = ET.fromstring(data)
    namespace = '{http://www.w3.org/2005/Atom}'
    for i, entry in enumerate(root.findall(namespace + 'entry')):
        title_elem = entry.find(namespace + 'title')
        title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None and title_elem.text else ''
        
        authors = []
        for author in entry.findall(namespace + 'author'):
            name_elem = author.find(namespace + 'name')
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text)
                
        published_elem = entry.find(namespace + 'published')
        published = published_elem.text if published_elem is not None and published_elem.text else ''
        
        summary_elem = entry.find(namespace + 'summary')
        summary = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None and summary_elem.text else ''
        
        print(f"[{i+1}] Title: {title}")
        print(f"Authors: {', '.join(authors)}")
        print(f"Year: {published[:4] if published else ''}")
        print(f"Summary: {summary}\n")
except Exception as e:
    print(e)
