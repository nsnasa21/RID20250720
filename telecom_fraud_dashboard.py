import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from collections import Counter
import numpy as np
from textblob import TextBlob
import time
from urllib.parse import urlparse
import validators
import io

# Page configuration
st.set_page_config(
    page_title="Global Telecom Fraud News Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
    }
    .news-card {
        background-color: #000000; /* Default background */
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        transition: box-shadow 0.2s ease-in-out;
    }
    .news-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .news-card.authentic {
        border-left: 5px solid #4CAF50; /* Green for authentic */
    }
    .news-card.suspicious {
        border-left: 5px solid #FFC107; /* Amber for suspicious */
        background-color: #fff8e1; /* Light amber background */
    }
    .news-card.rejected {
        border-left: 5px solid #F44336; /* Red for rejected */
        background-color: #ffebee; /* Light red background */
    }
    .fraud-type {
        background-color: #1f4e79; /* Darker blue */
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        margin-right: 0.5rem;
        display: inline-block;
    }
    .verification-tag {
        font-size: 0.7rem;
        padding: 0.2rem 0.4rem;
        border-radius: 0.2rem;
        margin-left: 0.5rem;
    }
    .tag-authentic { background-color: #4CAF50; color: white; }
    .tag-suspicious { background-color: #FFC107; color: black; }
    .tag-rejected { background-color: #F44336; color: white; }
    .status-filter-info {
        font-size: 0.8rem;
        color: #666;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class TelecomFraudDashboard:
    def __init__(self):
        # Initialize from session state or defaults
        if 'api_keys' not in st.session_state:
            st.session_state.api_keys = {
                'newsapi': '',
                'newsdata': '',
                'gnews': ''
            }
        if 'fraud_keywords' not in st.session_state:
            st.session_state.fraud_keywords = {
                'SIM Swapping': ['sim swap', 'sim swapping', 'sim hijack', 'sim card fraud', 'SIM jacking', 'number porting scam', 'mobile number takeover', '2FA bypass fraud', 'OTP interception'],
                'Smishing': ['smishing', 'sms phishing', 'text scam', 'sms fraud'],
                'Vishing': ['vishing', 'voice phishing', 'phone scam', 'robocall fraud'],
                'Caller ID Spoofing': ['caller id spoofing', 'number spoofing', 'call spoofing'],
                'Premium Rate Fraud': ['premium rate', 'premium sms', 'expensive calls'],
                'Telecom Infrastructure Fraud': ['telecom infrastructure', 'network fraud', 'billing fraud'],
                'Wangiri Fraud': ['wangiri', 'one ring scam', 'one ring fraud', 'missed call scam', 'missed call fraud', 'international callback scam', 'international callback fraud'],
                'International Revenue Share Fraud': ['irsf', 'international revenue sharing', 'revenue share fraud', 'SMS Pumping'],
                'Subscription Fraud': ['subscription fraud', 'fake subscription', 'unwanted subscription', 'telecom subscription fraud', 'fake id telecom', 'synthetic id fraud telecom', 'device jailbreak resale', 'unpaid service fraud'],
                'Toll Fraud': ['toll fraud', 'toll scam', 'pbx hacking', 'phone system fraud', 'VoIP exploitation', 'phone system compromise', 'unauthorized call routing', 'telecom system breach'],
                'SIMBox Fraud': ['SIMBox', 'SIM Box', 'SIM Box Fraud'],
                'VoIP Fraud': ['VoIP Fraud', 'Voice over IP Fraud', 'VoIP Scam', 'Voice over IP Scam'],
                'Traffic Pumping': ['traffic pumping', 'access simulation', 'interconnection fee', 'artificially inflated traffic', 'AIT', 'artificial traffic', 'interconnection fee abuse'],
                'Deposit Fraud': ['telecom deposit fraud', 'prepaid sim fraud', 'stolen credit card telecom', 'device fraud', 'online store fraud telecom'],
                'Account Takeover Fraud': ['telecom account takeover', 'ATO fraud telecom', 'account compromise telecom', 'unauthorized account access', 'stolen credentials telecom'],
                'Cellphone Cloning': ['cellphone cloning fraud', 'mobile subscriber fraud', 'telecom identity duplication', 'unauthorized cellular use'],
                'Cramming': ['phone bill cramming', 'unauthorized phone charges', 'deceptive billing telecom', 'misleading phone bill'],
                'Slamming': ['phone slamming', 'unauthorized carrier switch', 'telecom service hijacking', 'illegal phone service change'],
                'Voicemail Hacking Scam': ['voicemail hacking', 'collect call scam voicemail', 'international call fraud voicemail', 'default password exploit telecom'],
                'Robocall Scams': ['robocall scam', 'imposter scam', 'fake police call'],
                'Collect Call Scam': ['809 scam', 'collect call scam', 'international call back fraud', 'unfamiliar area code scam'],
                'Phishing': ['phishing scam', 'email phishing', 'malicious link scam', 'credential harvesting', 'fake login page', 'spoofed email', 'AI phishing'],
                'Imposter Scams': ['imposter scam', 'government imposter fraud', 'IRS scam call', 'tech support scam', 'Microsoft scam', 'anti-virus scam', 'charity fraud call', 'family emergency scam', 'grandparent scam', 'utility scam call', 'veteran benefits fraud', 'DHS scam', 'Social Security scam', 'Europol imposter scam'],
                'Romance and Catfishing Scams': ['romance scam', 'catfishing fraud', 'online dating scam', 'fake identity dating', 'deepfake romance scam'],
                'Advance Fee Scams': ['advance fee scam', 'loan scam upfront fee', 'lottery prize scam', 'sweepstakes fraud', 'government grant scam', 'inheritance scam', 'work from home scam fee', 'wire transfer scam', 'gift card scam payment', 'cryptocurrency scam payment'],
                'Online Shopping and Holiday Frauds': ['fake online store', 'counterfeit goods scam', 'holiday fraud', 'fake accommodation scam', 'AI generated listing scam', 'online marketplace fraud', 'triangulation fraud'],
                'Employment and Business Opportunity Scams': ['employment scam', 'job offer fraud', 'work from home scam', 'business opportunity fraud', 'upfront fee job', 'fake check employment', 'overpayment scam job'],
                'Copycat Websites': ['copycat website scam', 'fake government website', 'bogus customer service number', 'official document fraud online', 'search engine scam'],
                'Pharming': ['pharming attack', 'website redirection fraud', 'DNS poisoning scam', 'fake banking site redirect'],
                'Free Trial Scams': ['free trial scam', 'subscription trap fraud', 'unwanted charges trial', 'credit card trial scam'],
                'Mandate Fraud': ['mandate fraud', 'invoice scam email', 'business email compromise BEC', 'payment redirection fraud', 'supplier invoice scam'],
                'Cryptocurrency Scams': ['cryptocurrency scam', 'crypto investment fraud', 'fake crypto scheme', 'bitcoin scam', 'altcoin fraud'],
                'General Holiday Scams': ['holiday scam', 'holiday fraud', 'Eid fraud', 'Eid scam', 'Christmas scam', 'Black Friday fraud', 'seasonal scam', 'fake charity holiday'],
                'Malware Attacks': ['malware attack', 'ransomware attack', 'data encryption scam', 'keylogger fraud', 'trojan data theft', 'spyware data breach', 'malicious software data', 'AI malware'],
                'Social Engineering': ['social engineering attack', 'pretexting scam', 'baiting fraud', 'quizzes survey scam', 'psychological manipulation cyber', 'human hacking'],
                'Software Exploits and Vulnerabilities': ['software exploit', 'vulnerability exploitation', 'unpatched software attack', 'zero-day exploit', 'privilege escalation', 'remote code execution', 'system compromise'],
                'Insider Threats': ['insider threat', 'employee data theft', 'privileged access abuse', 'data compromise insider', 'disgruntled employee data'],
                'Payment Card Fraud (Skimming)': ['payment card fraud', 'credit card skimming', 'ATM skimmer', 'POS fraud', 'card data theft'],
                'Physical Theft (Devices, Documents)': ['physical data theft', 'laptop theft data breach', 'stolen hard drive data', 'document theft sensitive data'],
                'Unintended Disclosure / Human Error': ['unintended data disclosure', 'human error data breach', 'negligence data leak', 'accidental data exposure', 'misconfigured system data'],
                'Credential Theft': ['credential theft', 'weak password hack', 'stolen login info', 'dictionary attack', 'brute force attack', 'password compromise', 'account credential theft'],
                'Eavesdropping': ['network eavesdropping', 'unencrypted traffic interception', 'data sniffing', 'packet capture data theft']
            }
        self.news_apis = st.session_state.api_keys
        self.fraud_keywords = st.session_state.fraud_keywords

    def save_api_keys(self, newsapi_key, newsdata_key, gnews_key):
        """Save API keys to session state"""
        st.session_state.api_keys = {
            'newsapi': newsapi_key,
            'newsdata': newsdata_key,
            'gnews': gnews_key
        }
        self.news_apis = st.session_state.api_keys

    def add_fraud_type(self, fraud_type, keywords):
        """Add new fraud type with keywords"""
        if fraud_type and keywords:
            keyword_list = [kw.strip().lower() for kw in keywords.split(',') if kw.strip()]
            if keyword_list:
                st.session_state.fraud_keywords[fraud_type] = keyword_list
                self.fraud_keywords = st.session_state.fraud_keywords
                return True
        return False

    def remove_fraud_type(self, fraud_type):
        """Remove fraud type"""
        if fraud_type in st.session_state.fraud_keywords:
            del st.session_state.fraud_keywords[fraud_type]
            self.fraud_keywords = st.session_state.fraud_keywords
            return True
        return False

    def fetch_news_from_newsapi(self, query, days=7):
        """Fetch news from NewsAPI"""
        url = "https://newsapi.org/v2/everything"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        params = {
            'q': query,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': self.news_apis['newsapi']
        }
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json().get('articles', [])
        except Exception as e:
            st.error(f"Error fetching from NewsAPI: {str(e)}")
        return []

    def fetch_news_from_newsdata(self, query, days=7):
        """Fetch news from NewsData API"""
        url = "https://newsdata.io/api/1/news"
        params = {
            'apikey': self.news_apis['newsdata'],
            'q': query,
            'language': 'en',
            'category': 'technology'
        }
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json().get('results', [])
        except Exception as e:
            st.error(f"Error fetching from NewsData API: {str(e)}")
        return []

    def fetch_news_from_gnews(self, query, days=7):
        """Fetch news from GNews API"""
        url = "https://gnews.io/api/v4/search"
        params = {
            'q': query,
            'token': self.news_apis['gnews'],
            'lang': 'en',
            'max': 100
        }
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json().get('articles', [])
        except Exception as e:
            st.error(f"Error fetching from GNews API: {str(e)}")
        return []

    def verify_url(self, url):
        """Verify if URL is valid and accessible"""
        if not url or url.strip() == '':
            return False, "Empty URL"
        # Basic URL format validation
        if not validators.url(url):
            return False, "Invalid URL format"
        # Parse URL to check components
        parsed = urlparse(url)
        # Check for suspicious patterns (Note: This is a basic check)
        suspicious_patterns = [
            'bit.ly', 'tinyurl.com', 'goo.gl', 't.co',  # URL shorteners
            'localhost', '127.0.0.1', '0.0.0.0',  # Local addresses
            '.onion',  # Tor hidden services
        ]
        for pattern in suspicious_patterns:
            if pattern in url.lower():
                return False, f"Suspicious URL pattern: {pattern}"
        
        # Check for legitimate news domains (This list can be expanded)
        legitimate_domains = [
            'reuters.com', 'bbc.com', 'cnn.com', 'ap.org', 'bloomberg.com',
            'wsj.com', 'ft.com', 'guardian.co.uk', 'theguardian.com',
            'nytimes.com', 'washingtonpost.com', 'usatoday.com',
            'abcnews.go.com', 'cbsnews.com', 'nbcnews.com', 'foxnews.com',
            'techcrunch.com', 'wired.com', 'arstechnica.com', 'zdnet.com',
            'csoonline.com', 'securityweek.com', 'darkreading.com',
            'infosecurity-magazine.com', 'scmagazine.com', 'helpnetsecurity.com'
        ]
        domain = parsed.netloc.lower()
        is_legitimate_domain = any(legit_domain in domain for legit_domain in legitimate_domains)
        
        try:
            # Test URL accessibility with timeout
            response = requests.head(url, timeout=10, allow_redirects=True)
            # Check if URL is accessible
            if response.status_code >= 400:
                return False, f"URL not accessible (Status: {response.status_code})"
            # Additional checks for content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type and 'application/json' not in content_type:
                return False, "URL does not point to web content"
            return True, "Valid URL" if is_legitimate_domain else "Valid URL (unverified domain)"
        except requests.exceptions.RequestException as e:
            return False, f"URL verification failed: {str(e)}"

    def verify_article_authenticity(self, article):
        """Comprehensive article authenticity verification"""
        authenticity_score = 0
        issues = []
        # Check URL validity
        url_valid, url_message = self.verify_url(article.get('url', ''))
        if not url_valid:
            issues.append(f"URL Issue: {url_message}")
        else:
            authenticity_score += 30
            if "unverified domain" not in url_message:
                authenticity_score += 20

        # Check if title exists and is meaningful
        title = article.get('title', '').strip()
        if not title or len(title) < 10:
            issues.append("Title too short or missing")
        else:
            authenticity_score += 20

        # Check if description exists
        description = article.get('description', '').strip()
        if not description or len(description) < 20:
            issues.append("Description too short or missing")
        else:
            authenticity_score += 15

        # Check for publication date
        pub_date = article.get('published_at', '') or article.get('publishedAt', '')
        if not pub_date:
            issues.append("Missing publication date")
        else:
            authenticity_score += 10

        # Check for source information
        source = article.get('source', '')
        if isinstance(source, dict):
            source_name = source.get('name', '')
        else:
            source_name = str(source)
        if not source_name or source_name.strip() == '':
            issues.append("Missing source information")
        else:
            authenticity_score += 5

        # Determine status based on score and issues
        # Adjusted thresholds for better categorization
        if authenticity_score >= 70 and len(issues) <= 1:
            status = 'authentic'
        elif authenticity_score >= 40 or len(issues) <= 3: # More lenient for suspicious
            status = 'suspicious'
        else:
            status = 'rejected'

        # Return verification result
        return {
            'authenticity_score': authenticity_score,
            'issues': issues,
            'url_valid': url_valid,
            'status': status # Add status
        }

    def classify_fraud_type(self, text):
        """Classify the type of fraud based on managed keywords only"""
        text_lower = text.lower()
        detected_types = []
        for fraud_type, keywords in self.fraud_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_types.append(fraud_type)
                    break
        # Only return detected types if any are found - no default "General Telecom Fraud"
        return detected_types

    def get_search_queries(self):
        """Generate search queries based on managed fraud types"""
        queries = []
        # Create search queries from fraud type keywords
        for fraud_type, keywords in self.fraud_keywords.items():
            # Use the first few keywords from each fraud type as search terms
            primary_keywords = keywords[:3]  # Use top 3 keywords per fraud type
            for keyword in primary_keywords:
                queries.append(keyword)
        return list(set(queries))  # Remove duplicates

    def export_to_csv(self, articles):
        """Export articles to CSV with title, fraud types, URL, and verification status"""
        export_data = []
        for article in articles:
            fraud_types_str = ', '.join(article['fraud_types']) if article.get('fraud_types') else 'None'
            verification_status = article.get('verification', {}).get('status', 'unknown')
            export_data.append({
                'Title': article['title'],
                'Fraud Types': fraud_types_str,
                'URL': article['url'],
                'Source': article['source'],
                'Published Date': article['published_at'],
                'Verification Status': verification_status
            })
        df = pd.DataFrame(export_data)
        # Convert to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()
        return csv_string

    def process_news_data(self, all_articles):
        """Process and standardize news data from different APIs with authenticity verification"""
        processed_articles = []
        verification_stats = {'authentic': 0, 'suspicious': 0, 'rejected': 0}
        
        for article in all_articles:
            # Standardize article structure
            processed_article = {
                'title': article.get('title', ''),
                'description': article.get('description', '') or article.get('content', ''),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', '') or article.get('pubDate', ''),
                'source': article.get('source', {}).get('name', '') if isinstance(article.get('source'), dict) else str(article.get('source', '')),
                'urlToImage': article.get('urlToImage', '') or article.get('image_url', '')
            }
            
            # Verify article authenticity
            verification_result = self.verify_article_authenticity(processed_article)
            processed_article['verification'] = verification_result
            
            # Classify fraud types (only managed types, no default)
            content_for_analysis = f"{processed_article['title']} {processed_article['description']}"
            detected_fraud_types = self.classify_fraud_type(content_for_analysis)
            
            # Only include articles that match managed fraud types
            if detected_fraud_types:
                processed_article['fraud_types'] = detected_fraud_types
                # Extract sentiment
                blob = TextBlob(content_for_analysis)
                processed_article['sentiment'] = blob.sentiment.polarity
                
                processed_articles.append(processed_article)
                verification_stats[verification_result['status']] += 1
            else:
                 # Even if no fraud type matched, we still process for completeness if it had content
                 # But for this specific dashboard, we filter by fraud type keywords, so skipping is fine.
                 # If you wanted ALL fetched articles, you'd append here regardless of fraud type match.
                 # For now, we keep the original logic of filtering by fraud type keywords.
                 pass

        # Store verification stats for display
        st.session_state.verification_stats = verification_stats
        return processed_articles # Return ALL processed articles that matched fraud keywords

def main():
    st.markdown('<h1 class="main-header">🔍 Global Telecom Fraud News Dashboard</h1>', unsafe_allow_html=True)
    
    dashboard = TelecomFraudDashboard()

    # Sidebar for controls
    st.sidebar.header("Dashboard Controls")

    # API Key inputs
    st.sidebar.subheader("API Configuration")
    # Display current API key status
    newsapi_status = "✅ Configured" if st.session_state.api_keys['newsapi'] else "❌ Not configured"
    newsdata_status = "✅ Configured" if st.session_state.api_keys['newsdata'] else "❌ Not configured"
    gnews_status = "✅ Configured" if st.session_state.api_keys['gnews'] else "❌ Not configured"
    
    st.sidebar.write(f"NewsAPI: {newsapi_status}")
    st.sidebar.write(f"NewsData API: {newsdata_status}")
    st.sidebar.write(f"GNews API: {gnews_status}")

    with st.sidebar.expander("🔑 Manage API Keys"):
        newsapi_key = st.text_input("NewsAPI Key", 
                                   value=st.session_state.api_keys['newsapi'], 
                                   type="password", 
                                   help="Get your free key from newsapi.org")
        newsdata_key = st.text_input("NewsData API Key", 
                                    value=st.session_state.api_keys['newsdata'], 
                                    type="password", 
                                    help="Get your free key from newsdata.io")
        gnews_key = st.text_input("GNews API Key", 
                                 value=st.session_state.api_keys['gnews'], 
                                 type="password", 
                                 help="Get your free key from gnews.io")
        
        if st.button("💾 Save API Keys"):
            dashboard.save_api_keys(newsapi_key, newsdata_key, gnews_key)
            st.success("API Keys saved successfully!")
            st.rerun()

    # Fraud Types Management
    st.sidebar.subheader("🎯 Fraud Types Management")
    with st.sidebar.expander("Add New Fraud Type"):
        new_fraud_type = st.text_input("Fraud Type Name", placeholder="e.g., AI Voice Cloning")
        new_keywords = st.text_input("Keywords (comma-separated)", 
                                   placeholder="e.g., ai voice, voice cloning, deepfake voice")
        if st.button("➕ Add Fraud Type"):
            if dashboard.add_fraud_type(new_fraud_type, new_keywords):
                st.success(f"Added fraud type: {new_fraud_type}")
                st.rerun()
            else:
                st.error("Please provide both fraud type name and keywords")

    with st.sidebar.expander("Manage Existing Types"):
        st.write("**Current Fraud Types:**")
        for fraud_type, keywords in st.session_state.fraud_keywords.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{fraud_type}**")
                st.caption(f"Keywords: {', '.join(keywords[:3])}{'...' if len(keywords) > 3 else ''}")
            with col2:
                if st.button("🗑️", key=f"delete_{fraud_type}", help=f"Delete {fraud_type}"):
                    if dashboard.remove_fraud_type(fraud_type):
                        st.success(f"Removed {fraud_type}")
                        st.rerun()

    # Search parameters
    st.sidebar.subheader("Search Parameters")
    days_back = st.sidebar.slider("Days to look back", 1, 30, 7)

    # Fetch data button
    if st.sidebar.button("🔄 Fetch Latest News", type="primary"):
        with st.spinner("Fetching news from multiple sources..."):
            all_articles = []
            # Generate search queries from managed fraud types
            queries = dashboard.get_search_queries()
            if not queries:
                st.error("No fraud types configured. Please add fraud types in the sidebar.")
                return
                
            progress_bar = st.progress(0)
            total_queries = min(len(queries), 20) # Limit to 20 queries for performance
            for i, query in enumerate(queries[:total_queries]):
                # Fetch from NewsAPI
                if dashboard.news_apis['newsapi']:
                    articles_newsapi = dashboard.fetch_news_from_newsapi(query, days_back)
                    all_articles.extend(articles_newsapi)
                    
                # Fetch from NewsData
                if dashboard.news_apis['newsdata']:
                    articles_newsdata = dashboard.fetch_news_from_newsdata(query, days_back)
                    all_articles.extend(articles_newsdata)
                    
                # Fetch from GNews
                if dashboard.news_apis['gnews']:
                    articles_gnews = dashboard.fetch_news_from_gnews(query, days_back)
                    all_articles.extend(articles_gnews)
                    
                progress_bar.progress((i + 1) / total_queries)
                time.sleep(0.5)  # Rate limiting

            # Process the data (ALL articles that match fraud keywords are now processed)
            if all_articles:
                processed_articles = dashboard.process_news_data(all_articles)
                
                # Remove duplicates based on title similarity
                unique_articles = []
                seen_titles = set()
                for article in processed_articles:
                    title_words = set(article['title'].lower().split())
                    is_duplicate = False
                    for seen_title in seen_titles:
                        # Calculate Jaccard similarity
                        intersection = title_words.intersection(set(seen_title.split()))
                        union = title_words.union(set(seen_title.split()))
                        if union and len(intersection) / len(union) > 0.7:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_articles.append(article)
                        seen_titles.add(article['title'].lower())
                
                st.session_state.articles = unique_articles
                st.success(f"Successfully fetched {len(unique_articles)} unique articles matching managed fraud types!")
            else:
                st.warning("No articles found. Please check your API keys and try again.")

    # Display dashboard if data is available
    if 'articles' in st.session_state and st.session_state.articles:
        articles = st.session_state.articles

        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Articles", len(articles))
        with col2:
            fraud_types = [fraud_type for article in articles for fraud_type in article.get('fraud_types', [])]
            st.metric("Fraud Types Detected", len(set(fraud_types)))
        with col3:
            sources = set([article['source'] for article in articles if article['source']])
            st.metric("News Sources", len(sources))
        with col4:
            # Filtered articles for sentiment calculation (only those with sentiment)
            articles_with_sentiment = [a for a in articles if 'sentiment' in a]
            if articles_with_sentiment:
                avg_sentiment = np.mean([article['sentiment'] for article in articles_with_sentiment])
                sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
            else:
                avg_sentiment = 0
                sentiment_label = "N/A"
            st.metric("Avg Sentiment", sentiment_label)
        with col5:
            if 'verification_stats' in st.session_state:
                stats = st.session_state.verification_stats
                st.metric("Authentic Articles", stats['authentic'])

        # Export functionality
        st.header("📤 Export Data")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"Export {len(articles)} articles to CSV format (includes verification status)")
        with col2:
            csv_data = dashboard.export_to_csv(articles)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"telecom_fraud_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        # Sortable news list
        st.header("📰 Latest News Articles")
        
        # Sorting options
        sort_options = ['Published Date (Newest)', 'Published Date (Oldest)', 'Title (A-Z)', 'Source']
        sort_by = st.selectbox("Sort by:", sort_options)

        # Filter options
        st.subheader("Filter Articles")
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_fraud_types = st.multiselect(
                "Filter by Fraud Type:", 
                options=list(set([fraud_type for article in articles for fraud_type in article.get('fraud_types', [])])),
                default=[]
            )
        with col2:
            selected_sources = st.multiselect(
                "Filter by Source:",
                options=list(set([article['source'] for article in articles if article['source']])),
                default=[]
            )
        with col3:
            # Verification Status Filter
            verification_status_options = ['authentic', 'suspicious', 'rejected']
            selected_verification_statuses = st.multiselect(
                "Filter by Verification Status:",
                options=verification_status_options,
                default=verification_status_options, # Default to showing all
                format_func=lambda x: f"Show {x.capitalize()}"
            )
            st.markdown('<p class="status-filter-info">Select which article verification statuses to display.</p>', unsafe_allow_html=True)

        # Apply filters
        filtered_articles = articles.copy()
        
        if selected_fraud_types:
            filtered_articles = [
                article for article in filtered_articles 
                if any(fraud_type in article.get('fraud_types', []) for fraud_type in selected_fraud_types)
            ]
        if selected_sources:
            filtered_articles = [
                article for article in filtered_articles 
                if article['source'] in selected_sources
            ]
        # Apply Verification Status Filter
        if selected_verification_statuses and len(selected_verification_statuses) < len(verification_status_options):
             filtered_articles = [
                 article for article in filtered_articles
                 if article.get('verification', {}).get('status', 'unknown') in selected_verification_statuses
             ]

        # Apply sorting
        if sort_by == 'Published Date (Newest)':
            filtered_articles.sort(key=lambda x: x['published_at'], reverse=True)
        elif sort_by == 'Published Date (Oldest)':
            filtered_articles.sort(key=lambda x: x['published_at'])
        elif sort_by == 'Title (A-Z)':
            filtered_articles.sort(key=lambda x: x['title'])
        elif sort_by == 'Source':
            filtered_articles.sort(key=lambda x: x['source'])

        # Display filtered articles
        st.write(f"Showing {len(filtered_articles)} articles (out of {len(articles)} total)")
        for i, article in enumerate(filtered_articles[:100]): # Limit display for performance
            verification = article.get('verification', {})
            verification_status = verification.get('status', 'unknown')
            url_status = "✅ Verified" if verification.get('url_valid', False) else "❌ Invalid URL"
            authenticity_score = verification.get('authenticity_score', 0)
            
            # Determine CSS class based on verification status
            card_class = f"news-card {verification_status}"

            # Create verification tag
            tag_class = f"verification-tag tag-{verification_status}"
            tag_text = verification_status.capitalize()

            st.markdown(f"""
            <div class="{card_class}">
                <h4>
                    <a href="{article['url']}" target="_blank" rel="noopener noreferrer">{article['title']}</a>
                    <span class="{tag_class}">{tag_text}</span>
                </h4>
                <p><strong>Source:</strong> {article['source']} | <strong>Published:</strong> {article['published_at']} | <strong>URL Status:</strong> {url_status} | <strong>Auth Score:</strong> {authenticity_score}/100</p>
                <p>{article['description'][:300]}{'...' if len(article['description']) > 300 else ''}</p>
                <div>
                    {''.join([f'<span class="fraud-type">{fraud_type}</span>' for fraud_type in article.get('fraud_types', [])])}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👆 Please configure your API keys in the sidebar and click 'Fetch Latest News' to start!")
        
        # Display current fraud types for reference
        st.subheader("📋 Currently Configured Fraud Types")
        fraud_types_data = []
        for fraud_type, keywords in st.session_state.fraud_keywords.items():
            fraud_types_data.append({
                'Fraud Type': fraud_type,
                'Keywords': ', '.join(keywords[:5]) + ('...' if len(keywords) > 5 else ''),
                'Total Keywords': len(keywords)
            })
        df_fraud_types = pd.DataFrame(fraud_types_data)
        st.dataframe(df_fraud_types, use_container_width=True)
        
        # Display sample data for demo
        st.subheader("Dashboard Features")
        st.write("""
        This dashboard provides:
        1. **🔑 API Key Management** - Store and manage your news API keys securely
        2. **🎯 Custom Fraud Types** - Add new fraud types and keywords dynamically
        3. **🔍 Real-time News Aggregation** - Fetch news from NewsAPI, NewsData API, and GNews API
        4. **✅ Link Verification** - Automatically verify article URLs and mark articles as Authentic, Suspicious, or Rejected
        5. **📤 CSV Export** - Export news titles, fraud types, URLs, and verification status to CSV format
        6. **🔧 Sorting and Filtering** - Organize news by date, source, fraud type, or verification status
        7. **📱 Responsive Interface** - Clean, modern web interface with visual indicators
        
        **How to get started:**
        1. Click "🔑 Manage API Keys" in the sidebar to add your API keys
        2. Optionally add custom fraud types using "🎯 Fraud Types Management"
        3. Click "🔄 Fetch Latest News" to start aggregating news
        4. Use the filters and "📥 Download CSV" button to refine and export your results
        
        **Note:** Only articles matching your managed fraud types will be displayed. Articles are now marked based on verification status and can be filtered.
        """)
        
        # API information
        with st.expander("ℹ️ API Information"):
            st.write("""
            **Free API Keys Available:**
            - **NewsAPI** (newsapi.org): 1,000 requests/month free
            - **NewsData API** (newsdata.io): 200 requests/day free  
            - **GNews API** (gnews.io): 100 requests/day free
            
            All APIs provide comprehensive news coverage with different strengths and coverage areas.
            """)

    # Footer
    st.markdown("---")
    st.markdown("**Global Telecom Fraud News Dashboard** - Stay informed about the latest telecom security threats worldwide.")

if __name__ == "__main__":
    main()
