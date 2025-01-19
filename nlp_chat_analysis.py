import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx

# File path to your extracted CSV file
CSV_FILE_PATH = 'chat_log.csv'

# Expanded sensory word lists
visual_words = [
    'see', 'look', 'clear', 'bright', 'picture', 'view', 'focus', 'appear', 'colorful',
    'observe', 'imagine', 'glimpse', 'illustrate', 'envision', 'clarify', 'perspective',
    'shape', 'outline', 'illuminate', 'mirror', 'reflect', 'shadow', 'sparkle', 'horizon',
    'vivid', 'pattern', 'frame', 'exhibit', 'display', 'scene', 'landscape'
]

auditory_words = [
    'hear', 'listen', 'sound', 'loud', 'quiet', 'speak', 'say', 'music', 'noisy', 'echo',
    'resonate', 'ring', 'tone', 'melody', 'harmony', 'whisper', 'roar', 'chatter', 'murmur',
    'dialogue', 'converse', 'announce', 'hum', 'tune', 'articulate', 'phonics', 'utter',
    'shout', 'yell', 'volume', 'rhythm', 'tempo', 'cadence', 'silence', 'vocalize'
]

kinaesthetic_words = [
    'feel', 'touch', 'grasp', 'warm', 'soft', 'smooth', 'supported', 'relax', 'hold',
    'carry', 'press', 'firm', 'steady', 'hug', 'stretch', 'ease', 'tense', 'tactile',
    'tangible', 'brittle', 'cushioned', 'gritty', 'sturdy', 'dense', 'glide', 'scrape',
    'gentle', 'massage', 'soothe', 'balanced', 'push', 'shove', 'shake', 'shift', 'brace'
]

auditory_digital_words = [
    'logic', 'reason', 'understand', 'analyze', 'system', 'know', 'learn', 'sense',
    'consider', 'to sum up', 'due diligence', 'make sense', 'think', 'assess', 'deduce',
    'evaluate', 'examine', 'hypothesize', 'reflect', 'deliberate', 'calculate', 'measure',
    'prioritize', 'identify', 'synthesize', 'conclude', 'decide', 'sequence', 'solve',
    'diagnose', 'detail', 'determine', 'summarize', 'categorize', 'process', 'outcome',
    'formula', 'clarify', 'logic-driven'
]

# 1. Analyze sensory language
def analyze_sensory_words(df):
    def count_words(text, word_list):
        return sum(1 for word in word_list if word in text.lower())
    
    df['Message'] = df['Message'].fillna('').astype(str)
    df['Visual_Count'] = df['Message'].apply(lambda x: count_words(x, visual_words))
    df['Auditory_Count'] = df['Message'].apply(lambda x: count_words(x, auditory_words))
    df['Kinesthetic_Count'] = df['Message'].apply(lambda x: count_words(x, kinaesthetic_words))
    df['Auditory_Digital_Count'] = df['Message'].apply(lambda x: count_words(x, auditory_digital_words))
    return df

# 2. Perform sentiment analysis
def analyze_sentiment(df):
    def get_sentiment(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    
    df['Sentiment'] = df['Message'].apply(lambda x: get_sentiment(x))
    return df

# 3. Generate topic modeling
def perform_topic_modeling(df, num_topics=5):
    print("Performing topic modeling...")
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(df['Message'])
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)
    
    topics = {}
    for index, topic in enumerate(lda.components_):
        words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        topics[f'Topic {index + 1}'] = words
    print("Topic modeling complete!")
    return topics

# 4. Visualize word cloud
def generate_word_cloud(df):
    all_text = ' '.join(df['Message'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Messages')
    plt.show()

# 5. Analyze interaction patterns
def analyze_interactions(df):
    interactions = df.groupby(['Sender']).size().reset_index(name='Message_Count')
    interactions = interactions.sort_values(by='Message_Count', ascending=False)
    
    # Plot interaction frequencies
    plt.figure(figsize=(10, 6))
    plt.bar(interactions['Sender'], interactions['Message_Count'])
    plt.title('Message Counts by User')
    plt.xlabel('Sender')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45)
    plt.show()

# 6. Visualize sentiment trends
def visualize_sentiment(df):
    sentiment_trend = df.groupby(df['Timestamp'].str[:10])['Sentiment'].mean()
    plt.figure(figsize=(12, 6))
    plt.plot(sentiment_trend, marker='o')
    plt.title('Average Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment')
    plt.grid(True)
    plt.show()

# Main function
def main():
    print("Loading data...")
    df = pd.read_csv(CSV_FILE_PATH)
    print("Data loaded!")
    
    print("Analyzing sensory language...")
    df = analyze_sensory_words(df)
    
    print("Performing sentiment analysis...")
    df = analyze_sentiment(df)
    
    print("Generating word cloud...")
    generate_word_cloud(df)
    
    print("Analyzing interactions...")
    analyze_interactions(df)
    
    print("Visualizing sentiment trends...")
    visualize_sentiment(df)
    
    print("Performing topic modeling...")
    topics = perform_topic_modeling(df)
    for topic, words in topics.items():
        print(f"{topic}: {', '.join(words)}")
    
    # Save results
    output_file = 'chat_analysis_results.csv'
    df.to_csv(output_file, index=False)
    print(f"Analysis complete! Results saved to {output_file}")

if __name__ == "__main__":
    main()
