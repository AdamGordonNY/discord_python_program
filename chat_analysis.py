import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Sensory word dictionaries
visual_words = ['see', 'look', 'clear', 'bright', 'picture', 'view', 'focus', 'appear', 'colorful']
auditory_words = ['hear', 'listen', 'sound', 'loud', 'quiet', 'speak', 'say', 'music', 'noisy']
kinaesthetic_words = ['feel', 'touch', 'grasp', 'warm', 'soft', 'smooth', 'supported', 'relax']
auditory_digital_words = ['logic', 'reason', 'understand', 'analyze', 'system', 'know', 'learn', 
                          'sense', 'consider', 'to sum up', 'due diligence', 'make sense', 'think']

# File path to the CSV chat log
FILE_PATH = 'chat_log.csv'

# Load the chat data
def load_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    print("Data loaded successfully!")
    return df

# Analyze sentiment
def analyze_sentiment(df):
    print("Performing sentiment analysis...")
    def get_sentiment(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    
    df['Sentiment'] = df['Message'].apply(get_sentiment)
    print("Sentiment analysis complete!")
    return df

# Analyze sensory language
def analyze_sensory_language(df):
    print("Analyzing sensory language...")
    def count_words(text, word_list):
        text = text.lower()
        return sum(1 for word in word_list if word in text)

    df['Visual'] = df['Message'].apply(lambda x: count_words(x, visual_words))
    df['Auditory'] = df['Message'].apply(lambda x: count_words(x, auditory_words))
    df['Kinaesthetic'] = df['Message'].apply(lambda x: count_words(x, kinaesthetic_words))
    df['Auditory Digital'] = df['Message'].apply(lambda x: count_words(x, auditory_digital_words))
    print("Sensory language analysis complete!")
    return df

# Visualize sentiment over time
def visualize_sentiment(df):
    print("Generating sentiment trend visualization...")
    sentiment_trend = df.groupby(df['Timestamp'].dt.date)['Sentiment'].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(sentiment_trend, marker='o')
    plt.title('Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment')
    plt.grid(True)
    plt.show()

# Visualize sensory language distribution
def visualize_sensory_distribution(df):
    print("Generating sensory language distribution visualization...")
    sensory_counts = df[['Visual', 'Auditory', 'Kinaesthetic', 'Auditory Digital']].sum()
    plt.figure(figsize=(8, 6))
    sensory_counts.plot(kind='bar')
    plt.title('Sensory Language Distribution')
    plt.xlabel('Sensory Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Main function
def main():
    df = load_data(FILE_PATH)
    df = analyze_sentiment(df)
    df = analyze_sensory_language(df)
    
    # Save analyzed data
    output_file = 'chat_analysis_with_sensory.csv'
    df.to_csv(output_file, index=False)
    print(f"Analysis results saved to {output_file}")
    
    # Visualize results
    visualize_sentiment(df)
    visualize_sensory_distribution(df)

if __name__ == "__main__":
    main()