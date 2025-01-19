import pandas as pd
import matplotlib.pyplot as plt

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

# Function to count sensory words
def analyze_sensory_words(df):
    print("Analyzing sensory language...")
    
    def count_words(text, word_list):
        return sum(1 for word in word_list if word in text.lower())
    
    # Ensure all Message values are strings and replace NaN with an empty string
    df['Message'] = df['Message'].fillna('').astype(str)
    
    # Create new columns with word counts for each category
    df['Visual_Count'] = df['Message'].apply(lambda x: count_words(x, visual_words))
    df['Auditory_Count'] = df['Message'].apply(lambda x: count_words(x, auditory_words))
    df['Kinesthetic_Count'] = df['Message'].apply(lambda x: count_words(x, kinaesthetic_words))
    df['Auditory_Digital_Count'] = df['Message'].apply(lambda x: count_words(x, auditory_digital_words))
    
    print("Sensory language analysis complete!")
    return df

# Function to visualize sensory word usage by user
def visualize_sensory_by_user(df):
    print("Visualizing sensory language by user...")
    
    # Group by user and sum the sensory counts
    user_sensory_counts = df.groupby('Sender').sum()[[
        'Visual_Count', 'Auditory_Count', 'Kinesthetic_Count', 'Auditory_Digital_Count'
    ]]
    
    # Plot the sensory word counts for each user
    user_sensory_counts.plot(kind='bar', figsize=(12, 6))
    plt.title('Sensory Language Usage by User')
    plt.xlabel('User')
    plt.ylabel('Word Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Sensory Category')
    plt.tight_layout()
    plt.show()

# Function to visualize overall sensory word distribution
def visualize_overall_sensory(df):
    print("Visualizing overall sensory word usage...")
    
    # Sum all sensory word counts
    total_sensory_counts = df[['Visual_Count', 'Auditory_Count', 'Kinesthetic_Count', 'Auditory_Digital_Count']].sum()
    
    # Plot pie chart of overall sensory word distribution
    total_sensory_counts.plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8), startangle=90)
    plt.title('Overall Sensory Word Distribution')
    plt.ylabel('')  # Remove the default y-axis label
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Load the CSV file
    print("Loading CSV file...")
    df = pd.read_csv(CSV_FILE_PATH)
    print("CSV file loaded successfully!")
    
    # Perform sensory language analysis
    df = analyze_sensory_words(df)
    
    # Save the analyzed data to a new CSV
    output_file = 'chat_analysis_with_sensory_counts.csv'
    df.to_csv(output_file, index=False)
    print(f"Analysis results saved to {output_file}")
    
    # Visualize sensory language usage
    visualize_sensory_by_user(df)
    visualize_overall_sensory(df)

# Run the script
if __name__ == "__main__":
    main()
