from bs4 import BeautifulSoup
import pandas as pd

# File path to your HTML chat log
HTML_FILE_PATH = 'chatlog.html'

# Function to extract data from HTML
def extract_chat_data(file_path):
    print("Parsing HTML file...")
    
    # Read the HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    # Extract messages, timestamps, and senders
    messages = []
    for message_group in soup.find_all('div', class_='chatlog__message-group'):
        for message_container in message_group.find_all('div', class_='chatlog__message-container'):
            # Extract timestamp
            timestamp_element = message_container.find('span', class_='chatlog__timestamp')
            timestamp = timestamp_element.text.strip() if timestamp_element else None

            # Extract sender
            sender_element = message_container.find('span', class_='chatlog__author')
            sender = sender_element.text.strip() if sender_element else None

            # Extract message content
            content_element = message_container.find('div', class_='chatlog__content')
            content = content_element.text.strip() if content_element else None

            # Append extracted data to the list
            if timestamp and sender and content:
                messages.append({
                    'Timestamp': timestamp,
                    'Sender': sender,
                    'Message': content
                })
    
    print(f"Extracted {len(messages)} messages.")
    return messages

# Main function
def main():
    # Extract data from the HTML file
    chat_data = extract_chat_data(HTML_FILE_PATH)
    
    # Convert to a pandas DataFrame
    df = pd.DataFrame(chat_data)
    
    # Save to CSV for further analysis
    output_file = 'chat_log.csv'
    df.to_csv(output_file, index=False)
    print(f"Chat log saved to {output_file}")

# Run the script
if __name__ == "__main__":
    main()