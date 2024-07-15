import pandas as pd
import numpy as np
import random
import openai
from googletrans import Translator

# Load datasets with low_memory=False to avoid dtype warning
df1 = pd.read_csv('E:/EMA Project/New folder/Leads.csv', encoding='iso-8859-1', low_memory=False)
df2 = pd.read_csv('E:/EMA Project/New folder/SampleData.csv', encoding='iso-8859-1', low_memory=False)

# Merge datasets on common columns
merged_df = pd.merge(df1, df2, on=['Prospect ID', 'Lead Number'], how='outer')

# Remove columns with all null values
merged_df.dropna(axis=1, how='all', inplace=True)

# Add random language column
languages = ['English', 'Spanish', 'French', 'German', 'Chinese']
merged_df['Language'] = [random.choice(languages) for _ in range(len(merged_df))]

# Add random email column
email = ['ram@gmail.com', 'Sam@yahoo.com', 'han@baidu.com', 'fox@gmail.com', 'john@outlook.com']
merged_df['email'] = [random.choice(email) for _ in range(len(merged_df))]

# Placeholder for UserEmail column (User input will be taken later)
merged_df['UserEmail'] = ''

# Save combined data to CSV
merged_df.to_csv('E:/EMA Project/New folder/combined_leads.csv', index=False)

# Query leads based on specific criteria
filtered_leads = merged_df[merged_df['Company'].notna()]

# Save filtered leads to CSV
filtered_leads.to_csv('E:/EMA Project/New folder/filtered_leads.csv', index=False)

# Initialize OpenAI API 
openai.api_key = 'sk-proj-kgekVUbQnCNDcC97iAhZT3BlbkFJgs6vHZ3ZJF2lcJ96q92E'

# Language mapping
language_mapping = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Chinese': 'zh-cn'  # Ensure correct code for Chinese
}

def generate_email(first_name, last_name, product, language, sender_name, sender_position, sender_contact):
    prompt = f"""
    Write a sales email to {first_name} {last_name} about our product {product}.
    
    Make sure to include:
    - Dear {first_name} {last_name},
    - Sincerely, {sender_name}
    - Position: {sender_position}
    - Contact: {sender_contact}
    """
    
    # Using the chat completion endpoint
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    email_content = response['choices'][0]['message']['content'].strip()
    
    # Translate email if necessary
    if language != 'English':
        email_content = translate_email(email_content, language)
    
    return email_content

def translate_email(email_content, target_language):
    # Use language_mapping to get the correct language code
    lang_code = language_mapping.get(target_language, None)
    
    if lang_code is None:
        raise ValueError(f"Invalid destination language: {target_language}")
    
    # Use Google Translate API to translate the email content
    translator = Translator()
    translated = translator.translate(email_content, dest=lang_code)
    translated_text = translated.text
    
    # Print translated email content
    print(f"\nTranslated email content ({target_language}):\n{translated_text}\n")
    
    return translated_text

def query_university_leads(df):
    return df[df['Company'].str.contains('University', case=False, na=False)]

def get_dataset_info(question, df):
    # Handle specific questions programmatically
    if "number of rows" in question.lower():
        return f"The number of rows in the dataset is {df.shape[0]}."
    if "number of columns" in question.lower():
        return f"The number of columns in the dataset is {df.shape[1]}."
    if "university" in question.lower():
        university_leads_count = query_university_leads(df).shape[0]
        return f"There are {university_leads_count} leads related to universities in the dataset."
        
    # Default to using OpenAI for other questions
    prompt = f"The dataset has the following columns: {', '.join(df.columns)}. Answer the following question based on the dataset: {question}"

    # Using the chat completion endpoint
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    answer = response['choices'][0]['message']['content'].strip()
    return answer

def count_leads_by_word(df, word):
    word = word.lower()
    count = 0
    
    for col in df.columns:
        count += df[col].astype(str).str.lower().str.contains(word, na=False).sum()
        
    return count

def count_word_occurrences(df, word):
    count = 0
    word = word.lower()
    for col in df.columns:
        count += df[col].astype(str).str.lower().str.contains(word, na=False).sum()
    return count

def main():
    # Load combined data
    combined_df = pd.read_csv('E:/EMA Project/New folder/combined_leads.csv')

    # Ensure 'UserEmail' column is of type string
    combined_df['UserEmail'] = combined_df['UserEmail'].astype(str)

    # Example of asking a question about the dataset before generating emails
    question = input("Ask a question about the dataset: ")
    answer = get_dataset_info(question, combined_df)
    print(f"Answer: {answer}")

    # Ask for a word to count occurrences in the entire dataset
    input_word = input("Enter the word to count occurrences in the dataset: ")
    word_count = count_word_occurrences(combined_df, input_word)
    print(f"The word '{input_word}' appears {word_count} times in the dataset.")

    # Generate emails for leads
    for index, row in combined_df.iterrows():
        # Prompt user for email address
        user_email = input(f"Enter email for {row.get('First Name', '')} {row.get('Last Name', '')}: ")
        combined_df.at[index, 'UserEmail'] = str(user_email)
        
        email_content = generate_email(
            first_name=row.get('First Name', ''),
            last_name=row.get('Last Name', ''),
            product='Ema',  # Replacing with actual product name
            language=row['Language'],
            sender_name='Your Name',  # Replacing with actual sender's name
            sender_position='Your Position',  # Replacing with my actual position
            sender_contact='Your Contact Information'  # Replacing with actual contact information
        )
        combined_df.at[index, 'GeneratedEmail'] = email_content

    # Query university leads
    university_leads = query_university_leads(combined_df)

    # Save results
    combined_df.to_csv('E:/EMA Project/New folder/leads_with_emails.csv', index=False)
    university_leads.to_csv('E:/EMA Project/New folder/university_leads.csv', index=False)

if __name__ == "__main__":
    main()
