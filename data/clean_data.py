#!/usr/bin/env python3

import pandas as pd
import re
import sys

# Function to clean text: convert to lowercase and remove unnecessary punctuation
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        # Remove all punctuation except alphanumeric characters and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Remove extra spaces
        text = ' '.join(text.split())
    return text

# Function to count "yta" and "nta" mentions in top comments
def tally_comments(row):
    yta_phrases = ['yta', 'youre the asshole', "you're the asshole"]
    nta_phrases = ['nta', 'not the asshole']

    yta_count = 0
    nta_count = 0

    for i in range(1, 11):  # Iterate over top_comment_1 to top_comment_10
        comment = row.get(f'top_comment_{i}', '')
        for phrase in yta_phrases:
            yta_count += comment.count(phrase)
        for phrase in nta_phrases:
            nta_count += comment.count(phrase)
    
    # Return the tallied counts
    return yta_count, nta_count

# Function to assign a verdict based on the tally or known verdict
def assign_verdict(row):
    if pd.notna(row['verdict']):  # Use the known verdict from aita_clean if available
        if 'everyone sucks' in row['verdict'].lower():
            return 'asshole'
        elif 'no assholes here' in row['verdict'].lower():
            return 'not the asshole'
        return row['verdict'].lower()
    else:
        # Use the tally to determine the verdict
        if row['yta_count'] > row['nta_count']:
            return 'asshole'
        elif row['nta_count'] > row['yta_count']:
            return 'not the asshole'
        else:
            return 'unknown'  # In case of a tie or no mentions

def main():
    # Load the datasets
    aita_clean = pd.read_csv(sys.argv[1])
    parquet_data = pd.read_parquet(sys.argv[2])
    parquet_data = parquet_data.loc[:, ~parquet_data.columns.str.contains('^Unnamed')]
    # Clean the text columns in the Parquet data
    parquet_data['submission_text'] = parquet_data['submission_text'].apply(clean_text)
    parquet_data['submission_title'] = parquet_data['submission_title'].apply(clean_text)
    
    # Clean all top_comment columns
    for i in range(1, 11):
        parquet_data[f'top_comment_{i}'] = parquet_data[f'top_comment_{i}'].apply(clean_text)
        
        # Merge with the aita_clean dataset (titles cleaned similarly to handle comparisons)
    aita_clean['title'] = aita_clean['title'].apply(clean_text)
    parquet_data['submission_title'] = parquet_data['submission_title'].apply(clean_text)

    # Merge datasets based on matching cleaned titles
    merged_data = pd.merge(parquet_data, aita_clean[['title', 'verdict']], left_on='submission_title', right_on='title', how='left')
    print(merged_data)

    # Apply tallying function to each row in merged_data
    merged_data[['yta_count', 'nta_count']] = merged_data.apply(tally_comments, axis=1, result_type='expand')

    # Add the 'verdict' column based on the logic
    merged_data['verdict'] = merged_data.apply(assign_verdict, axis=1)
    merged_data.drop(columns=['title'])
    merged_data.rename(columns={'submission_title': 'title', 'submission_text': 'body'}, inplace=True)

    # Drop unnecessary columns to get the final output
    # OLD version
    #final_data = merged_data[['title', 'body', 'yta_count', 'nta_count'] + [f'top_comment_{i}' for i in range(1, 11)] + ['verdict']] 
    final_data = merged_data[['title', 'body'] + [f'top_comment_{i}' for i in range(1, 11)] + ['verdict']]

    # Print or save the final result
    print(final_data)

    final_data.to_csv('cleaned_data.csv')
    

if __name__ == '__main__':
    main()
