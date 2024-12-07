import json

import re

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



nltk.download('punkt')

nltk.download('stopwords')



stop_words = set(stopwords.words('english'))



def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"[^\w\s@./~]", "", text)

    tokens = word_tokenize(text)

    tokens = [

        word for word in tokens 

        if (word not in stop_words or '@' in word or '.' in word)

    ]

    preprocessed_text = " ".join(tokens)

    return preprocessed_text



def flatten_and_label_json(data, parent_key=''):

    flat_text = ""

    if isinstance(data, dict):

        for key, value in data.items():

            full_key = f"{parent_key} {key}".strip()

            flat_text += flatten_and_label_json(value, full_key)

    elif isinstance(data, list):

        for item in data:

            flat_text += flatten_and_label_json(item, parent_key)

    elif isinstance(data, str):

        labeled_text = f"{parent_key}: {data}"

        flat_text += labeled_text + " | "

    return flat_text



def preprocess_entire_page(file_path, max_chunk_size=512):

    with open(file_path, 'r') as file:

        data = json.load(file)

    flattened_text = flatten_and_label_json(data)

    preprocessed_text = preprocess_text(flattened_text)

    text_chunks = [

        preprocessed_text[i:i+max_chunk_size]

        for i in range(0, len(preprocessed_text), max_chunk_size)

    ]

    return text_chunks



if __name__ == "__main__":

    files = [

        './dataset/computer_science_department_labs_research_centers.json',

        './dataset/computer_science_department_research_areas.json',

        './dataset/computer_science_primary_course_list.json',

        './dataset/cs_contact_info.json',

        './dataset/cs_student_orgs.json',

        './dataset/cs_alumni.json',

        './dataset/cs_faculty_contact.json',

        './dataset/cs_courses_by_area.json',

        './dataset/ms_cs_degree_requirements.json',

        './dataset/nlp_courses.json',

        './dataset/nlp_members.json',

        './dataset/nlp_projects.json',

        './dataset/nlp_publications.json'

    ]



    all_chunks = []

    for file_path in files:

        chunks = preprocess_entire_page(file_path)

        all_chunks.extend(chunks)



    with open('preprocessed_data.json', 'w') as outfile:

        json.dump(all_chunks, outfile)


