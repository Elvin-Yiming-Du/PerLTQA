# PerLTQA Dataset Documentation

## Overview

The `PerLTQA` (Personal Long-Term Memory Question and Answering) dataset is a comprehensive collection of data simulating the long-term memory of virtual characters. This dataset includes detailed profiles, social relationships, significant life events, and dialogues (we defined profiles and social relationships as semantic memory and events and dialogues as episodic memory) generated by ChatGPT. It aims to provide a deep understanding of the long term memory in different character's life. We split the dataset into `PerLT_Mem` and `PerLT_QA`. `PerLT_Mem` stores all character's memory information. `PerLT_QA` consists of question, answer, reference memory and memory anchors.

## Data Structure

### A. PerLT_Mem  Dataset Structure

`PerLT_Mem` is structured into several main sections: `profile`, `profile_description`, `social_relationship`, `events`, and `dialogues`. Each section captures different aspects of the virtual character's life and interactions.

#### Profile

- **Protagonist**: The main character of the dataset.
- **Gender**: Gender of the protagonist.
- **Nickname**: Any nickname(s) the protagonist might have.
- **Title**: Professional title or occupation.
- **Age**: Age of the protagonist.
- **Occupation**: Detailed description of the protagonist's job.
- **Nationality**: The nationality of the protagonist.
- **Physical Characteristics**: Descriptive details about the protagonist's appearance.
- **Hobbies**: List of hobbies and interests.
- **Achievements**: Notable achievements.
- **Ethnic Background**: Ethnic background of the protagonist.
- **Education Background**: Education history.
- **Employer**: Name of the employer or the company the protagonist works for.
- **Awards and Role Models**: Inspirations and role models.

####  Profile Description

A comprehensive narrative that provides an overview of the protagonist's life, summarizing the details provided in the `profile` section.

####  Social Relationships

An array of entries detailing the protagonist's social circle, including family, friends, colleagues, and other significant relationships. Each entry includes:
- **Supporting Characters**: Name of the character.
- **Description**: Brief description of the relationship and interactions with the protagonist.
- **Relationship**: The nature of the relationship (e.g., friend, colleague, family).

####  Events

This section chronicles significant events in the protagonist's life, providing a timeline of personal experiences, achievements, and memorable moments. Each event entry includes:
- **Content**: Detailed narrative of the event.
- **Summary**: A brief summary of the event.
- **Characters**: Characters involved in the event.
- **Creation Time**: Date or period the event occurred.
- **Last Accessed Time**: The last time the event was referenced or recalled.
- **Theme**: The overarching theme or category of the event.

####  Dialogues

Recorded dialogues between the protagonist and AI or other characters, related to specific events. These dialogues are timestamped and provide insight into the protagonist's thoughts, emotions, and interactions.



### B. PerLT_QA Dataset Structure

The `PerLT_QA` is structured as a JSON file, where each key represents an individual's unique identifier, encapsulating their personal memory in a structured format. Below is the detailed structure and description of each component within the dataset.

Under each individual's identifier, the dataset contains `profile`, `social relationship`, `events`, `dialogues` list, which encompasses multiple memory records. Each record  is a structured object containing the following fields:

- **Question**: A string that poses a question related to the individual's personal memory or historical information.

- **Answer**: A string providing the answer to the corresponding question, utilizing the individual's personal memory.

- **Reference Memory**: The memory index which indicate one specific memory stored in `PerLT_Mem` dataset.  (e.g., "1_0_0").

- **Memory Anchors**: A list of objects that serve as pointers or references within the memory, helping to anchor the answer's context. Each object within the list maps a key phrase to its occurrence indices within the personal memory context.
#### Memory Anchors Structure

Each Memory Anchor is a key-value pair where:
- The **key** is a string representing a significant word or phrase in the answer or history memory..
- The **value** is a list of two integers, representing the start and end positions of the key phrase within the reference memory. A value of `-1` indicates that the specific location is not applicable or not specified.


## Dataset Usage

The `PerLTQA` dataset is designed for use in simulations, storytelling, and AI training, providing a rich narrative framework to explore character development, social dynamics, and personalized interactions within a virtual environment.

The `PerLT_Mem` can be used to explore the long-term memory usage in the dialogue system, agents and social networks.

You can load `PerLT_Mem` memory database via class `class: PerLTMem`.
```
# load PerLT_Mem dataset
dataset = PerLTMem("data_json_file")

# extract all the characters from json file.
character_names = dataset.extract_character_names()

# extract all the memory information from given character name.
samples = dataset.extract_sample(character_name)
```
The `PerLT_QA` dataset can be employed in a variety of research and application scenarios, including but not limited to:
- Enhancing conversational AI with personalized responses based on long-term memory.
- Studying memory anchoring mechanisms in the context of personalized information retrieval.

You can load `PerLT_QA` dataset via `class: PerLTQA`.
```
#initialize class PerLTQA.
dataset = PerLTQA()

#extract all data group by characters.
character_data = dataset.read_json_data('perltqa_dataset_file_name.json')

# extract all the memory-based qa records from given character name.
samples = dataset.extract_sample(character_name)
```



If there is any issues, please contact us.



## Citation

```
@article{du2024perltqa,
title={PerLTQA: A Personal Long-Term Memory Dataset for Memory Classification, Retrieval, and Synthesis in Question Answering},
author={Du, Yiming and Wang, Hongru and Zhao, Zhengyi and Liang, Bin and Wang, Baojun and Zhong, Wanjun and Wang, Zezhong and Wong, Kam-Fai},
journal={arXiv preprint arXiv:2402.16288},
year={2024}
}
```