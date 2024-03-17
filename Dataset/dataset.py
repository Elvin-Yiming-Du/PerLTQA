import json

# Class PerLTMem is a tool class to extract the personal memory including semantic_memory(profiles, social relationships) and episodic memory(events and dialogues).
class PerLTMem:
    def __init__(self):
        self.character = {}

    def read_json_data(self, file_name):
        with open(file_name, 'r', encoding="utf-8") as f:
            content = f.read()
            dataset = json.loads(content)

        for character in dataset:
            character_name = character["profile"]["Protagonist"]
            self.character[character_name] = character

        character_data = self.character
        return character_data

    def extract_character_names(self):
        return self.character.keys()

    def extract_sample(self, character_name):
        try:
            character_sample = self.character[character_name]

        except Exception as e:
            print(f"No such sample in AgentMem for {character_name}")
            character_sample = {}
        return character_sample

    def extract_events(self, character_name):
        try:
            character_episodic_memory = self.character[character_name]["events"]
        except Exception as e:
            print(e)
            print(f"Parse error for {character_name}")
            return None
        return character_episodic_memory

    def extract_social_relationships(self, character_name):
        try:
            character_sample = self.character[character_name]
        except Exception as e:
            print(e)
            print(f"Parse error for {character_name}")
            return None
        return character_sample["social_relationship"]

    def extract_profile(self, character_name):
        try:
            character_sample = self.character[character_name]
        except Exception as e:
            print(e)
            print(f"Parse error for {character_name}")
            return None
        return character_sample["profile"]

    def extract_profile_description(self, character_name):
        try:
            character_sample = self.character[character_name]
        except Exception as e:
            print(e)
            print(f"Parse error for {character_name}")
            return None
        return character_sample["profile_description"]

    def extract_dialogues(self, character_name):
        try:
            character_sample = self.character[character_name]
        except Exception as e:
            print(e)
            print(f"Parse error for {character_name}")
            return None
        return character_sample["dialogues"]

    def extract_social_relationship_by_id(self,character_name, id):
        try:
            character_semantic_memory = self.character[character_name]["social_relationship"][id]
        except Exception as e:
            print(e)
            print(f"Parse error for {character_name}")
            return None
        return character_semantic_memory

    def extract_events_by_id(self,character_name, id):
        try:
            character_episodic_memory = self.character[character_name]["events"][id]
        except Exception as e:
            print(e)
            print(f"Parse error for {character_name}")
            return None
        return character_episodic_memory

    def extract_dialogues_by_id(self, character_name, id):
        try:
            character_dialogues = self.character[character_name]["dialogues"][id]
        except Exception as e:
            print(e)
            print(f"Parse error for {character_name}")
            return None
        return character_dialogues

    def extract_relationships(self, character_name):
        try:
            character_sample = self.character[character_name]
        except Exception as e:
            print(e)
            print(f"Parse error for {character_name}")
            return None
        relationships = []
        for k,v in character_sample["social_relationship"].items():
            relationships.append(v["关系"])
        return relationships


# Class PerLTQA is a tool class used to extract the question, answers and memory ahchors.
class PerLTQA:
    def __init__(self):
        self.character = {}

    def read_json_data(self, file_name):
        with open(file_name, 'r', encoding="utf-8") as f:
            content = f.read()

            dataset = json.loads(content)
        for sample in dataset:
            for k, v in sample.items():
                self.character[k] = v
        character_data = self.character
        return character_data

    def extract_character_names(self):
        return self.character.keys()

    def extract_sample(self, character_name):
        try:
            character_sample = self.character[character_name]

        except Exception as e:
            print(f"No such sample in AgentMem for {character_name}")
            character_sample = {}
        return character_sample

    def extract_event_questions(self, character_name):
        try:
            character_episodic_memory_questions = self.character[character_name]["events"]
        except Exception as e:
            print(e)
            print(f"Parse error for {character_name}")
            return None
        return character_episodic_memory_questions

    def extract_social_relationship_questions(self, character_name):
        try:
            character_semantic_memory_questions = self.character[character_name]["social_relationship"]
        except Exception as e:
            print(e)
            print(f"Parse error for {character_name}")
            return None
        return character_semantic_memory_questions

    def extract_profile_questions(self, character_name):
        try:
            character_sample = self.character[character_name]
        except Exception as e:
            print(e)
            print(f"Parse error for {character_name}")
            return None
        return character_sample["profile"]

    def extract_dialogue_questions(self, character_name):
        try:
            character_sample = self.character[character_name]
        except Exception as e:
            print(e)
            print(f"Parse error for {character_name}")
            return None
        return character_sample["dialogues"]