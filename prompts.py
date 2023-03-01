from typing import Any, Text, Dict
from dataclasses import dataclass

@dataclass
class SimpleTemplatePrompt:
    template: str
    args_order: list

    def __call__(self, **kwargs: Any) -> Text:
        args = [kwargs[arg] for arg in self.args_order]
        return self.template.format(*args)


@dataclass
class FewShotPrompt(SimpleTemplatePrompt):

    def __call__(self, positive_examples: list[Dict], negative_examples: list[Dict], **kwargs) -> Any:
        positive_examples = self._process_positive_examples(positive_examples)
        negative_examples = self._process_negative_examples(negative_examples)
        args = [kwargs[arg] for arg in self.args_order]
        return self.template.format(positive_examples, negative_examples, *args)
    
    def _process_positive_examples(self, positive_examples: list) -> Text:
        output = ""
        for n, example in enumerate(positive_examples):
            output += f"Positive example {n}:\n" + \
                      f"input: {example['input']}\n" + \
                      f"output: {example['output']}\n"
        return output
    
    def _process_negative_examples(self, negative_examples: list) -> Text:
        output = ""
        for n, example in enumerate(negative_examples):
            output += f"Negative example {n}:\n" + \
                      f"input: {example['input']}\n" + \
                      f"output: {example['output']}\n"
        return output


STATE_PROMPTS = {
    'restaurant': FewShotPrompt(template="""
Definition: Capture values from converstation in JSON according to examples.
Values that should be captured are:
 - "pricerange" that specifies the price range of the restaurant (cheap/moderate/expensive)
 - "area" that specifies the area where the restaurant is located (north/east/west/south/centre)
 - "food" that specifies the type of food the restaurant serves
If not specified, leave the value empty.
{}{}Now complete the following example:
input: {}
Customer: {}
output: state:""",
                                    args_order=["history", "utterance"]),
    'hotel': FewShotPrompt(template="""
Definition: Capture values from converstation in JSON according to examples.
Values that should be captured are:
 - "area" that specifies the area where the hotel is located (north/east/west/south/centre)
 - "internet" that specifies if the hotel has internet (yes/no)
 - "parking" that specifies if the hotel has parking (yes/no)
 - "stars" that specifies the number of stars the hotel has (1/2/3/4/5)
 - "type" that specifies the type of the hotel (hotel/bed and breakfast/guest house)
 - "pricerange" that specifies the price range of the hotel (cheap/expensive)
If not specified, leave the value empty.
{}{}Now complete the following example:
input: {}
Customer: {}
output: state:""",
                                    args_order=["history", "utterance"]),
    'train': FewShotPrompt(template="""
Definition: Capture values from converstation in JSON according to examples.
Values that should be captured are:
 - "arriveBy" that specifies what time the train should arrive
 - "leaveAt" that specifies what time the train should leave
 - "day" that specifies what day the train should leave (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - "departure" that specifies the departure station
 - "destination" that specifies the destination station
If not specified, leave the value empty.
{}{}Now complete the following example:
input: {}
Customer: {}
output: state:""",
                                    args_order=["history", "utterance"]),
    'taxi': FewShotPrompt(template="""
Definition: Capture values from converstation in JSON according to examples.
If not specified, leave the value empty.
{}{}Now complete the following example:
input: {}
Customer: {}
output: state:""",
                                    args_order=["history", "utterance"]),
    'attraction': FewShotPrompt(template="""
Definition: Capture values from converstation in JSON according to examples.
Values that should be captured are:
 - "type" that specifies the type of attraction (museum/gallery/theatre/concert/stadium)
 - "area" that specifies the area where the attraction is located (north/east/west/south/centre)
If not specified, leave the value empty.
{}{}Now complete the following example:
input: {}
Customer: {}
output: state:""",
                                    args_order=["history", "utterance"]),
}

RESPONSE_PROMPTS = {
    'restaurant': FewShotPrompt(template="""
Definition: You are an assistant that helps people to book a restaurant.
You can search for a restaurant by area, food, or price.
There is also a number of restaurants in the database currently corresponding to the user's request.
If you find a restaurant, provide [restaurant_name], [restaurant_address], [restaurant_phone] or [restaurant_postcode] if asked.
If booking, provide [reference] in the answer.
{}{}Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"]),
    'hotel': FewShotPrompt(template="""
    Definition: You are an assistant that helps people to book a hotel.
The customer can ask for a hotel by name, area, parking, internet availability, or price.
There is also a number of hotel in the database currently corresponding to the user's request.
If you find a hotel, provide [hotel_name], [hotel_address], [hotel_phone] or [hotel_postcode] if asked.
If booking, provide [reference] in the answer.
{}{}Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"]),
    'train': FewShotPrompt(template="""
    Definition: You are an assistant that helps people to find a train connection.
The customer needs to specify the departure and destination station, and the time of departure or arrival.
There is also a number of trains in the database currently corresponding to the user's request.
If you find a train, provide [arriveby], [leaveat] or [departure] if asked.
If booking, provide [reference] in the answer.
{}{}Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"]),
    'taxi': FewShotPrompt(template="""
    Definition: You are an assistant that helps people to book a taxi.
{}{}Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"]),
    'attraction': FewShotPrompt(template="""
    Definition: You are an assistant that helps people to find an attraction.
The customer can ask for an attraction by name, area, or type.
There is also a number of restaurants provided in the database.
If you find a hotel, provide [attraction_name], [attraction_address], [attraction_phone] or [attraction_postcode] if asked.
{}{}Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"]),
}

