from dataclasses import dataclass
from prompts import FewShotPrompt, SimpleTemplatePrompt


SLOT_MAPPINGS = {
    "restaurant": [('price range of the restaurant; allowed values are cheap, moderate, expensive', 'pricerange'),
                   ('area where the restaurant is located; allowed values are north, east, west, south, centre', 'area'),
                   ('type of food the restaurant serves; copy the value from the user utterance', 'food'),
                   ('name of the restaurant; copy the value from the user utterance', 'name')],
    "hotel": [('area where the hotel is located; allowed values are north, east, west, south, centre', 'area'),
              ('specifies if the hotel has internet; allowed values are yes, no', 'internet'),
              ('specifies if the hotel has parking; allowed values are yes, no', 'parking'),
              ('star rating of the hotel; allowed values are 1, 2, 3, 4, 5', 'stars'),
              ('price range of the hotel; allowed values are cheap, expensive', 'pricerange'),
              ('type of the hotel; allowed values are hotel, bed and breakfast, guest house', 'type')],
    "attraction": [('area where the attraction is located; allowed values are north, east, west, south, centre', 'area'),
                   ('type of the attraction; copy the value from the user utterance', 'type'),
                   ('name of the attraction; copy the value from the user utterance', 'name')],
    "train": [('departure time of the train; copy the value from the user utterance', 'leaveAt'),
              ('arrival time of the train; copy the value from the user utterance', 'arriveBy'),
              ('departure station of the train; copy the value from the user utterance', 'departure'),
              ('arrival station of the train; copy the value from the user utterance' , 'destination'),
              ('day of the travel; allowed values are monday, tuesday, wednesday, thursday, friday, saturday, sunday', 'day')],
    "taxi": [('departure time of the taxi; copy the value from the user utterance', 'leaveAt'),
            ('arrival time of the taxi; copy the value from the user utterance', 'arriveBy'),
            ('departure address of the taxi; copy the value from the user utterance', 'departure'),
            ('arrival address of the taxi; copy the value from the user utterance', 'destination')],
    "hospital": [('department of the hospital; copy the value from the user utterance', 'department')],
    "bus": [('station of the bus; copy the value from the user utterance', 'station')],
}


"""
######################
FEW SHOT
######################
"""

@dataclass
class FewShotRestaurantDefinition:
    state_prompt = FewShotPrompt(template="""
You are an assistant that helps people to book a restaurant.
Your task is to capture entities in the user utterance.
Each entity is associated with a number and the description.
If the entity is present in the user utterance, output the value associated with the correct number.
See examples.
Values that should be captured are:""" + "\n".join([f"{n}: {desc[0]}" for n, desc in enumerate(SLOT_MAPPINGS["restaurant"])]) + \
"""If not specified, leave a dash instead of value.
{}{}Now complete the following example:
input: {}
Customer: {}
output: state:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
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
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["pricerange", "area", "food"]


@dataclass
class FewShotHotelDefinition:
    state_prompt = FewShotPrompt(template="""
Definition: You are an assistant that helps people to book a hotel.
Your task is to capture entities in the user utterance.
Each entity is associated with a number and the description.
If the entity is present in the user utterance, output the value associated with the correct number.
See examples.
Values that should be captured are:""" + "\n".join([f"{n}: {desc[0]}" for n, desc in enumerate(SLOT_MAPPINGS["hotel"])]) + \
"""If not specified, leave a dash instead of value.
{}{}Now complete the following example:
input: {}
Customer: {}
output: state:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
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
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["area", "internet", "parking", "stars", "type", "pricerange"]


@dataclass
class FewShotTrainDefinition:
    state_prompt = FewShotPrompt(template="""
You are an assistant that helps people to book a train.
Your task is to capture entities in the user utterance.
Each entity is associated with a number and the description.
If the entity is present in the user utterance, output the value associated with the correct number.
See examples.
Values that should be captured are:""" + "\n".join([f"{n}: {desc[0]}" for n, desc in enumerate(SLOT_MAPPINGS["train"])]) + \
"""If not specified, leave a dash instead of value.
{}{}Now complete the following example:
input: {}
Customer: {}
output: state:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
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
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["arriveBy", "leaveAt", "day", "departure", "destination"]


@dataclass
class FewShotTaxiDefinition:
    state_prompt = FewShotPrompt(template="""
You are an assistant that helps people to book a taxi.
Your task is to capture entities in the user utterance.
Each entity is associated with a number and the description.
If the entity is present in the user utterance, output the value associated with the correct number.
See examples.
Values that should be captured are:""" + "\n".join([f"{n}: {desc[0]}" for n, desc in enumerate(SLOT_MAPPINGS["taxi"])]) + \
"""If not specified, leave a dash instead of value.
{}{}Now complete the following example:
input: {}
Customer: {}
output: state:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
Definition: You are an assistant that helps people to book a taxi.
{}{}Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ['departure', 'destination', 'leaveAt', 'arriveBy']



@dataclass
class FewShotHospitalDefinition:
    state_prompt = FewShotPrompt(template="""
You are an assistant that helps people to find a hospital.
Your task is to capture entities in the user utterance.
Each entity is associated with a number and the description.
If the entity is present in the user utterance, output the value associated with the correct number.
See examples.
Values that should be captured are:""" + "\n".join([f"{n}: {desc[0]}" for n, desc in enumerate(SLOT_MAPPINGS["hospital"])]) + \
"""If not specified, leave the value empty.
{}{}Now complete the following example:
input: {}
Customer: {}
output: state:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
Definition: You are an assistant that helps people to find a hospital.
{}{}Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = []



@dataclass
class FewShotBusDefinition:
    state_prompt = FewShotPrompt(template="""
You are an assistant that helps people to find a bus.
Your task is to capture entities in the user utterance.
Each entity is associated with a number and the description.
If the entity is present in the user utterance, output the value associated with the correct number.
See examples.
Values that should be captured are:""" + "\n".join([f"{n}: {desc[0]}" for n, desc in enumerate(SLOT_MAPPINGS["bus"])]) + \
"""If not specified, leave the value empty.
{}{}Now complete the following example:
input: {}
Customer: {}
output: state:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
Definition: You are an assistant that helps people to find a bus.
{}{}Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = []


@dataclass
class FewShotAttractionDefinition:
    state_prompt = FewShotPrompt(template="""
You are an assistant that helps people to find a specific attraction.
Your task is to capture entities in the user utterance.
Each entity is associated with a number and the description.
If the entity is present in the user utterance, output the value associated with the correct number.
See examples.
Values that should be captured are:""" + "\n".join([f"{n}: {desc[0]}" for n, desc in enumerate(SLOT_MAPPINGS["attraction"])]) + \
"""If not specified, leave a dash instead of value.
{}{}Now complete the following example:
input: {}
Customer: {}
output: state:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
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
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["type", "area"]


"""
######################
ZERO SHOT
######################
"""


@dataclass
class ZeroShotRestaurantDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Definition: Capture values from converstation in JSON.
Values that should be captured are:
 - "pricerange" that specifies the price range of the restaurant (cheap/moderate/expensive)
 - "area" that specifies the area where the restaurant is located (north/east/west/south/centre)
 - "food" that specifies the type of food the restaurant serves
 - "name" that is the name of the restaurant
Do not capture ay other values!
If not specified, leave the value empty.
input: {}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
Definition: You are an assistant that helps people to book a restaurant.
You can search for a restaurant by area, food, or price.
There is also a number of restaurants in the database currently corresponding to the user's request.
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [address].
If you find a restaurant, provide [restaurant_name], [restaurant_address], [restaurant_phone] or [restaurant_postcode] if asked.
If booking, provide [reference] in the answer.
input:{}
Customer: {}
state: {}
database: {}
output:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["pricerange", "area", "food", "name"]


@dataclass
class ZeroShotHotelDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Definition: Capture values from converstation in JSON.
Values that should be captured are:
 - "area" that specifies the area where the hotel is located (north/east/west/south/centre)
 - "internet" that specifies if the hotel has internet (yes/no)
 - "parking" that specifies if the hotel has parking (yes/no)
 - "stars" that specifies the number of stars the hotel has (1/2/3/4/5)
 - "type" that specifies the type of the hotel (hotel/bed and breakfast/guest house)
 - "pricerange" that specifies the price range of the hotel (cheap/expensive)
Do not capture ay other values!
If not specified, leave the value empty.
input: {}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
    Definition: You are an assistant that helps people to book a hotel.
The customer can ask for a hotel by name, area, parking, internet availability, or price.
There is also a number of hotel in the database currently corresponding to the user's request.
If you find a hotel, provide [hotel_name], [hotel_address], [hotel_phone] or [hotel_postcode] if asked.
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [address].
If booking, provide [reference] in the answer.
input:{}
Customer: {}
state: {}
database: {}
output:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["area", "internet", "parking", "stars", "type", "pricerange"]


@dataclass
class ZeroShotTrainDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Definition: Capture values from converstation in JSON.
Values that should be captured are:
 - "arriveBy" that specifies what time the train should arrive
 - "leaveAt" that specifies what time the train should leave
 - "day" that specifies what day the train should leave (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - "departure" that specifies the departure station
 - "destination" that specifies the destination station
Do not capture any other values!
If not specified, leave the value empty
input: {}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
Definition: You are an assistant that helps people to find a train connection.
The customer needs to specify the departure and destination station, and the time of departure or arrival.
There is also a number of trains in the database currently corresponding to the user's request.
If you find a train, provide [arriveby], [leaveat] or [departure] if asked.
Do not provide real entities in the response! Just provide entity name in brackets, like [duration] or [price].
If booking, provide [reference] in the answer.
input:{}
Customer: {}
state: {}
database: {}
output:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["arriveBy", "leaveAt", "day", "departure", "destination"]


@dataclass
class ZeroShotTaxiDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Definition: Capture values from converstation in JSON.
If not specified, leave the value empty.
Values that should be captured are:
 - "arriveBy" that specifies what time the train should arrive
 - "leaveAt" that specifies what time the train should leave
 - "departure" that specifies the departure station
 - "destination" that specifies the destination station
input: {}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
Definition: You are an assistant that helps people to book a taxi.
Do not provide real entities in the response! Just provide entity name in brackets, like [color] or [type].
input:{}
Customer: {}
state: {}
database: {}
output:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ['departure', 'destination', 'leaveAt', 'arriveBy']



@dataclass
class ZeroShotHospitalDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Definition: Capture values from converstation in JSON according to examples.
If not specified, leave the value empty.
input: {}
Customer: {}
output: state:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
Definition: You are an assistant that helps people to find a hospital.
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = []



@dataclass
class ZeroShotBusDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Definition: Capture values from converstation in JSON according to examples.
If not specified, leave the value empty.
input: {}
Customer: {}
output: state:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
Definition: You are an assistant that helps people to find a bus.
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = []


@dataclass
class ZeroShotAttractionDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Definition: Capture values from converstation in JSON.
Values that should be captured are:
 - "type" that specifies the type of attraction (museum/gallery/theatre/concert/stadium)
 - "area" that specifies the area where the attraction is located (north/east/west/south/centre)
Do not capture ay other values!
If not specified, leave the value empty.
input: {}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
    Definition: You are an assistant that helps people to find an attraction.
The customer can ask for an attraction by name, area, or type.
There is also a number of restaurants provided in the database.
Do not provide real entities in the response! Just provide entity name in brackets, like [address] or [name].
If you find a hotel, provide [attraction_name], [attraction_address], [attraction_phone] or [attraction_postcode] if asked.
input:{}
Customer: {}
state: {}
database: {}
output:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["type", "area"]


FEW_SHOT_DOMAIN_DEFINITIONS = {
    "restaurant": FewShotRestaurantDefinition,
    "hotel": FewShotHotelDefinition,
    "attraction": FewShotAttractionDefinition,
    "train": FewShotTrainDefinition,
    "taxi": FewShotTaxiDefinition,
    "hospital": FewShotHospitalDefinition,
    "bus": FewShotBusDefinition,
}


ZERO_SHOT_DOMAIN_DEFINITIONS = {
    "restaurant": ZeroShotRestaurantDefinition,
    "hotel": ZeroShotHotelDefinition,
    "attraction": ZeroShotAttractionDefinition,
    "train": ZeroShotTrainDefinition,
    "taxi": ZeroShotTaxiDefinition,
    "hospital": ZeroShotHospitalDefinition,
    "bus": ZeroShotBusDefinition,
}
