from dataclasses import dataclass
from prompts import FewShotPrompt, SimpleTemplatePrompt


sgd_domain_prompt = SimpleTemplatePrompt(template="""
Determine which domain is considered in the following dialogue situation.
Choose one domain from this list:
 - music
 - hotels
 - rentalcars
 - buses
 - restaurants
 - homes
 - ridesharing
 - services
 - events
 - flights
 - media
 - calendar
 - banks
 - movies

Answer with only one word, the selected domain from the list.
You have to always select the closest possible domain.
Consider the last domain mentioned, so focus mainly on the last utterance.

----------------
Example1:
Customer: I need a cheap place to eat
Assistant: We have several not expensive places available. What foor are you interested in?
Customer: Chinese food.

Domain: restaurant

-------

Example 2:
Customer: I also need a hotel in the north.
Assistant: Ok, can I offer you the Molly's place?
Customer: What is the address?

Domain: hotel

---------

Example 3:
Customer: What is the address?
Assistant: It's 123 Northfolk Road.
Customer: That's all. I also need a train from London.

Domain: train
""

Now complete the following example:
{}
{}
Domain:""", args_order=["history", "utterance"])


multiwoz_domain_prompt = SimpleTemplatePrompt(template="""
Determine which domain is considered in the following dialogue situation.
Choose one domain from this list:
 - restaurant
 - hotel
 - attraction
 - taxi
 - train
Answer with only one word, the selected domain from the list.
You have to always select the closest possible domain.
Consider the last domain mentioned, so focus mainly on the last utterance.

-------------------
Example1:
Customer: I need a cheap place to eat
Assistant: We have several not expensive places available. What food are you interested in?
Customer: Chinese food.

Domain: restaurant

-------

Example 2:
Customer: I also need a hotel in the north.
Assistant: Ok, can I offer you the Molly's place?
Customer: What is the address?

Domain: hotel

---------

Example 3:
Customer: What is the address?
Assistant: It's 123 Northfolk Road.
Customer: That's all. I also need a train from London.

Domain: train
""

Now complete the following example:
{}
{}
Domain:""", args_order=["history", "utterance"])

"""
######################
FEW SHOT
######################
"""

@dataclass
class FewShotRestaurantDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - "pricerange" that specifies the price range of the restaurant (cheap/moderate/expensive)
 - "area" that specifies the area where the restaurant is located (north/east/west/south/centre)
 - "food" that specifies the type of food the restaurant serves
 - "name" that specifies the name of the restaurant
 - "bookday" that specifies the day of the booking
 - "booktime" that specifies the time of the booking
 - "bookpeople" that specifies for how many people is the booking made
Do not capture ay other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
Definition: You are an assistant that helps people to book a restaurant.
You can search for a restaurant by area, food, or price.
There is also a number of restaurants in the database currently corresponding to the user's request.
If you find a restaurant, provide [restaurant_name], [restaurant_address], [restaurant_phone] or [restaurant_postcode] if asked.
If booking, provide [reference] in the answer.
Always act as if the booking is successfully done.
------
{}{}
---------
Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["pricerange", "area", "food", "name", "bookday", "bookpeople", "booktime"]


@dataclass
class FewShotHotelDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.

Values that should be captured are:
 - "area" that specifies the area where the hotel is located (north/east/west/south/centre)
 - "internet" that specifies if the hotel has internet (yes/no)
 - "parking" that specifies if the hotel has parking (yes/no)
 - "stars" that specifies the number of stars the hotel has (1/2/3/4/5)
 - "type" that specifies the type of the hotel (hotel/bed and breakfast/guest house)
 - "pricerange" that specifies the price range of the hotel (cheap/expensive)
 - "name" that specifies name of the hotel
 - "bookstay" specifies length of the stay
 - "bookday" specifies the day of the booking
 - "bookpeople" specifies how many people should be booked for.
Do not capture ay other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
    Definition: You are an assistant that helps people to book a hotel.
The customer can ask for a hotel by name, area, parking, internet availability, or price.
There is also a number of hotels in the database currently corresponding to the user's request.
If you find a hotel, provide [hotel_name], [hotel_address], [hotel_phone] or [hotel_postcode] if asked. Use brackets like that.
If booking, provide [reference] in the answer.
Always act as if the booking is successfully done.
------
{}{}
---------
Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["area", "internet", "parking", "stars", "type", "pricerange", "name", "booktime", "bookpeople", "bookstay"]


@dataclass
class FewShotTrainDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.

Values that should be captured are:
 - "arriveby" that specifies what time the train should arrive
 - "leaveat" that specifies what time the train should leave
 - "day" that specifies what day the train should leave (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - "departure" that specifies the departure station
 - "destination" that specifies the destination station
 - "bookpeople" that specifies how many people the booking is for
Do not capture ay other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
Definition: You are an assistant that helps people to find a train connection.
The customer needs to specify the departure and destination station, and the time of departure or arrival.
There is also a number of trains in the database currently corresponding to the user's request.
If you find a train, provide [arriveby], [leaveat] or [departure] if asked.
If booking, provide [reference] in the answer.
------
{}{}
---------
Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["arriveby", "leaveat", "bookpeople", "day", "departure", "destination"]


@dataclass
class FewShotTaxiDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.

If not specified, leave the value empty.
Values that should be captured are:
 - "arriveby" that specifies what time the train should arrive
 - "leaveat" that specifies what time the train should leave
 - "departure" that specifies the departure station
 - "destination" that specifies the destination station
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
Definition: You are an assistant that helps people to book a taxi.
If you find a taxi, provide the type of the car as [type] and [phone] as the phone number in the answer.
------
{}{}
---------
Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ['departure', 'destination', 'leaveat', 'arriveby']



@dataclass
class FewShotHospitalDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.

 - "department" that specifies the department of interest
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
Definition: You are an assistant that helps people to find a hospital.
------
{}{}
---------
Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ['department']



@dataclass
class FewShotBusDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.

If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
Definition: You are an assistant that helps people to find a bus.
------
{}{}
---------
Now complete the following example:
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
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.

Values that should be captured are:
 - "type" that specifies the type of attraction (museum/gallery/theatre/concert/stadium)
 - "area" that specifies the area where the attraction is located (north/east/west/south/centre)
 - "name" that specigies the name of the attraction
Do not capture ay other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
    Definition: You are an assistant that helps people to find an attraction.
The customer can ask for an attraction by name, area, or type.
There is also a number of restaurants provided in the database.
If you find a hotel, provide [attraction_name], [attraction_address], [attraction_phone] or [attraction_postcode] if asked.
------
{}{}
---------
Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["type", "area", "name"]


"""
######################
ZERO SHOT
######################
"""


@dataclass
class ZeroShotRestaurantDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
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
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
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
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
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
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
If not specified, leave the value empty.
Values that should be captured are:
 - "arriveBy" that specifies what time the train should arrive
 - "leaveAt" that specifies what time the train should leave
 - "departure" that specifies the departure station
 - "destination" that specifies the destination station
 - "day" that specifies what day the train should leave (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
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
    expected_slots = ['departure', 'destination', 'leaveAt', 'arriveBy', 'date']



@dataclass
class ZeroShotHospitalDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
If not specified, leave the value empty.
input: {}
Customer: {}
output:
state:""",
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
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
If not specified, leave the value empty.
input: {}
Customer: {}
output:
state:""",
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
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
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


MW_FEW_SHOT_DOMAIN_DEFINITIONS = {
    "restaurant": FewShotRestaurantDefinition,
    "hotel": FewShotHotelDefinition,
    "attraction": FewShotAttractionDefinition,
    "train": FewShotTrainDefinition,
    "taxi": FewShotTaxiDefinition,
    "hospital": FewShotHospitalDefinition,
    "bus": FewShotBusDefinition,
}


MW_ZERO_SHOT_DOMAIN_DEFINITIONS = {
    "restaurant": ZeroShotRestaurantDefinition,
    "hotel": ZeroShotHotelDefinition,
    "attraction": ZeroShotAttractionDefinition,
    "train": ZeroShotTrainDefinition,
    "taxi": ZeroShotTaxiDefinition,
    "hospital": ZeroShotHospitalDefinition,
    "bus": ZeroShotBusDefinition,
}


@dataclass
class FewShotSGD_ServicesDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - type; type of the service
 - offers_cosmetic_services;
 - city; the ity where the service is offered
 - dentist_name; name of the dentist in question
 - appointment_date; the date of the appointment
 - stylist_name; name of the stylist being discussed
 - doctor_name; name of the doctor being discussed
 - appointment_time; time of the appointment
Do not capture ay other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""",
                                args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in various services (doctor, dentist, cosmetic, ...).
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [time].
Use similiar approach as in the following examples.
------
{}{}
---------

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])
    expected_slots = ["type", "offers_cosmetic_services", "city", "dentist_name", "appointment_date", "stylist_name", "doctor_name", "appointment_time"]


@dataclass
class FewShotSGD_EventsDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:

event_type; type of the event
subcategory; subcategory of the event
number_of_tickets; number of tickets required
city; city where the event is happening
category; category of the event
date; date of the event
event_name; name of the event
number_of_seats; number of seats required
city_of_event; city where the event is taking place
Do not capture any other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in various events (concerts, shows, sports events, etc.).
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [time].
Use similar approach as in the following examples.
------
{}{}
---------

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])
    expected_slots = ["event_type", "subcategory", "number_of_tickets", "city", "category", "date", "event_name", "number_of_seats", "city_of_event"]


@dataclass
class FewShotSGD_MoviesDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:

theater_name; name of the theater where the movie is being screened

genre; genre of the movie

location; the location where the theater is situated

movie_name; name of the movie being discussed

show_type; type of show (e.g. 3D, IMAX, etc.)

show_time; time of the show

show_date; date of the show
Do not capture any other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = FewShotPrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in movies and their show timings.
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [time].
Use a similar approach as in the following examples.
------
{}{}
---------

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["theater_name", "genre", "location", "movie_name", "show_type", "show_time", "show_date"]


@dataclass
class FewShotSGD_HotelsDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:

where_to; destination of the hotel

rating; rating of the hotel

number_of_rooms; number of rooms required

star_rating; star rating of the hotel

has_laundry_service; whether the hotel has laundry service or not

number_of_days; number of days of stay

has_wifi; whether the hotel has wifi or not

destination; location of the hotel

check_out_date; date of check-out from the hotel

check_in_date; date of check-in to the hotel

number_of_adults; number of adults staying in the hotel

hotel_name; name of the hotel

location; location of the hotel
Do not capture any other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = FewShotPrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in booking a hotel and needs assistance with the booking process.
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [date].
Use similar approach as in the following examples.
------
{}{}
---------

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["where_to", "rating", "number_of_rooms", "star_rating", "has_laundry_service", "number_of_days", "has_wifi", "destination", "check_out_date", "check_in_date", "number_of_adults", "hotel_name"]


@dataclass
class FewShotSGD_FlightsDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - seating_class; the seating class of the flight
 - passengers; number of passengers travelling
 - airlines; the airline(s) being discussed
 - refundable; whether the ticket is refundable or not
 - origin_city; the city where the flight originates
 - outbound_departure_time; departure time of the outbound flight
 - departure_date; date of departure
 - origin; the origin airport code
 - destination_city; the city where the flight is headed
 - return_date; return date (if any)
 - inbound_departure_time; departure time of the inbound flight (if any)
Do not capture any other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = FewShotPrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in booking a flight with specific details (seating class, airline, refundable, etc.).
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [time].
Use a similar approach as in the following examples.
------
{}{}
---------

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["seating_class", "passengers", "airlines", "refundable", "origin_city", "outbound_departure_time", "departure_date", "origin", "destination_city", "return_date", "inbound_departure_time"]


@dataclass
class FewShotSGD_MusicDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - genre; genre of the songs
 - year; year the song or ablum was released
 - playback_device; the device that can play the music
 - song_name; name of the mentioned song
 - album; name of the mentioned album
 - artist; name fo the artist
Do not capture any other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = FewShotPrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in finding music to listen to.
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [time].
Use a similar approach as in the following examples.
------
{}{}
---------

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["genre", "year", "playback_device", "song_name", "album", "artist"]


@dataclass
class FewShotSGD_RestaurantsDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:

serves_alcohol;

party_size;

time; the time of the reservation

city; the city where the restaurant is located

cuisine; the type of cuisine offered by the restaurant

restaurant_name; the name of the restaurant

date; the date of the reservation

price_range; the price range of the restaurant
Do not capture any other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = FewShotPrompt(template="""
Complete a conversation between a customer and a restaurant booking assistant.
The customer is interested in making a reservation at a restaurant.
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [time].
Use similar approach as in the following examples.
------
{}{}
---------

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["serves_alcohol", "party_size", "time", "city", "cuisine", "restaurant_name", "date", "price_range"]

@dataclass
class FewShotSGD_BusesDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - group_size; the size of the group traveling
 - fare_type; type of the fare chosen
 - departure_time; the time of the departure
 - departure_date; the date of the departure
 - origin; the origin location of the bus
 - from_location; the starting location of the trip
 - to_location; the final destination of the trip
 - destination; the destination location of the bus
 - leaving_date; the date of the leaving
 - travelers; the number of travelers
Do not capture any other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = FewShotPrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in bus services.
Do not provide real entities in the response! Just provide entity name in brackets, like [location] or [time].
Use a similar approach as in the following examples.
------
{}{}
---------

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["group_size", "fare_type", "departure_time", "departure_date", "origin", "from_location", "to_location", "destination", "leaving_date", "travelers"]


@dataclass
class FewShotSGD_MediaDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - directed_by; Name of the movie director
 - genre; genre of the movie
 - subtitles; ehat language are the subtitles available at
 - title; the name or title of the movie
Do not capture any other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = FewShotPrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in movies on medias.
Do not provide real entities in the response! Just provide entity name in brackets, like [location] or [time].
Use a similar approach as in the following examples.
------
{}{}
---------

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["directed_by", "genre", "subtitles", "title"]


@dataclass
class FewShotSGD_BanksDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - account_type; the type of the account - checkings or savings
 - recipient_account_type; the account type of the recipient, checkings or savings
 - amount; the amount of money to be transfered
 - recipient_account_name; the name of the recipient's account
Do not capture any other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = FewShotPrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer wants to do a bnak transfer.
Do not provide real entities in the response! Just provide entity name in brackets, like [location] or [time].
Use a similar approach as in the following examples.
------
{}{}
---------

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["account_type", "recipient_account_name", "amount", "recipient_account_type"]


@dataclass
class FewShotSGD_RidesharingDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - ride_type; the type of the shared ride
 - destination; to where the customer wants to go
 - number_of_riders; the number of riders inthe car
 - number_of_seats; number of seats available inthe car
 - shared_ride; whether the ride is shared
Do not capture any other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = FewShotPrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in sharing a car ride.
Do not provide real entities in the response! Just provide entity name in brackets, like [location] or [time].
Use a similar approach as in the following examples.
------
{}{}
---------

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["ride_type", "destination", "number_of_seats", "number_of_riders", "shared_ride"]


@dataclass
class FewShotSGD_CalendarDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - event_time; the time of the event
 - event_name; the name of the event
 - event_date; the date of the event
 - event_location; the location of the event.
Do not capture any other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = FewShotPrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer wants the assistant to handle calendar events for him.
Do not provide real entities in the response! Just provide entity name in brackets, like [location] or [time].
Use a similar approach as in the following examples.
------
{}{}
---------

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["event_date", "event_location", "event_name", "event_time"]


@dataclass
class FewShotSGD_RentalDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:

type; type of the rental car
pickup_date; the date of the pickup
dropoff_date; the date of the dropoff
pickup_city; the city where the rental car will be picked up
pickup_time; the time of the pickup
car_type; the type of the rental car
pickup_location; the location where the rental car will be picked up
Do not capture any other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in renting a car and needs information about available cars and pricing.
Do not provide real entities in the response! Just provide entity name in brackets, like [pickup_date] or [car_type].
Use a similar approach as in the following examples.
------
{}{}
---------

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])
    expected_slots = ["type", "pickup_date", "dropoff_date", "pickup_city", "pickup_time", "car_type", "pickup_location"]


@dataclass
class FewShotSGD_HomesDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:

pets_allowed; whether pets are allowed in the property or not

number_of_beds; number of beds in the property

furnished; whether the property is furnished or not

number_of_baths; number of bathrooms in the property

property_name; name of the property in question

area; area of the property

visit_date; the date when the customer wants to visit the property
Do not capture any other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = FewShotPrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in various properties.
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [date].
Use a similar approach as in the following examples.
------
{}{}
---------

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["pets_allowed", "number_of_beds", "furnished", "number_of_baths", "property_name", "area", "visit_date"]


@dataclass
class ZeroShotSGD_ServicesDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - type; type of the service
 - offers_cosmetic_services;
 - city; the ity where the service is offered
 - dentist_name; name of the dentist in question
 - appointment_date; the date of the appointment
 - stylist_name; name of the stylist being discussed
 - doctor_name; name of the doctor being discussed
 - appointment_time; time of the appointment
Do not capture ay other values!
If not specified, leave the value empty.
input: {}
Customer: {}
output:
state:""",
                                args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in various services (doctor, dentist, cosmetic, ...).
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [time].

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])
    expected_slots = ["type", "offers_cosmetic_services", "city", "dentist_name", "appointment_date", "stylist_name", "doctor_name", "appointment_time"]


@dataclass
class ZeroShotSGD_EventsDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:

event_type; type of the event
subcategory; subcategory of the event
number_of_tickets; number of tickets required
city; city where the event is happening
category; category of the event
date; date of the event
event_name; name of the event
number_of_seats; number of seats required
city_of_event; city where the event is taking place
Do not capture any other values!
If not specified, leave the value empty.
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in various events (concerts, shows, sports events, etc.).
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [time].

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])
    expected_slots = ["event_type", "subcategory", "number_of_tickets", "city", "category", "date", "event_name", "number_of_seats", "city_of_event"]


@dataclass
class ZeroShotSGD_MoviesDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:

theater_name; name of the theater where the movie is being screened

genre; genre of the movie

location; the location where the theater is situated

movie_name; name of the movie being discussed

show_type; type of show (e.g. 3D, IMAX, etc.)

show_time; time of the show

show_date; date of the show
Do not capture any other values!
If not specified, leave the value empty.
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = SimpleTemplatePrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in movies and their show timings.
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [time].

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["theater_name", "genre", "location", "movie_name", "show_type", "show_time", "show_date"]


@dataclass
class ZeroShotSGD_HotelsDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:

where_to; destination of the hotel

rating; rating of the hotel

number_of_rooms; number of rooms required

star_rating; star rating of the hotel

has_laundry_service; whether the hotel has laundry service or not

number_of_days; number of days of stay

has_wifi; whether the hotel has wifi or not

destination; location of the hotel

check_out_date; date of check-out from the hotel

check_in_date; date of check-in to the hotel

number_of_adults; number of adults staying in the hotel

hotel_name; name of the hotel

location; location of the hotel
Do not capture any other values!
If not specified, leave the value empty.
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = SimpleTemplatePrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in booking a hotel and needs assistance with the booking process.
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [date].

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["where_to", "rating", "number_of_rooms", "star_rating", "has_laundry_service", "number_of_days", "has_wifi", "destination", "check_out_date", "check_in_date", "number_of_adults", "hotel_name"]


@dataclass
class ZeroShotSGD_FlightsDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - seating_class; the seating class of the flight
 - passengers; number of passengers travelling
 - airlines; the airline(s) being discussed
 - refundable; whether the ticket is refundable or not
 - origin_city; the city where the flight originates
 - outbound_departure_time; departure time of the outbound flight
 - departure_date; date of departure
 - origin; the origin airport code
 - destination_city; the city where the flight is headed
 - return_date; return date (if any)
 - inbound_departure_time; departure time of the inbound flight (if any)
Do not capture any other values!
If not specified, leave the value empty.
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = SimpleTemplatePrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in booking a flight with specific details (seating class, airline, refundable, etc.).
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [time].

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["seating_class", "passengers", "airlines", "refundable", "origin_city", "outbound_departure_time", "departure_date", "origin", "destination_city", "return_date", "inbound_departure_time"]


@dataclass
class ZeroShotSGD_MusicDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - genre; genre of the songs
 - year; year the song or ablum was released
 - playback_device; the device that can play the music
 - song_name; name of the mentioned song
 - album; name of the mentioned album
 - artist; name fo the artist
Do not capture any other values!
If not specified, leave the value empty.
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = SimpleTemplatePrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in finding music to listen to.
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [time].

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["genre", "year", "playback_device", "song_name", "album", "artist"]


@dataclass
class ZeroShotSGD_RestaurantsDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:

serves_alcohol;

party_size;

time; the time of the reservation

city; the city where the restaurant is located

cuisine; the type of cuisine offered by the restaurant

restaurant_name; the name of the restaurant

date; the date of the reservation

price_range; the price range of the restaurant
Do not capture any other values!
If not specified, leave the value empty.
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = SimpleTemplatePrompt(template="""
Complete a conversation between a customer and a restaurant booking assistant.
The customer is interested in making a reservation at a restaurant.
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [time].

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["serves_alcohol", "party_size", "time", "city", "cuisine", "restaurant_name", "date", "price_range"]

@dataclass
class ZeroShotSGD_BusesDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - group_size; the size of the group traveling
 - fare_type; type of the fare chosen
 - departure_time; the time of the departure
 - departure_date; the date of the departure
 - origin; the origin location of the bus
 - from_location; the starting location of the trip
 - to_location; the final destination of the trip
 - destination; the destination location of the bus
 - leaving_date; the date of the leaving
 - travelers; the number of travelers
Do not capture any other values!
If not specified, leave the value empty.
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = SimpleTemplatePrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in bus services.
Do not provide real entities in the response! Just provide entity name in brackets, like [location] or [time].

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["group_size", "fare_type", "departure_time", "departure_date", "origin", "from_location", "to_location", "destination", "leaving_date", "travelers"]


@dataclass
class ZeroShotSGD_MediaDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - directed_by; Name of the movie director
 - genre; genre of the movie
 - subtitles; ehat language are the subtitles available at
 - title; the name or title of the movie
Do not capture any other values!
Now complete the following example:
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = SimpleTemplatePrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in movies on medias.
Do not provide real entities in the response! Just provide entity name in brackets, like [location] or [time].

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["directed_by", "genre", "subtitles", "title"]


@dataclass
class ZeroShotSGD_BanksDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - account_type; the type of the account - checkings or savings
 - recipient_account_type; the account type of the recipient, checkings or savings
 - amount; the amount of money to be transfered
 - recipient_account_name; the name of the recipient's account
Do not capture any other values!
If not specified, leave the value empty.
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = SimpleTemplatePrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer wants to do a bnak transfer.
Do not provide real entities in the response! Just provide entity name in brackets, like [location] or [time].

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["account_type", "recipient_account_name", "amount", "recipient_account_type"]


@dataclass
class ZeroShotSGD_RidesharingDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - ride_type; the type of the shared ride
 - destination; to where the customer wants to go
 - number_of_riders; the number of riders inthe car
 - number_of_seats; number of seats available inthe car
 - shared_ride; whether the ride is shared
Do not capture any other values!
Now complete the following example:
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = SimpleTemplatePrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in sharing a car ride.
Do not provide real entities in the response! Just provide entity name in brackets, like [location] or [time].

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["ride_type", "destination", "number_of_seats", "number_of_riders", "shared_ride"]


@dataclass
class ZeroShotSGD_CalendarDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - event_time; the time of the event
 - event_name; the name of the event
 - event_date; the date of the event
 - event_location; the location of the event.
Do not capture any other values!
If not specified, leave the value empty.
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = SimpleTemplatePrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer wants the assistant to handle calendar events for him.
Do not provide real entities in the response! Just provide entity name in brackets, like [location] or [time].

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["event_date", "event_location", "event_name", "event_time"]


@dataclass
class ZeroShotSGD_RentalDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:

type; type of the rental car
pickup_date; the date of the pickup
dropoff_date; the date of the dropoff
pickup_city; the city where the rental car will be picked up
pickup_time; the time of the pickup
car_type; the type of the rental car
pickup_location; the location where the rental car will be picked up
Do not capture any other values!
Now complete the following example:
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in renting a car and needs information about available cars and pricing.
Do not provide real entities in the response! Just provide entity name in brackets, like [pickup_date] or [car_type].

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])
    expected_slots = ["type", "pickup_date", "dropoff_date", "pickup_city", "pickup_time", "car_type", "pickup_location"]


@dataclass
class ZeroShotSGD_HomesDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:

pets_allowed; whether pets are allowed in the property or not

number_of_beds; number of beds in the property

furnished; whether the property is furnished or not

number_of_baths; number of bathrooms in the property

property_name; name of the property in question

area; area of the property

visit_date; the date when the customer wants to visit the property
Do not capture any other values!
Now complete the following example:
input: {}
Customer: {}
output:
state:""", args_order=["history", "utterance"])

    response_prompt = SimpleTemplatePrompt(template="""
Complete a conversation between a polite and knowledgeable personal assistant and a customer.
The customer is interested in various properties.
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [date].

input:{}
Customer: {}
state: {}
database: {}
output:""", args_order=["history", "utterance", "state", "database"])

    expected_slots = ["pets_allowed", "number_of_beds", "furnished", "number_of_baths", "property_name", "area", "visit_date"]


SGD_FEW_SHOT_DOMAIN_DEFINITIONS = {
    "music": FewShotSGD_MusicDefinition,
    "hotels": FewShotSGD_HotelsDefinition,
    "rentalcars": FewShotSGD_RentalDefinition,
    "buses": FewShotSGD_BusesDefinition,
    "restaurants": FewShotSGD_RestaurantsDefinition,
    "homes": FewShotSGD_HomesDefinition,
    "ridesharing": FewShotSGD_RidesharingDefinition,
    "services": FewShotSGD_ServicesDefinition,
    "events": FewShotSGD_EventsDefinition,
    "flights": FewShotSGD_FlightsDefinition,
    "media": FewShotSGD_MediaDefinition,
    "calendar": FewShotSGD_CalendarDefinition,
    "banks": FewShotSGD_BanksDefinition,
    "movies": FewShotSGD_MoviesDefinition,
}


SGD_ZERO_SHOT_DOMAIN_DEFINITIONS = {
    "music": ZeroShotSGD_MusicDefinition,
    "hotels": ZeroShotSGD_HotelsDefinition,
    "rentalcars": ZeroShotSGD_RentalDefinition,
    "buses": ZeroShotSGD_BusesDefinition,
    "restaurants": ZeroShotSGD_RestaurantsDefinition,
    "homes": ZeroShotSGD_HomesDefinition,
    "ridesharing": ZeroShotSGD_RidesharingDefinition,
    "services": ZeroShotSGD_ServicesDefinition,
    "events": ZeroShotSGD_EventsDefinition,
    "flights": ZeroShotSGD_FlightsDefinition,
    "media": ZeroShotSGD_MediaDefinition,
    "calendar": ZeroShotSGD_CalendarDefinition,
    "banks": ZeroShotSGD_BanksDefinition,
    "movies": ZeroShotSGD_MoviesDefinition,

}
