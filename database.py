import json
import random
import copy
import os
from typing import Text, Dict

from fuzzywuzzy import fuzz


class MultiWOZDatabase:
    """ MultiWOZ database implementation. """

    IGNORE_VALUES = {
        'hospital' : ['id'],
        'police' : ['id'],
        'attraction' : ['location', 'openhours'],
        'hotel' : ['location', 'price'],
        'restaurant' : ['location', 'introduction']
    }

    FUZZY_KEYS = {
        'hospital' : {'department'},
        'hotel' : {'name'},
        'attraction' : {'name'},
        'restaurant' : {'name', 'food'},
        'bus' : {'departure', 'destination'},
        'train' : {'departure', 'destination'},
        'police' : {'name'}
    }

    DOMAINS = [
    	'restaurant',
    	'hotel',
    	'attraction',
    	'train',
    	'taxi',
    	'police',
    	'hospital'
    ]

    def __init__(self, database_path: Text):
        self.data, self.data_keys = self._load_data(database_path)

    def _load_data(self, database_path: Text):
        database_data = {}
        database_keys = {}

        for domain in self.DOMAINS:
            with open(os.path.join(database_path, f"{domain}_db.json"), "r") as f:
                for l in f:
                    if not l.startswith('##') and l.strip() != "":
                        f.seek(0)
                        break
                database_data[domain] = json.load(f)

            if domain in self.IGNORE_VALUES:
                for i in database_data[domain]:
                    for ignore in self.IGNORE_VALUES[domain]:
                        if ignore in i:
                            i.pop(ignore)

            database_keys[domain] = set()
            if domain == 'taxi':
                database_data[domain] =  {k.lower(): v for k, v in database_data[domain].items()}
                database_keys[domain].update([k.lower() for k in database_data[domain].keys()])
            else:
                for i, database_item in enumerate(database_data[domain]):
                    database_data[domain][i] =  {k.lower(): v for k, v in database_item.items()}
                    database_keys[domain].update([k.lower() for k in database_item.keys()])

        return database_data, database_keys

    def time_str_to_minutes(self, time_string) -> Text:
        """ Convert the time string into a number (number of minutes, hours, seconds from 1970, whatever you want) """
        time = time_string
        # Copied from https://github.com/Tomiinek/MultiWOZ_Evaluation/blob/4e60b60e58ff885412b630af3a86ad1f63135118/mwzeval/normalization.py#L165-L219
        """ Converts time to the only format supported by database, e.g. 07:15. """
        time = time.strip().lower()

        if time == "afternoon": return "13:00"
        if time == "lunch" or time == "noon" or time == "mid-day" or time == "around lunch time": return "12:00"
        if time == "morning": return "08:00"
        if time.startswith("one o'clock p.m"): return "13:00"
        if time.startswith("ten o'clock a.m"): return "10:00"
        if time == "seven o'clock tomorrow evening":  return "07:00"
        if time == "three forty five p.m":  return "15:45"
        if time == "one thirty p.m.":  return "13:30"
        if time == "six fourty five":  return "06:45"
        if time == "eight thirty":  return "08:30"

        if time.startswith("by"):
            time = time[3:]

        if time.startswith("after"):
            time = time[5:].strip()

        if time.startswith("afer"):
            time = time[4:].strip()

        if time.endswith("am"):   time = time[:-2].strip()
        if time.endswith("a.m."): time = time[:-4].strip()

        if time.endswith("pm") or time.endswith("p.m."):
            if time.endswith("pm"):   time = time[:-2].strip()
            if time.endswith("p.m."): time = time[:-4].strip()
            tokens = time.split(':')
            if len(tokens) == 2:
                return str(int(tokens[0]) + 12) + ':' + tokens[1]
            if len(tokens) == 1 and tokens[0].isdigit():
                return str(int(tokens[0]) + 12) + ':00'

        if len(time) == 0:
            return "00:00"

        if time[-1] == '.' or time[-1] == ',' or time[-1] == '?':
            time = time[:-1]

        if time.isdigit() and len(time) == 4:
            return time[:2] + ':' + time[2:]

        if time.isdigit(): return time.zfill(2) + ":00"

        if ':' in time:
            time = ''.join(time.split(' '))

        if len(time) == 4 and time[1] == ':':
            tokens = time.split(':')
            return tokens[0].zfill(2) + ':' + tokens[1]

        return time


    def query(self,
              domain: Text,
              constraints: Dict[Text, Text],
              fuzzy_ratio: int=90):
        """
        Returns the list of entities (dictionaries) for a given domain based on the annotation of the belief state.

        Arguments:
            domain:      Name of the queried domain.
            constraints: Hard constraints to the query results.
        """

        if domain == 'taxi':
            c, t, p = None, None, None

            c = str(constraints.get('color', []))
            c = c[0] if len(c) > 0 else random.choice(self.data[domain]['taxi_colors'])
            t = str(constraints.get('type', []))
            t = t[0] if len(t) > 0 else random.choice(self.data[domain]['taxi_types'])
            p = str(constraints.get('phone', []))
            p = p[0] if len(p) > 0 else ''.join([str(random.randint(1, 9)) for _ in range(11)])

            return [{'color': c, 'type' : t, 'phone' : p}]

        elif domain == 'hospital':

            hospital = {
                'hospital phone': '01223245151',
                'address': 'Hills Rd, Cambridge',
                'postcode': 'CB20QQ',
                'name': 'Addenbrookes'
            }

            departments = [x.strip().lower() for x in constraints.get('department', [])]
            phones = [x.strip().lower() for x in constraints.get('phone', [])]

            if len(departments) == 0 and len(phones) == 0:
                return [dict(hospital)]
            else:
                results = []
                for i in self.data[domain]:
                    if 'department' in self.FUZZY_KEYS[domain]:
                        f = (lambda x: fuzz.partial_ratio(i['department'].lower(), x) > fuzzy_ratio)
                    else:
                        f = (lambda x: i['department'].lower() == x)

                    if any(f(x) for x in departments) and \
                       (len(phones) == 0 or any(i['phone'] == p.strip() for p in phones)):
                        results.append(dict(i))
                        results[-1].update(hospital)

                return results

        elif domain in self.DOMAINS:
            # Hotel database keys:      address, area, name, phone, postcode, pricerange, type, internet, parking, stars, takesbookings (other are ignored)
            # Attraction database keys: address, area, name, phone, postcode, pricerange, type, entrance fee (other are ignored)
            # Restaurant database keys: address, area, name, phone, postcode, pricerange, type, food

            # Train database contains keys: arriveby, departure, day, leaveat, destination, trainid, price, duration
            # The keys arriveby, leaveat expect a time format such as 8:45 for 8:45 am

            results = []
            query = {}

            if domain == 'attraction' and 'entrancefee' in constraints:
                constraints['entrance fee'] = constraints.pop('entrancefee')

            for key in self.data_keys[domain]:
                query[key] = constraints.get(key, [])
                if len(query[key]) > 0 and key in ['arriveby', 'leaveat']:
                    if isinstance(query[key][0], str):
                        query[key] = [query[key]]
                    query[key] = [self.time_str_to_minutes(x) for x in query[key]]
                    query[key] = list(set(query[key]))

            for i, item in enumerate(self.data[domain]):
                for k, v in query.items():
                    if k not in item:
                        continue
                    if len(v) == 0 or item[k] == '?':
                        continue

                    if k == 'arriveby':

                        # accept item[k] if it is earlier than times in the query
                        # if the database entry is not ok:
                        #     break
                        for t in v:
                            if item[k] != ":":
                                if item[k] < t:
                                    break

                    elif k == 'leaveat':

                        # accept item[k] if it is later than times in the query
                        # if the database entry is not ok:
                        #     break
                        for t in v:
                            if item[k] != ":":
                                if item[k] > v[0]:
                                    break

                    else:

                        # accept item[k] if it matches to the values in query

                        # Consider using fuzzy matching! See `partial_ratio` method in the fuzzywuzzy library.
                        # Also, take a look into self.FUZZY_KEYS which stores slots suitable for being done in a fuzzy way.

                        # if the database entry is not ok:
                        #     break

                        # Make sure we are processing a list of lowercase strings
                        if isinstance(v, str):
                            v = [v.strip().lower()]
                        else:
                            v = [x.strip().lower() for x in v]

                        if k in self.FUZZY_KEYS[domain]:
                            f = (lambda x: fuzz.partial_ratio(item[k].lower(), x) > fuzzy_ratio)
                        else:
                            f = (lambda x: item[k].lower() == x)
                        if not any(f(x) for x in v):
                            break

                else: # This gets executed iff the above loop is not terminated
                    result = copy.deepcopy(item)
                    if domain in ['train', 'hotel', 'restaurant']:
                        ref = constraints.get('ref', [])
                        result['ref'] = '{0:08d}'.format(i) if len(ref) == 0 else ref

                    results.append(result)

            if domain == 'attraction':
               for result in results:
                   result['entrancefee'] = result.pop('entrance fee')

            return results
        else:
            return []
