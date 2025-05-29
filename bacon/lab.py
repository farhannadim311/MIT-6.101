"""
6.101 Lab:
Bacon Number
"""

#!/usr/bin/env python3

import pickle
# import typing # optional import
# import pprint # optional import

# NO ADDITIONAL IMPORTS ALLOWED!


def transform_data(raw_data):
    """
    Transforming the list into a set of tuples for efficiency
    """
    graph = dict()
    for a1,a2, _ in raw_data:
        if a1 not in graph:
            graph[a1] = set()
        if a2 not in graph:
            graph[a2] = set()
        graph[a1].add(a2)
        graph[a2].add(a1)
    return graph
def acted_together(transformed_data, actor_id_1, actor_id_2):
    if(actor_id_1 == actor_id_2):
        return True
    return actor_id_2 in transformed_data.get(actor_id_1, set())

def actors_with_bacon_number(transformed_data, n):
    raise NotImplementedError("Implement me!")


def bacon_path(transformed_data, actor_id):
    raise NotImplementedError("Implement me!")


def actor_to_actor_path(transformed_data, actor_id_1, actor_id_2):
    raise NotImplementedError("Implement me!")


def actor_path(transformed_data, actor_id_1, goal_test_function):
    raise NotImplementedError("Implement me!")


def actors_connecting_films(transformed_data, film1, film2):
    raise NotImplementedError("Implement me!")


if __name__ == "__main__":
    with open("resources/small.pickle","rb") as f:
        db = pickle.load(f)
    with open("resources/names.pickle","rb") as f:
        name = pickle.load(f)
    actor_id1 = name["Joseph McKenna"]
    actor_id2 = name["Dean Paraskevopoulos"]
    data = transform_data(db)
    print(acted_together(data, actor_id1, actor_id2))
    # additional code here will be run only when lab.py is invoked directly
    # (not when imported from test.py), so this is a good place to put code
    # used, for example, to generate the results for the online questions.
