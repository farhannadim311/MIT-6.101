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
    bacon_set = set()
    idx = 0
    result = [transformed_data.get(4724, set())]
    explored = [4724]
    if (n == 0):
        return 4724
    if (n == 1):
        return transformed_data.get(4724, frozenset())
    for i in range(n):
        for k in result[i]:
            if (k in explored):
                pass
            else:
                result.append(transformed_data.get(k, frozenset()))
                explored.append(k)
    explored = set(explored)
    for i in result:
        bacon_set = bacon_set ^ i
    bacon_set = bacon_set - explored
    return bacon_set

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
    print(db)
    # additional code here will be run only when lab.py is invoked directly
    # (not when imported from test.py), so this is a good place to put code
    # used, for example, to generate the results for the online questions.
