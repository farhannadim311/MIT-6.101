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
    graph = dict()
    for a1,a2, m in raw_data:
        if a1 not in graph:
            graph[a1] = []
        if a2 not in graph:
            graph[a2] = []
        if str(m) not in graph:
            graph[str(m)] = []
        if a2 not in graph[a1] and a2 != a1:
            graph[a1].append(a2)
        if a1 not in graph[a2] and a1 != a2:
            graph[a2].append(a1) 
        if str(m) not in graph[a1]: 
            graph[a1].append(str(m))
        if str(m) not in graph[a2]:  
            graph[a2].append(str(m))   
        if a1 not in graph[str(m)]:    
            graph[str(m)].append(a1)  
        if a2 not in graph[str(m)]:    
            graph[str(m)].append(a2)  
    return graph


def movie_helper(raw_data):
    graph = dict()
    for a1, a2, m in raw_data:
        if a1 not in graph:
            graph[a1] = set()
        if a2 not in graph:
            graph[a2] = set()
        graph[a1].add(m)
        graph[a2].add(m)
    return graph

def actors(transformed_data, actor):
    lst = transformed_data[actor]
    ret = []
    for k in lst:
        if (isinstance(k, int)):
            ret.append(k)
    return ret

def movies(transformed_data, movie):
    lst = transformed_data[str(movie)]
    ret = []
    for k in lst:
        ret.append(k)
    return ret

def actors_in_movies(transformed_data, actor):
    lst = transformed_data[actor]
    ret = []
    for k in lst:
        if (isinstance(k, str)):
            ret.append(int(k))
    return ret

def acted_together(transformed_data, actor_id_1, actor_id_2):
    if(actor_id_1 == actor_id_2):
        return True
    return actor_id_2 in transformed_data.get(actor_id_1, set())

def actors_with_bacon_number(transformed_data, n):
    if n == 0:
        return {4724}
    
    visited = set([4724])
    queue = [(4724, 0)]  
    result = set()

    while queue:
        actor, depth = queue.pop(0)  
        if depth == n:
            result.add(actor)
        elif depth < n:
            for neighbor in actors(transformed_data, actor):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
    
    return result

def bacon_path(transformed_data, actor_id):
    return actor_path(transformed_data, 4724, lambda x: x == actor_id)


def actor_to_actor_path(transformed_data, actor_id_1, actor_id_2):
    return actor_path(transformed_data, actor_id_1, lambda x: x == actor_id_2)

def actor_path(transformed_data, actor_id_1, goal_test_function):
    if(goal_test_function(actor_id_1)):
        return [actor_id_1]
    visited = set([actor_id_1])
    possible_paths = [(actor_id_1, )]
    path = []
    while(possible_paths):
        path = possible_paths.pop(0)
        curr_loc = path[-1]
        if (goal_test_function(curr_loc)):
            return path
        for neighbor in actors(transformed_data, curr_loc):
            if neighbor not in visited:
                visited.add(neighbor)
                possible_paths.append(path + (neighbor,))
    return None


def movie_path(transformed_data, actor_id_1, actor_id_2, movie_data):
    if(actor_id_1 == actor_id_2):
        return movie_data[actor_id_1]
    path = actor_to_actor_path(transformed_data, actor_id_1, actor_id_2)
    if(path == None):
        return None
    path = list(path)
    movie_visited = set()
    movie = []
    while(path):
        a1 = path.pop(0)
        if a1 == actor_id_2:
            return movie
        s = movie_data[a1]
        t = next(iter(s))
        if t not in movie_visited:
            movie_visited.add(t)
            movie.append(t)
    if (len(movie) == 0):
        return None
    return movie_path
    
    
def actors_connecting_films(transformed_data, film1, film2):
    # If either film doesn't exist, return None
    if str(film1) not in transformed_data or str(film2) not in transformed_data:
        return None

    # Get the list of actors in each film
    actors_in_film1 = transformed_data[str(film1)]
    actors_in_film2 = set(transformed_data[str(film2)])

    # If both film IDs are the same, return any actor in that film as a singleton list
    if film1 == film2:
        return [actors_in_film1[0]]

    # BFS from all actors in film1, stop at first actor in film2
    visited = set()
    queue = []

    for actor in actors_in_film1:
        queue.append((actor, [actor]))
        visited.add(actor)

    while queue:
        current_actor, path = queue.pop(0)

        if current_actor in actors_in_film2:
            return path

        for neighbor in actors(transformed_data, current_actor):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None




if __name__ == "__main__":

    # Load raw data
    with open("resources/tiny.pickle", "rb") as f:
        raw_data = pickle.load(f)

    print(raw_data)

