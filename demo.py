# Copyright 2021 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import itertools
from collections import defaultdict

import imageio
import matplotlib
import numpy as np
import networkx as nx
from dimod import DiscreteQuadraticModel
from dwave.system import LeapHybridDQMSampler

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

def read_inputs(flow_file, cost_file, verbose=True):
    """Reads in scenario information on passenger flow and route cost.

    Args:
        - flow_file: CSV file. Number of passengers that desire to travel from city i to city j.
        - cost_file: CSV file. Cost for airline to operate leg from city i to city j.
        - verbose: Print to command-line for user.

    Returns:
        - W: Numpy matrix. Represents passenger demand. Normalized with total demand equal to 1.
        - C: Numpy matrix. Represents airline leg cost.
        - n: Int. Number of cities in play.
    """

    if verbose:
        print("\nReading in flow/cost info...\n")

    W = np.genfromtxt(flow_file, delimiter=',')
    W = W/np.sum(np.sum(W))
    C = np.genfromtxt(cost_file, delimiter=',')
    n = W.shape[0]

    return W, C, n

def read_city_info(file_name, verbose=True):
    """Reads in scenario information on airports and lat/long coordinates.

    Args:
        - file_name: Text file. Includes airport code and lat/long coordinates.
        - verbose: Print to command-line for user.

    Returns:
        - city_names: List of all airport codes.
        - city_lats: List of all airport lat coordinates.
        - city_longs: List of all airport long coordinates.

    All returned lists have airports in the same order, i.e. airport city_names[i]
    has latitude city_lats[i] and longitude city_longs[i].
    """

    file1 = open(file_name, 'r') 
    lines = file1.readlines()
    city_names = []
    city_lats = []
    city_longs = []

    # Strips the newline character 
    for line in lines: 
        info = line.split(",")
        city_names.append(info[1])
        city_lats.append(float(info[2]))
        city_longs.append(float(info[3].strip()))

    file1.close() 
    if verbose:
        print("\nProcessed", info[0], "city locations.\n")

    return city_names, city_lats, city_longs

def build_graph(dist_mat, city_names, verbose=True):
    """Builds weighted graph based on cities and distances.

    Args:
        - dist_mat: Numpy matrix providing distance between cities i and j.
        - city_names: List of all airport codes.
        - verbose: Print to command-line for user.

    Returns:
        - G: NetworkX weighted graph of cities with distances on edges.
    """

    if verbose:
        print("\nConstructing map...\n")

    G = nx.Graph()

    num_cities = len(city_names)
    for i in range(num_cities):
        for j in range(i+1, num_cities):
            G.add_edge(city_names[i], city_names[j], weight=dist_mat[i,j])

    return G

def draw_graph(G, city_names, city_lats, city_longs):
    """Visualizes the city graph and saves as file.

    Args:
        - G: NetworkX weighted graph of cities with distances on edges.
        - city_names: List of all airport codes.
        - city_lats: List of all airport lat coordinates.
        - city_longs: List of all airport long coordinates.

    All city info lists have airports in the same order, i.e. airport city_names[i]
    has latitude city_lats[i] and longitude city_longs[i].

    Returns:
        None. Saves visual as 'complete_network.png'.
    """
    
    positions = {}
    for i in range(len(city_names)):
        positions[city_names[i]] = [-city_longs[i], city_lats[i]]

    nx.draw(G, pos=positions, with_labels=True)
    plt.savefig('complete_network.png')
    plt.close()

def build_dqm(W, C, n, p, a, verbose=True):
    """Builds discrete quadratic model representing the optimization problem.

    Args:
        - W: Numpy matrix. Represents passenger demand. Normalized with total demand equal to 1.
        - C: Numpy matrix. Represents airline leg cost.
        - n: Int. Number of cities in play.
        - p: Int. Number of hubs airports allowed.
        - a: Float in [0.0, 1.0]. Discount allowed for hub-hub legs.
        - verbose: Print to command-line for user.

    Returns:
        - dqm: DiscreteQuadraticModel representing the optimization problem.
    """

    if verbose:
        print("\nBuilding DQM...\n")

    # Initialize DQM object.
    dqm = DiscreteQuadraticModel()
    for i in range(n):
        dqm.add_variable(n, label=i)

    # Objective: Minimize cost.
    for i in range(n):
        for j in range(n):
            for k in range(n):
                dqm.set_linear_case(i, k, dqm.get_linear_case(i,k)+C[i][k]*W[i][j])
                dqm.set_linear_case(j, k, dqm.get_linear_case(j,k)+C[j][k]*W[i][j])     
                for m in range(n):
                    if i != j:
                        dqm.set_quadratic_case(i, k, j, m, a*C[k][m]*W[i][j])

    # Constraint: Every leg must connect to a hub.
    gamma1 = 150
    for i in range(n):
        for j in range(n):
            if i != j:
                dqm.set_linear_case(i, j, dqm.get_linear_case(i,j) + 1*gamma1)
                dqm.set_quadratic_case(i, j, j, j, dqm.get_quadratic_case(i, j, j, j) - 1*gamma1)

    # Constraint: Exactly p hubs required.
    gamma2 = 250
    for i in range(n):
        dqm.set_linear_case(i, i, dqm.get_linear_case(i,i) + (1-2*p)*gamma2)
        for j in range(i+1,n):
            dqm.set_quadratic_case(i, i, j, j, dqm.get_quadratic_case(i, i, j, j) + 2*gamma2)

    return dqm

def get_layout_from_sample(ss, city_names, p):
    """Determines the airline route network from a sampleset.

    Args:
        - ss: Sampleset dictionary. One solution returned from the hybrid solver.
        - city_names: List of all airport codes, in order.
        - p: Int. Number of hubs airports allowed.

    Returns:
        - hubs: List of airports designated as hubs.
        - legs: List of airline city-city route legs that will be operated.
        - valid: Boolean designated whether provided solution satisfies the constraints.
    """

    hubs = []
    legs = []
    valid = True
    for key, val in ss.items():
        if key == val:
            hubs.append(city_names[key])
        else:
            legs.append((city_names[key],city_names[val]))
            if ss[val] != val:
                valid = False

    if len(hubs) != p:
        valid = False

    return hubs, legs, valid

def get_cost(ss, a, dist_mat, C, n):
    """Determines the cost of an airline route network from a sampleset.

    Args:
        - ss: Sampleset dictionary. One solution returned from the hybrid solver.
        - a: Float in [0.0, 1.0]. Discount allowed for hub-hub legs.
        - dist_mat: Numpy matrix providing distance between cities i and j.
        - C: Numpy matrix. Represents airline leg cost.
        - n: Int. Number of cities in play.

    Returns:
        - cost: Cost of provided route network.
    """

    cost = 0
    for i in range(n):
        for j in range(i+1, n):
            cost += dist_mat[i][j]*(C[i][ss[i]] + C[j][ss[j]] + a*C[ss[i]][ss[j]])

    return cost

def visualize_results(dist_mat, city_names, hubs, legs, city_lats, city_longs, cost, filenames=None, counter=0, verbose=True):
    """Visualizes a given route layout and saves the file as a .png.

    Args:
        - dist_mat: Numpy matrix providing distance between cities i and j.
        - city_names: List of all airport codes.
        - hubs: List of airports designated as hubs.
        - legs: List of airline city-city route legs that will be operated.
        - city_lats: List of all airport lat coordinates, in order.
        - city_longs: List of all airport long coordinates, in order.
        - cost: Cost of provided route network.
        - filenames: List of image filenames produced so far.
        - counter: Counter for image filename.
        - verbose: Print results to command-line.

    Returns:
        - filenames: List of image filenames produced so far with new image filename appended.
    """

    if filenames is None:
        filenames = []
    
    num_cities = len(city_names)

    positions = {city_names[i]: [-city_longs[i], city_lats[i]] for i in range(num_cities)}

    hub_cxn = list(itertools.combinations(hubs, 2))

    H = nx.Graph()
    H.add_nodes_from(city_names)
    H.add_edges_from(legs)

    d = dict(H.degree)
    hub_degrees = {k:d[k]+len(hubs)-1 for k in hubs if k in d}

    plt.figure(figsize=(10,5))
    ax = plt.gca()
    ax.set_title("Cost: {}".format(cost))

    nx.draw_networkx_nodes(H, node_size=[v * 10 for v in d.values()], pos=positions, edgecolors='k', ax=ax)
    nx.draw_networkx_nodes(hubs, node_size=[v * 100 for v in hub_degrees.values()], pos=positions, node_color='r', edgecolors='k', ax=ax)
    nx.draw_networkx_edges(H, pos=positions, edgelist=H.edges(), width=1.0, ax=ax)
    nx.draw_networkx_edges(H, pos=positions, edgelist=hub_cxn, width=3.0, ax=ax)

    hub_graph = H.subgraph(hubs)
    nx.draw_networkx_labels(hub_graph, pos=positions, ax=ax)

    filename = str(counter)+'.png'
    filenames.append(filename)

    plt.savefig(filename)
    plt.close()

    if verbose:
        print("Hubs:", hubs, "\tCost:", cost)

    return filenames

if __name__ == '__main__':

    passenger_demand, leg_cost, num_cities = read_inputs(flow_file='flow.csv', cost_file='cost.csv')
    city_names, city_lats, city_longs = read_city_info('city-data.txt')
    p = 3 # number of hubs
    a = 0.4 # discount for hub-hub routes

    # Uncomment lines below to visualize total network options
    # G = build_graph(passenger_demand, city_names)
    # draw_graph(G, city_names, city_lats, city_longs)

    dqm = build_dqm(passenger_demand, leg_cost, num_cities, p, a)

    print("\nRunning hybrid solver...\n")
    sampler = LeapHybridDQMSampler()
    sampleset = sampler.sample_dqm(dqm, label='Example - DQM Airline Hubs')

    print("\nInterpreting solutions...\n")

    ss = list(sampleset.data(['sample']))

    cost_dict = {index: get_cost(ss[index].sample, a, passenger_demand, leg_cost, num_cities) for index in range(len(ss))}

    ordered_samples = dict(sorted(cost_dict.items(), key=lambda item: item[1], reverse=True))
    filenames = []
    counter = 0
    print("\nGenerating images for output GIF...\n")
    print("\nFeasible solutions found:")
    print("---------------------------\n")
    output_string = []
    for key, val in ordered_samples.items():
        hubs, legs, valid = get_layout_from_sample(ss[key].sample, city_names, p)
        if counter > 0:
            if prev_val == val:
                valid = False
        if valid:
            filenames = visualize_results(passenger_demand, city_names, hubs, legs, city_lats, city_longs, cost_dict[key], filenames, counter, verbose=False)
            output_string.append("Hubs: "+str(hubs)+"\tCost: "+str(cost_dict[key]))
            counter += 1
        prev_val = val

    output_string.reverse()
    for line in output_string:
        print(line)

    # build gif
    print("\nSaving best solution to best_soln_found.png...\n")
    img = plt.imread(filenames[-1])
    fig = plt.figure(dpi=100, tight_layout=True, frameon=False, figsize=(img.shape[1]/100.,img.shape[0]/100.)) 
    fig.figimage(img, cmap=plt.cm.binary)
    fig.suptitle('Best Solution Found', x=0.5, y=0.08, fontsize=16)
    plt.savefig("best_soln_found.png")
    plt.close(fig)
    
    print("\nBuilding output GIF (airline-hubs.gif)...\n")
    with imageio.get_writer('airline-hubs.gif', mode='I') as writer:
        for filename in filenames:
            for i in range(15):
                image = imageio.imread(filename)
                writer.append_data(image)
        for i in range(40):
            image = imageio.imread("best_soln_found.png")
            writer.append_data(image)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
