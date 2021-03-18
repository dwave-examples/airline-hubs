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

import imageio
import itertools
import matplotlib
import os
import numpy as np
import networkx as nx
from dimod import DiscreteQuadraticModel
from dwave.system import LeapHybridDQMSampler
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

def read_inputs(flow_file, cost_file, verbose=True):
    # Import the flow and cost information for 25 cities

    if verbose:
        print("\nReading in flow/cost info...\n")

    W = np.genfromtxt(flow_file, delimiter=',')
    W = W/np.sum(np.sum(W))
    C = np.genfromtxt(cost_file, delimiter=',')
    n = W.shape[0]

    return W, C, n

def read_city_info(file_name, verbose=True):

    file1 = open(file_name, 'r') 
    Lines = file1.readlines()
    city_names = []
    city_lats = []
    city_longs = []

    # Strips the newline character 
    for line in Lines: 
        info = line.split(",")
        city_names.append(info[1])
        city_lats.append(float(info[2]))
        city_longs.append(float(info[3].strip()))

    file1.close() 
    if verbose:
        print("\nProcessed", info[0], "city locations.\n")

    return city_names, city_lats, city_longs

def build_graph(dist_mat, city_names, verbose=True):

    if verbose:
        print("\nConstructing map...\n")

    G = nx.Graph()

    num_cities = len(city_names)
    for i in range(num_cities):
        for j in range(i+1, num_cities):
            G.add_edge(city_names[i], city_names[j], weight=dist_mat[i,j])

    return G

def draw_graph(G, city_names, city_lats, city_longs):
    
    positions = {}
    for i in range(len(city_names)):
        positions[city_names[i]] = [-city_longs[i], city_lats[i]]

    nx.draw(G, pos = positions, with_labels=True)
    plt.savefig('complete_network.png')
    plt.close()

def build_dqm(W, C, n, p, a, verbose=True):

    if verbose:
        print("\nBuilding DQM...\n")

    dqm = DiscreteQuadraticModel()
    for i in range(n):
        dqm.add_variable(n, label=i)

    # add objective
    for i in range(n):
        for j in range(n):
            for k in range(n):
                dqm.set_linear_case(i, k, C[i][k]*W[i][j])
            for m in range(n):
                dqm.set_linear_case(j, m, dqm.get_linear_case(j,m)+C[j][m]*W[i][j])
            for k in range(n):
                for m in range(n):
                    if i != j:
                        dqm.set_quadratic_case(i, k, j, m, a*C[k][m]*W[i][j])

    # constraint 1
    gamma1 = 30
    for i in range(n):
        for j in range(n):
            dqm.set_linear_case(i,j, dqm.get_linear_case(i,j) + 1*gamma1)
            if i != j:
                dqm.set_quadratic_case(i, j, j, j, dqm.get_quadratic_case(i, j, j, j) - 1*gamma1)

    # constraint 3
    gamma3 = 20
    for j in range(n):
        dqm.set_linear_case(j, j, dqm.get_linear_case(j,j) + (1-2*p)*gamma3)
        for k in range(j+1,n):
            dqm.set_quadratic_case(j, j, k, k, dqm.get_quadratic_case(j, j, k, k) + 2*gamma3)

    return dqm

def get_layout_from_sample(ss, city_names, p):

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

def get_cost(index, ss, a, dist_mat, cost_mat):

    cost = 0
    for i in range(n):
        for j in range(i+1, n):
            cost += dist_mat[i][j]*(cost_mat[i][ss[i]] + cost_mat[j][ss[j]] + a*cost_mat[ss[i]][ss[j]])

    return cost

def visualize_results(dist_mat, city_names, hubs, legs, city_lats, city_longs, cost, filenames=None, counter=0):

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
    ax.set_title(f'Cost: {cost}')

    nx.draw_networkx_nodes(H, node_size=[v * 10 for v in d.values()], pos = positions, edgecolors='k', ax=ax)
    nx.draw_networkx_nodes(hubs, node_size = [v * 100 for v in hub_degrees.values()], pos=positions, node_color='r', edgecolors='k', ax=ax)
    nx.draw_networkx_edges(H, pos=positions, edgelist=H.edges(), width=1.0, ax=ax)
    nx.draw_networkx_edges(H, pos=positions, edgelist=hub_cxn, width=3.0, ax=ax)

    hub_graph = H.subgraph(hubs)
    nx.draw_networkx_labels(hub_graph, pos=positions, ax=ax)

    filename = f'{counter}.png'
    filenames.append(filename)

    plt.savefig(filename)
    plt.close()

    return filenames

if __name__ == '__main__':

    W, C, n = read_inputs(flow_file='flow.csv', cost_file='cost.csv')
    city_names, city_lats, city_longs = read_city_info('city-data.txt')
    p = 3 # number of hubs
    a = 0.4 # discount for hub-hub routes

    # Uncomment lines below to visualize total network options
    # G = build_graph(W, city_names)
    # draw_graph(G, city_names, city_lats, city_longs)

    dqm = build_dqm(W, C, n, p, a)

    print("\nRunning hybrid solver...\n")
    sampler = LeapHybridDQMSampler()
    sampleset = sampler.sample_dqm(dqm, label='Example - DQM Airline Hubs')

    print("\nInterpreting solutions...\n")

    ss = list(sampleset.data(['sample']))

    cost_dict = {index: get_cost(index, ss[index].sample, a, W, C) for index in range(len(ss))}

    ordered_samples = dict(sorted(cost_dict.items(), key=lambda item: item[1], reverse=True))
    filenames = []
    counter = 0
    print("\nGenerating images...\n")
    for key, val in ordered_samples.items():
        hubs, legs, valid = get_layout_from_sample(ss[key].sample, city_names, p)
        if counter > 0:
            if prev_val == val:
                valid = False
        if valid:
            filenames = visualize_results(W, city_names, hubs, legs, city_lats, city_longs, cost_dict[key], filenames, counter)
            counter += 1
        prev_val = val

    # build gif
    print("\nBuilding output GIF...\n")
    with imageio.get_writer('airline-hubs.gif', mode='I') as writer:
        for filename in filenames:
            for i in range(5):
                image = imageio.imread(filename)
                writer.append_data(image)
        for i in range(20):
            image = imageio.imread(filenames[-1])
            writer.append_data(image)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
