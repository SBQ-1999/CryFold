#
#
# from ortools.constraint_solver import routing_enums_pb2
# from ortools.constraint_solver import pywrapcp
# import numpy as np
# import os
#
# def create_ortool_data_model(nc_coords,edge_logits, num_vehicles=20,n_c_distance_threshold=2.5):
#     """Stores the data for the problem."""
#     max_Value = 99999999
#     N_C_dist_matrix = np.sqrt(np.sum(np.square((nc_coords[:,1][:,None] - nc_coords[:,0][None])),axis=-1))
#     L = len(N_C_dist_matrix)
#     temp_matrix = np.ones([L,L])*max_Value
#     edge_add = (edge_logits>0.1) * (N_C_dist_matrix<n_c_distance_threshold)
#     temp_matrix[edge_add] = (100*(N_C_dist_matrix[edge_add]/edge_logits[edge_add])).astype(int)
#     W = np.max(temp_matrix[temp_matrix<max_Value])
#     edge_add = (edge_logits<=0.1) * (N_C_dist_matrix<n_c_distance_threshold)
#     temp_matrix[edge_add] = (W+N_C_dist_matrix[edge_add]*100).astype(int)
#     edge_add = N_C_dist_matrix>=n_c_distance_threshold
#     temp_matrix[edge_add] = (2*W+N_C_dist_matrix[edge_add]*200).astype(int)
#     adj_matrix = np.zeros([L+1,L+1])
#     adj_matrix[1:,1:] = temp_matrix
#     print("number of edges in the data: ",len(np.argwhere(adj_matrix!=max_Value)))
#     print("distance mean %f"%np.mean(adj_matrix[adj_matrix!=max_Value]))
#
#     #adj_matrix[adj_matrix==99999]=sys.maxsize
#     adj_matrix[np.arange(1,L+1),np.arange(1,L+1)]=0
#     data = {}
#     data['distance_matrix'] = adj_matrix#final_adj_matrix
#     data['num_vehicles'] = num_vehicles
#     print("number vehicles: ",num_vehicles,"adj matrix shape: ",len(adj_matrix))
#     data['depot'] = 0
#     return data,W
#
# def print_solution_vrp(data, manager, routing, solution):
#     """Prints solution on console."""
#     print(f'Objective: {solution.ObjectiveValue()}')
#     max_route_distance = 0
#     All_Route_List = []
#     for vehicle_id in range(data['num_vehicles']):
#         cur_Route_List =[]
#         index = routing.Start(vehicle_id)
#         plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
#         route_distance = 0
#         while not routing.IsEnd(index):
#             plan_output += ' {} -> '.format(manager.IndexToNode(index))
#             previous_index = index
#             cur_Route_List.append(manager.IndexToNode(index))
#             index = solution.Value(routing.NextVar(index))
#             route_distance += routing.GetArcCostForVehicle(
#                 previous_index, index, vehicle_id)
#         plan_output += '{}\n'.format(manager.IndexToNode(index))
#         cur_Route_List.append(manager.IndexToNode(index))
#         plan_output += 'Distance of the route: {}m\n'.format(route_distance)
#         print(plan_output)
#         All_Route_List.append(cur_Route_List)
#         max_route_distance = max(route_distance, max_route_distance)
#     print('Maximum of the route distances: {}m'.format(max_route_distance))
#     return All_Route_List
#
# def build_travel_path(travel_list):
#     new_travel_list = []
#     for k in range(len(travel_list)):
#         travel_id = int(travel_list[k])
#         if travel_id==0:
#             continue
#         travel_id -=1
#         new_travel_list.append(travel_id)
#     return new_travel_list
# def ortools_build_path(nc_coords,edge_logits,n_c_distance_threshold=2.5):
#
#     max_Value = 99999999
#     #number_vehicles = int(np.ceil(len(coordinate_list) / 100))
#     number_vehicles = int(np.ceil(len(nc_coords) / 100))#find 100 did not work so well for very big structures for new model
#     if number_vehicles<=5:
#         number_vehicles=5
#     ortool_data,W = create_ortool_data_model(nc_coords,edge_logits,number_vehicles,n_c_distance_threshold)
#
#     drop_penalty= int(2*W+n_c_distance_threshold*200)#int(np.max(distance_matrix[distance_matrix!=max_Value]))
#     print("drop penalty %d"%drop_penalty)
#     # Create the routing index manager.
#     manager = pywrapcp.RoutingIndexManager(len(ortool_data['distance_matrix']),
#                                                ortool_data['num_vehicles'], ortool_data['depot'])
#     routing = pywrapcp.RoutingModel(manager)
#
#    # for vehicle_id in range(ortool_data['num_vehicles']):
#     #    routing.ConsiderEmptyRouteCostsForVehicle(True, vehicle_id)
#
#     def distance_callback(from_index, to_index):
#         """Returns the distance between the two nodes."""
#         # Convert from routing variable Index to distance matrix NodeIndex.
#         from_node = manager.IndexToNode(from_index)
#         to_node = manager.IndexToNode(to_index)
#         return ortool_data['distance_matrix'][from_node][to_node]
#         # Create Routing Model.
#
#     print("start routing")
#
#     transit_callback_index = routing.RegisterTransitCallback(distance_callback)
#
#     # Define cost of each arc.
#     routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
#     print("add capacity")
#     # add dimension for vrp program
#     dimension_name = 'Distance'
#     routing.AddDimension(
#         transit_callback_index,
#         0,  # no slack
#         max_Value,  # vehicle maximum travel distance
#         True,  # start cumul to zero
#         dimension_name)
#     print("add penalty")
#     # Allow to drop nodes.
#     penalty = drop_penalty
#     for node in range(1, len(ortool_data['distance_matrix'])):
#         routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
#     # Setting first solution heuristic.
#     print("add params")
#     search_parameters = pywrapcp.DefaultRoutingSearchParameters()
#     search_parameters.first_solution_strategy = (
#          routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
#     # search_parameters.first_solution_strategy = (
#     # routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
#     search_parameters.local_search_metaheuristic = (
#         routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
#     search_parameters.time_limit.seconds = max(3600,int(1000*len(nc_coords)/400)) # few hours search
#     search_parameters.solution_limit = 1000
#     search_parameters.log_search = True
#     # Solve the problem.
#     solution = routing.SolveWithParameters(search_parameters)
#     travel_list = print_solution_vrp(ortool_data, manager, routing, solution)
#     travel_edges = []
#     for j, cur_travel_list in enumerate(travel_list):
#         travel_edge = build_travel_path(cur_travel_list)
#         travel_edges.append(travel_edge)
#     return travel_edges
#
