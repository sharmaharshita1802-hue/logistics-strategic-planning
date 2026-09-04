"""
Route Optimization (simplified)
--------------------------------
Solves a vehicle routing problem (VRP) for a delivery zone using
Google OR-Tools, minimizing total route distance/cost.
"""

from ortools.constraint_solver import routing_enums_pb2, pywrapcp


def solve_routes(distance_matrix, num_vehicles: int, depot: int):
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    return routing.SolveWithParameters(params)


if __name__ == "__main__":
    # Example toy distance matrix (5 stops incl. depot at index 0)
    distance_matrix = [
        [0, 9, 8, 7, 6],
        [9, 0, 5, 4, 3],
        [8, 5, 0, 3, 2],
        [7, 4, 3, 0, 1],
        [6, 3, 2, 1, 0],
    ]
    solution = solve_routes(distance_matrix, num_vehicles=2, depot=0)
    print("Solved:", solution is not None)
