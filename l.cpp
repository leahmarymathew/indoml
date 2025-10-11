#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

#ifdef MPI_ENABLED
#include <mpi.h>
#endif

const int INF = 1e9;

void printMatrix(const std::vector<std::vector<int>>& matrix, int V) {
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (matrix[i][j] >= INF)
                std::cout << "INF\t";
            else
                std::cout << matrix[i][j] << "\t";
        }
        std::cout << std::endl;
    }
}

void floydWarshallSerial(std::vector<std::vector<int>>& dist, int V) {
    for (int k = 0; k < V; ++k) {
        for (int i = 0; i < V; ++i) {
            for (int j = 0; j < V; ++j) {
                if (dist[i][k] < INF && dist[k][j] < INF) {
                    dist[i][j] = std::min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }
}

void runTest(int test_case_num, int V, std::vector<std::vector<int>> graph) {
    int rank = 0;
    int size = 1;
    double serial_exec_time = 0.0;
    double parallel_exec_time = 0.0;
    
#ifdef MPI_ENABLED
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<std::vector<int>> parallel_graph = graph; // Copy for parallel run
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = MPI_Wtime();

    int rows_per_proc = V / size;
    std::vector<std::vector<int>> local_dist(rows_per_proc, std::vector<int>(V));
    
    MPI_Scatter(parallel_graph.data()->data(), rows_per_proc * V, MPI_INT,
                local_dist.data()->data(), rows_per_proc * V, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> k_row(V);
    for (int k = 0; k < V; ++k) {
        int root_proc = k / rows_per_proc;
        if (rank == root_proc) {
            int local_k = k % rows_per_proc;
            k_row = local_dist[local_k];
        }
        MPI_Bcast(k_row.data(), V, MPI_INT, root_proc, MPI_COMM_WORLD);
        
        for (int i = 0; i < rows_per_proc; ++i) {
            for (int j = 0; j < V; ++j) {
                if (local_dist[i][k] < INF && k_row[j] < INF) {
                    local_dist[i][j] = std::min(local_dist[i][j], local_dist[i][k] + k_row[j]);
                }
            }
        }
    }

    MPI_Gather(local_dist.data()->data(), rows_per_proc * V, MPI_INT,
               parallel_graph.data()->data(), rows_per_proc * V, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = MPI_Wtime();

    if (rank == 0) {
        parallel_exec_time = (end_time - start_time) * 1e6;
        std::cout << "\n Parallel Code Result (Test Case " << test_case_num << ") ---" << std::endl;
        printMatrix(parallel_graph, V);
        
        auto start_serial = std::chrono::high_resolution_clock::now();
        floydWarshallSerial(graph, V);
        auto end_serial = std::chrono::high_resolution_clock::now();
        serial_exec_time = std::chrono::duration<double, std::micro>(end_serial - start_serial).count();

        std::cout << "\n Comparison Table (Test Case " << test_case_num << ") " << std::endl;
        std::cout << "| Number of Nodes, Edges | Serial Code (Time) | Parallel Code (Time) |" << std::endl;
        std::cout << "| (" << V << ", N/A)                  | " << serial_exec_time << " us         | " << parallel_exec_time << " us           |" << std::endl;
    }

#else
    if (rank == 0) {
        std::cout << "\n Serial Code Result (Test Case " << test_case_num << ") ---" << std::endl;
        auto start_serial = std::chrono::high_resolution_clock::now();
        floydWarshallSerial(graph, V);
        auto end_serial = std::chrono::high_resolution_clock::now();
        serial_exec_time = std::chrono::duration<double, std::micro>(end_serial - start_serial).count();
        printMatrix(graph, V);
        std::cout << "\nSerial Execution Time: " << serial_exec_time << " us\n";
    }
#endif
}

int main(int argc, char** argv) {
#ifdef MPI_ENABLED
    MPI_Init(&argc, &argv);
#endif

    int V1 = 6;
    std::vector<std::vector<int>> graph1 = {
        {0, 5, INF, 11, INF, INF}, {5, 0, 9, INF, 3, INF}, {INF, 9, 0, 4, 3, INF},
        {11, INF, 4, 0, INF, 14}, {INF, 3, 3, INF, 0, 7}, {INF, INF, INF, 14, 7, 0}
    };
    runTest(1, V1, graph1);

    int V2 = 6;
    std::vector<std::vector<int>> graph2 = {
        {0, 3, INF, INF, -4, INF}, {INF, 0, 4, INF, INF, 2}, {INF, INF, 0, 8, 1, INF},
        {INF, INF, -5, 0, INF, 6}, {INF, 2, INF, INF, 0, INF}, {3, INF, INF, INF, INF, 0}
    };
    runTest(2, V2, graph2);

#ifdef MPI_ENABLED
    MPI_Finalize();
#endif
    return 0;
}
