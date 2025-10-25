#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    int rank, size;
    const int MAX_STRING = 100;
    char message1[MAX_STRING];
    char message2[MAX_STRING];
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            printf("Error: This program requires at least 2 processes.\n");
            printf("Run with: mpirun -np 2 ./your_program_name\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        strcpy(message1, "Chandrayaan3");
        strcpy(message2, "23-August-2023");
        
        printf("Process 0 (ISRO) sending: %s\n", message1);
        MPI_Send(message1, strlen(message1) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        
        printf("Process 0 (ISRO) sending: %s\n", message2);
        MPI_Send(message2, strlen(message2) + 1, MPI_CHAR, 1, 1, MPI_COMM_WORLD);
        
        printf("Process 0 (ISRO) has sent all data.\n");

    } else if (rank == 1) {
        
        printf("Process 1 (Moon) is waiting for data...\n");
        
        MPI_Recv(message1, MAX_STRING, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        printf("Process 1 (Moon) received: %s\n", message1);
        
        MPI_Recv(message2, MAX_STRING, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &status);
        printf("Process 1 (Moon) received: %s\n", message2);
        
        printf("Process 1 (Moon) has received all data.\n");
    }

    MPI_Finalize();
    return 0;
}

