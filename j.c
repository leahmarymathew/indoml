#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank;
    int a = 9; 
    int b = 7;
    int c;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        c = a & b;
        printf("Rank %d: c = a & b = %d\n", rank, c);
    } else if (rank == 1) {
        c = a | b;
        printf("Rank %d: c = a | b = %d\n", rank, c);
    } else {
        c = a ^ b;
        printf("Rank %d: c = a ^ b = %d\n", rank, c);
    }

    MPI_Finalize();
    return 0;
}

