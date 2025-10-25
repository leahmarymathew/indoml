#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("Rank %d: Name: Leah, Roll Number: 2023BCS0190\n", rank);

    MPI_Finalize();
    return 0;
}
