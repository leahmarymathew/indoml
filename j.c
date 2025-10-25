#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, np;
    const int N = 4;
    char B[N][N];
    char local_row[N];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (np != N) {
        if (rank == 0) {
            printf("This program must be run with exactly %d processes.\n", N);
            printf("Please run again: mpirun -np %d ./your_program_name\n", N);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        B[0][0] = 'A'; B[0][1] = 'B'; B[0][2] = 'C'; B[0][3] = 'D';
        B[1][0] = 'E'; B[1][1] = 'F'; B[1][2] = 'G'; B[1][3] = 'H';
        B[2][0] = 'I'; B[2][1] = 'J'; B[2][2] = 'K'; B[2][3] = 'L';
        B[3][0] = 'M'; B[3][1] = 'N'; B[3][2] = 'O'; B[3][3] = 'P';
    }

    MPI_Scatter(B, N, MPI_CHAR, local_row, N, MPI_CHAR, 0, MPI_COMM_WORLD);

    printf("Process %d received row: %c, %c, %c, %c\n", 
           rank, local_row[0], local_row[1], local_row[2], local_row[3]);

    MPI_Finalize();
    return 0;
}

