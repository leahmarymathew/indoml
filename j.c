#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, np;
    int N, rows_per_proc, i, j;
    double *A, *x, *y, *local_A, *local_y;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (rank == 0) {
        printf("Enter the dimension N of the N*N matrix (must be >= 3): ");
        scanf("%d", &N);

        if (N < 3) {
            printf("Matrix size must be at least 3*3.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (N % np != 0) {
            printf("Matrix dimension N (%d) must be divisible by the number of processes (%d).\n", N, np);
            printf("Please run again with a compatible number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    rows_per_proc = N / np;

    x = (double*)malloc(N * sizeof(double));
    y = (double*)malloc(N * sizeof(double));
    local_y = (double*)malloc(rows_per_proc * sizeof(double));
    local_A = (double*)malloc(rows_per_proc * N * sizeof(double));

    if (rank == 0) {
        A = (double*)malloc(N * N * sizeof(double));
        printf("Enter the %d elements of the %d*%d matrix A (row-wise):\n", N*N, N, N);
        for (i = 0; i < N * N; i++) {
            scanf("%lf", &A[i]);
        }
        
        printf("Enter the %d elements of the vector x:\n", N);
        for (i = 0; i < N; i++) {
            scanf("%lf", &x[i]);
        }
    }

    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE, 
                local_A, rows_per_proc * N, MPI_DOUBLE, 
                0, MPI_COMM_WORLD);

    for (i = 0; i < rows_per_proc; i++) {
        local_y[i] = 0.0;
        for (j = 0; j < N; j++) {
            local_y[i] += local_A[i * N + j] * x[j];
        }
    }

    MPI_Allgather(local_y, rows_per_proc, MPI_DOUBLE, 
                  y, rows_per_proc, MPI_DOUBLE, 
                  MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nResult vector y = A*x:\n");
        for (i = 0; i < N; i++) {
            printf("y[%d] = %f\n", i, y[i]);
        }
        free(A);
    }

    free(x);
    free(y);
    free(local_A);
    free(local_y);

    MPI_Finalize();
    return 0;
}

