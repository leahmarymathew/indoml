/**
 * @file mpi_info.c
 * @brief An MPI program in C to print name, roll number, rank, and time.
 *
 * This program initializes an MPI environment and has each process
 * print its rank (process ID), predefined name and roll number,
 * and the current system time.
 *
 * It is intended to be run with 4 processes as per the requirement.
 *
 * To compile:
 * mpicc mpi_info.c -o mpi_info
 *
 * To run (with 4 processes):
 * mpirun -np 4 ./mpi_info
 */

#include <mpi.h>     // For MPI functions
#include <stdio.h>   // For printf
#include <time.h>    // For time() and ctime()
#include <string.h>  // For strcspn (to remove newline from ctime)

int main(int argc, char** argv) {
    
    // --- User-specific data ---
    // !! PLEASE REPLACE these placeholders with your actual details !!
    const char* my_name = "Your Name"; 
    const char* my_roll_no = "123456789";
    // -------------------------

    // Initialize the MPI environment
    // This must be the first MPI call
    MPI_Init(&argc, &argv);

    // Get the total number of processes in the communicator (MPI_COMM_WORLD)
    // We expect this to be 4 when run as `mpirun -np 4`
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank (ID) of the current process
    // Ranks will be from 0 to (world_size - 1)
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the current system time
    time_t current_time;
    char* time_string;

    // time(NULL) gets the current time in seconds since the epoch
    current_time = time(NULL);
    // ctime() converts the time_t value to a human-readable string
    // Note: ctime() includes a newline character ('\n') at the end.
    time_string = ctime(&current_time);

    // Remove the trailing newline character from the time string for cleaner output
    // strcspn finds the first occurrence of '\n' and we replace it with '\0' (null terminator)
    time_string[strcspn(time_string, "\n")] = 0;

    // Print the required information from this specific process
    // [Rank]: Name, RollNo. Time: [Current Time]
    printf("Process %d: %s, %s. Time: %s\n", 
           world_rank, my_name, my_roll_no, time_string);
    
    // It's good practice to flush stdout in parallel programs
    // to ensure output isn't buffered and mixed up.
    fflush(stdout);

    // Finalize the MPI environment
    // This must be the last MPI call
    MPI_Finalize();

    return 0;
}
