#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

double function_to_integrate(double x, int P) {
    if (P == 1) {
        return x * x;
    } else if (P == 2) {
        return exp(-x * x);
    } else {
        return 0.0;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Command-line argument processing
    int P = 1;            // Integral selector: 1 or 2
    int N = 1000000;      // Total number of samples

    if (argc == 5) {
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "-P") {
                P = std::atoi(argv[++i]);
            } else if (std::string(argv[i]) == "-N") {
                N = std::atoi(argv[++i]);
            }
        }
    }

    // Broadcast P and N to all processes
    MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process calculates its share of random samples
    int local_N = N / size;
    double local_sum = 0.0;
    double x, fx;

    // Seed the random number generator differently for each process
    std::srand(time(0) + rank);

    for (int i = 0; i < local_N; ++i) {
        x = static_cast<double>(std::rand()) / RAND_MAX; // Random x in [0, 1]
        fx = function_to_integrate(x, P);               // Evaluate the function
        local_sum += fx;
    }

    // Each process has a partial sum; now gather and sum them on root
    double total_sum = 0.0;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double integral_estimate = total_sum / N; // Average value over [0,1]
        if (P == 1) {
            std::cout << "Estimated integral of x^2 from 0 to 1: " << integral_estimate << std::endl;
            std::cout << "Expected value: 1/3 ≈ 0.3333" << std::endl;
        } else if (P == 2) {
            std::cout << "Estimated integral of exp(-x^2) from 0 to 1: " << integral_estimate << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}