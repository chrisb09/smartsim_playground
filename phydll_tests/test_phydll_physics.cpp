#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>

#include "ml_coupling.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    std::string mode = "perfect";
    if (argc > 1) {
        mode = argv[1];
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int field_size = 10;
    std::string model_path = "minimal_model.pt";

    if (mode == "imperfect") {
        if (rank == 0) {
            field_size = 11;
        } else {
            field_size = 10;
        }
    } else if (mode == "shape_mismatch") {
        field_size = 10;
        model_path = "shape_mismatch_model.pt";
    }

    // Prepare configuration TOML
    std::string config_str = 
        "[general]\n"
        "coupling_type = \"STATIC\"\n"
        "[logging]\n"
        "level = \"debug\"\n"
        "error_separate = false\n"
        "[provider]\n"
        "class = \"Phydll\"\n"
        "backend = \"TORCH\"\n"
        "model_file = \"" + model_path + "\"\n"
        "device = \"CPU\"\n"
        "[behavior]\n"
        "class=\"periodic\"\n"
        "inference_interval = 1\n"
        "[normalization]\n"
        "class=\"minmax\"\n"
        "input_min=-10.0\n"
        "input_max=10.0\n"
        "output_min=-10.0\n"
        "output_max=10.0\n"
        "[application]\n"
        "class=\"MLCouplingApplicationTurbulenceClosure\"\n";

    std::vector<double> input_raw(field_size, rank + 1.0);
    std::vector<double> output_raw(field_size, 0.0);

    MLCouplingData<double> input_data;
    input_data.add_tensor(MLCouplingTensor<double>::wrap_flat(input_raw.data(), std::vector<int>{field_size}));

    MLCouplingData<double> output_data;
    output_data.add_tensor(MLCouplingTensor<double>::wrap_flat(output_raw.data(), std::vector<int>{field_size}));

    if (rank == 0) {
        std::cout << "Initializing MLCoupling for mode: " << mode << "..." << std::endl;
    }

    try {
        MLCoupling<double, double>* coupling = create_mlcoupling_from_config<double, double>(
            config_str, input_data, output_data
        );

        if (rank == 0) {
            std::cout << "Starting C++ ML Coupling step for mode: " << mode << std::endl;
        }

        // Run 2 steps
        for (int iter = 1; iter <= 2; ++iter) {
            coupling->step();
            if (rank == 0) {
                std::cout << "Iter " << iter << " Completed. Output sample [0]: " << output_raw[0] << std::endl;
            }
        }

        delete coupling;
    } catch (const std::exception& e) {
        std::cerr << "Exception on rank " << rank << ": " << e.what() << std::endl;
    }

    MPI_Finalize();
    return 0;
}
