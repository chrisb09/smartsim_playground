#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <mpi.h>

#include "../CPP-ML-Interface/include/ml_coupling.hpp"

int main(int argc, char** argv)
{
	const std::string config_path = (argc > 1) ? argv[1] : "config.toml";

	const char* provider = std::getenv("PROVIDER");
	
	if (provider == nullptr || std::string(provider).empty()) {
		std::cerr << "PROVIDER is not set. Aborting.\n";
		return 1;
	}

	bool use_mpi = (std::string(provider) == "AIX");
	if (use_mpi) {
		MPI_Init(&argc, &argv);
	}

	if (std::string(provider) == "SMARTSIM") {
		std::cout << "Running with SmartSim provider\n";
		std::cout << "Checking SSDB environment variable...\n";
		const char* ssdb = std::getenv("SSDB");
		if (ssdb == nullptr || std::string(ssdb).empty()) {
			std::cerr << "SSDB is not set. Aborting.\n";
			if (use_mpi) MPI_Finalize();
			return 1;
		}

		std::cout << "Using SSDB=" << ssdb << "\n";
		std::cout << "Loading config from " << config_path << "\n";
	} else if (std::string(provider) == "AIX") {
		std::cout << "Running with AIX provider\n";
		std::cout << "Loading config from " << config_path << "\n";
	} else {
		std::cerr << "Unsupported provider: " << provider << "\n";
		if (use_mpi) MPI_Finalize();
		return 1;
	}


	// ******************************
	// Create dummy data
	// ******************************

	// 1 input of size Bx18
	float* flat_data = new float[18];
	for (int i = 0; i < 9; ++i) {
		flat_data[i] = (4 + i * 17) % 100; // First 9 values (water)
	}
	for (int i = 0; i < 9; ++i) {
		flat_data[9 + i] = (7 + i * 24) % 200; // Next 9 values (terrain)
	}

	MLCouplingTensor<float> input_tensor = MLCouplingTensor<float>::wrap_flat(
		flat_data,
		std::vector<int>{1, 18},
		MLCouplingMemLayoutContiguous,
		MLCouplingOwnershipExternal);

	MLCouplingData<float> input_data{std::vector<MLCouplingTensor<float>>{
		input_tensor
	}};

	std::cout << "Input data:\n";
	std::cout << input_data.to_string() << "\n";

	MLCouplingData<float> output_data;

    float* output_buffer = new float[1];

	// Just to ensure the buffer is changed, we set it to -1 initially
	output_buffer[0] = -1;

    output_data.add_tensor(MLCouplingTensor<float>::wrap_flat(
		output_buffer,
		std::vector<int>{1},
		MLCouplingMemLayoutContiguous,
		MLCouplingOwnershipExternal
	));

	std::cout << "Output data before inference:\n";
	std::cout << output_data.to_string() << "\n";

	// *************************************
	// Create coupling object
	// *************************************

	MLCoupling<float, float>* coupling = MLCoupling<float, float>::create_from_config(config_path, std::move(input_data), output_data
	/*,ConfigOverrides{
		{"provider.device", std::string("GPU")},
		{"provider.num_gpus", 1}
	}*/);

	if (coupling == nullptr) {
		std::cerr << "Failed to create MLCoupling from config.\n";
		if (use_mpi) MPI_Finalize();
		return 2;
	}



	// *************************************
	// Perform a single model call
	// *************************************

	try {
		coupling->ml_step();
	} catch (const std::exception& e) {
		std::cerr << "Inference failed: " << e.what() << "\n";
		delete coupling;
		if (use_mpi) MPI_Finalize();
		return 3;
	}


	// *************************************
	// Print output
	// *************************************

	std::cout << "Inference output: [";
		std::cout << output_buffer[0];
	std::cout << "]\n";

	delete coupling;

	delete[] flat_data;
	delete[] output_buffer;
	
	if (use_mpi) MPI_Finalize();
	return 0;
}
