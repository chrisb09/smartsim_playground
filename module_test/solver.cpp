#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../CPP-ML-Interface/include/ml_coupling.hpp"

int main(int argc, char** argv)
{
	const std::string config_path = (argc > 1) ? argv[1] : "config.toml";

	const char* ssdb = std::getenv("SSDB");
	if (ssdb == nullptr || std::string(ssdb).empty()) {
		std::cerr << "SSDB is not set. Aborting.\n";
		return 1;
	}

	std::cout << "Using SSDB=" << ssdb << "\n";
	std::cout << "Loading config from " << config_path << "\n";


	// ******************************
	// Create dummy data
	// ******************************

    //2 inputs a Bx1x3x3
    float***** data = new float****[2];
    data[0] = new float***[1];
    data[0][0] = new float**[1];
    data[0][0][0] = new float*[3];
    for (int i = 0; i < 3; ++i) {
        data[0][0][0][i] = new float[3];
        for (int j = 0; j < 3; ++j) {
            data[0][0][0][i][j] = (4 + i * 17 + j *4 ) % 100;
        }
    }
    data[1] = new float***[1];
    data[1][0] = new float**[1];
    data[1][0][0] = new float*[3];
    for (int i = 0; i < 3; ++i) {
        data[1][0][0][i] = new float[3];
        for (int j = 0; j < 3; ++j) {
            data[1][0][0][i][j] = (7 + i * 24 + j * 7 ) % 200;
        }
    }

	MLCouplingTensor<float> input_water_tensor = MLCouplingTensor<float>::wrap_nested(
		static_cast<void*>(data[0]),
		std::vector<int>{1, 1, 3, 3},
		MLCouplingMemLayoutNested,
		MLCouplingOwnershipExternal);

	MLCouplingTensor<float> input_terrain_tensor = MLCouplingTensor<float>::wrap_nested(
		static_cast<void*>(data[1]),
		std::vector<int>{1, 1, 3, 3},
		MLCouplingMemLayoutNested,
		MLCouplingOwnershipExternal);

	MLCouplingData<float> input_data{std::vector<MLCouplingTensor<float>>{
		input_water_tensor,
		input_terrain_tensor
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

	MLCoupling<float, float>* coupling = MLCoupling<float, float>::create_from_config(config_path, std::move(input_data), output_data, ConfigOverrides{
		{"provider.device", std::string("GPU")},
		{"provider.num_gpus", 1}
	});

	if (coupling == nullptr) {
		std::cerr << "Failed to create MLCoupling from config.\n";
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
		return 3;
	}


	// *************************************
	// Print output
	// *************************************

	std::cout << "Inference output: [";
		std::cout << output_buffer[0];
	std::cout << "]\n";

	delete coupling;

	// Clean up the data buffers
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 3; ++j) {
			delete[] data[i][0][0][j];
		}
		delete[] data[i][0][0];
		delete[] data[i][0];
		delete[] data[i];
	}
	delete[] data;
	delete[] output_buffer;
	return 0;
}
