#include "/hpcwork/ro092286/smartsim/SmartRedis/include/client.h"
#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include <hdf5.h>
// lookup h5
#include <filesystem>
#include <chrono>
#include <iomanip>

std::vector<long long> measure_times(SmartRedis::Client& client, size_t rows) {

    std::vector<long long> times;

    auto large_input_memory_alloc_start = std::chrono::high_resolution_clock::now();

    // Create a large input tensor with the specified number of rows and 10 columns
    float* large_data_flat = new float[rows * 10];
    float** large_data = new float*[rows];
    for (size_t i = 0; i < rows; ++i) {
        large_data[i] = &large_data_flat[i * 10];
        /*for (size_t j = 0; j < 10; ++j) {
            large_data[i][j] = static_cast<float>(i * 10 + j / (i * j + 1.0f)); // Fill with some values based on indices
        }*/
    }

    auto large_input_memory_alloc_end = std::chrono::high_resolution_clock::now();

    client.put_tensor("test_large_input_" + std::to_string(rows), large_data, {rows, 10}, SRTensorType::SRTensorTypeFloat, SRMemoryLayout::SRMemLayoutNested);

    auto large_input_put_end = std::chrono::high_resolution_clock::now();

    client.run_model("test_model", {"test_large_input_" + std::to_string(rows)}, {"test_large_output_" + std::to_string(rows)});

    auto large_input_run_end = std::chrono::high_resolution_clock::now();

    client.run_model("test_model", {"test_large_input_" + std::to_string(rows)}, {"test_large_output_" + std::to_string(rows)});

    auto large_input_run_2_end = std::chrono::high_resolution_clock::now();

    // Unpack output tensor with explicit memory management
    std::vector<size_t> large_output_dims = {rows, 1};
    SRTensorType large_output_type = SRTensorType::SRTensorTypeFloat;
    SRMemoryLayout large_output_mem_layout = SRMemoryLayout::SRMemLayoutNested;
    
    float** large_output = new float*[rows];
    for (size_t i = 0; i < rows; ++i) {
        large_output[i] = new float[1];
    }
    
    client.unpack_tensor("test_large_output_" + std::to_string(rows), large_output, large_output_dims, large_output_type, large_output_mem_layout);

    auto large_input_get_output_end = std::chrono::high_resolution_clock::now();

    // Free allocated memory
    delete[] large_data_flat;
    delete[] large_data;
    for (size_t i = 0; i < rows; ++i) {
        delete[] large_output[i];
    }
    delete[] large_output;

    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(large_input_memory_alloc_end - large_input_memory_alloc_start).count());
    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(large_input_put_end - large_input_memory_alloc_end).count());
    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(large_input_run_end - large_input_put_end).count());
    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(large_input_run_2_end - large_input_run_end).count());
    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(large_input_get_output_end - large_input_run_2_end).count());
    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(large_input_get_output_end - large_input_memory_alloc_start).count());

    return times;
}

int main(int argc, char* argv[]) {
    // Set default SSDB environment variable if not already set
    if (getenv("SSDB") == nullptr) {
        setenv("SSDB", "127.0.0.1:6379", 0);
        std::cout << "SSDB not set, using default: 127.0.0.1:6379" << std::endl;
    }

    try {
        // Create a client to connect to the database
        SmartRedis::Client client("dummy_solver_client");
        std::cout << "Successfully connected to SmartRedis database" << std::endl;

        // Load hdf5 data file from
        std::cout << "Loading HDF5 data from input/data.hdf5..." << std::endl;
        std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
        hid_t file_id = H5Fopen("/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/input/data.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
        hid_t dataset_id = H5Dopen(file_id, "data", H5P_DEFAULT);
        hid_t label_dataset_id = H5Dopen(file_id, "label", H5P_DEFAULT);
        
        // Allocate contiguous memory for HDF5 reading
        float* data_flat = new float[100];
        float** data = new float*[10];
        for (int i = 0; i < 10; ++i) {
            data[i] = &data_flat[i * 10];
        }

        float* data_label_flat = new float[10];
        float** data_label = new float*[1];
        data_label[0] = data_label_flat;


        H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_flat);
        H5Dclose(dataset_id);

        H5Dread(label_dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_label_flat);
        H5Dclose(label_dataset_id);
        H5Fclose(file_id);
        std::cout << "Loaded HDF5 data successfully." << std::endl;

        std::cout << "Data values: \n  ";
        for (size_t i = 0; i < 100; ++i) {
            std::cout << std::fixed << std::setprecision(2) << data[i/10][i%10] << " ";
            if ((i + 1) % 10 == 0) {
                std::cout << std::endl << "  ";
            }
        }
        std::cout << std::endl;

        std::cout << "Label values: \n  ";
        for (size_t i = 0; i < 10; ++i) {
            std::cout << std::fixed << std::setprecision(2) << data_label[0][i] << " ";
        }
        std::cout << std::endl;
        
        // void set_model_from_file(const std::string &name, const std::string &model_file, const std::string &backend, const std::string &device, int batch_size = 0, int min_batch_size = 0, int min_batch_timeout = 0, const std::string &tag = "", const std::vector<std::string> &inputs = std::vector<std::string>(), const std::vector<std::string> &outputs = std::vector<std::string>())

        std::vector<std::string> inputs = {};
        std::vector<std::string> outputs = {};

        std::cout << "Setting model from file..." << std::endl;
        auto set_model_start = std::chrono::high_resolution_clock::now();
        client.set_model_from_file("test_model", "/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/input/model_complex.pt", "TORCH", "CPU", 0, 0, 0, "", inputs, outputs);
        auto set_model_end = std::chrono::high_resolution_clock::now();
        auto set_model_duration = std::chrono::duration_cast<std::chrono::milliseconds>(set_model_end - set_model_start);
        std::cout << "Model set successfully. [" << set_model_duration.count() << " ms]" << std::endl;
        std::cout << "Setting the model a second time to see how much dependencies slowed it down the first time..." << std::endl;
        set_model_start = std::chrono::high_resolution_clock::now();
        client.set_model_from_file("test_model", "/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/input/model_complex.pt", "TORCH", "CPU", 0, 0, 0, "", inputs, outputs);
        set_model_end = std::chrono::high_resolution_clock::now();
        set_model_duration = std::chrono::duration_cast<std::chrono::milliseconds>(set_model_end - set_model_start);
        std::cout << "Model set the second time. [" << set_model_duration.count() << " ms]" << std::endl;
        std::cout << "Putting tensor data into database..." << std::endl;
        auto put_tensor_start = std::chrono::high_resolution_clock::now();
        client.put_tensor("test_data_input", data, {10, 10}, SRTensorType::SRTensorTypeFloat, SRMemoryLayout::SRMemLayoutNested);
        auto put_tensor_end = std::chrono::high_resolution_clock::now();
        auto put_tensor_duration = std::chrono::duration_cast<std::chrono::milliseconds>(put_tensor_end - put_tensor_start);
        std::cout << "Tensor data put successfully. [" << put_tensor_duration.count() << " ms]" << std::endl;
        delete[] data;
        delete[] data_flat;
        delete[] data_label;
        std::cout << "Freed input data memory." << std::endl;

        std::cout << "Read the input tensor back from the database to verify it was stored correctly..." << std::endl;
        void* input = nullptr; // get_tensor allocates memory for us
        std::vector<size_t> input_dims;
        SRTensorType input_type;
        SRMemoryLayout input_mem_layout = SRMemoryLayout::SRMemLayoutNested;
        auto get_tensor_start = std::chrono::high_resolution_clock::now();
        client.get_tensor("test_data_input", input, input_dims, input_type, input_mem_layout);
        auto get_tensor_end = std::chrono::high_resolution_clock::now();
        auto get_tensor_duration = std::chrono::duration_cast<std::chrono::milliseconds>(get_tensor_end - get_tensor_start);
        std::cout << "  [get_tensor: " << get_tensor_duration.count() << " ms]" << std::endl;

        std::cout << "Input tensor dimensions: " << std::endl << "  ";
        for (size_t dim : input_dims) {
            std::cout << dim << " ";
        }        std::cout << std::endl;
        std::cout << "Input tensor type: " << static_cast<int>(input_type) << std::endl;
        std::cout << "Input tensor memory layout: " << static_cast<int>(input_mem_layout) << std::endl;
        std::cout << "Input tensor values: ";
        float** input_data = static_cast<float**>(input);
        for (size_t i = 0; i < input_dims[0] * input_dims[1]; ++i) {
            std::cout << std::fixed << std::setprecision(2) << input_data[i/input_dims[1]][i%input_dims[1]] << " ";
            if ((i + 1) % 10 == 0) {
                std::cout << std::endl << "  ";
            }
        }
        std::cout << std::endl;

        // Use unpack instead of get_tensor to verify that it works as well
        std::cout << "Unpacking the input tensor from the database to verify it was stored correctly..." << std::endl;
        float* unpacked_input_flat = new float[100];
        float** unpacked_input = new float*[10];
        for (size_t i = 0; i < 10; ++i) {
            unpacked_input[i] = new float[10];
        }

        input_dims = {10, 10};
        input_type = SRTensorType::SRTensorTypeFloat;
        input_mem_layout = SRMemoryLayout::SRMemLayoutNested;

        auto unpack_tensor_start = std::chrono::high_resolution_clock::now();
        client.unpack_tensor("test_data_input", unpacked_input, input_dims, input_type, input_mem_layout);
        auto unpack_tensor_end = std::chrono::high_resolution_clock::now();
        auto unpack_tensor_duration = std::chrono::duration_cast<std::chrono::milliseconds>(unpack_tensor_end - unpack_tensor_start);
        std::cout << "  [unpack_tensor: " << unpack_tensor_duration.count() << " ms]" << std::endl;

        std::cout << "Unpacked input tensor values: " << std::endl << "  ";
        for (size_t i = 0; i < input_dims[0] * input_dims[1]; ++i) {
            std::cout << std::fixed << std::setprecision(2) << unpacked_input[i/input_dims[1]][i%input_dims[1]] << " ";
            if ((i + 1) % 10 == 0) {
                std::cout << std::endl << "  ";
            }
        }
        std::cout << std::endl;

        delete[] unpacked_input_flat;
        for (size_t i = 0; i < 10; ++i) {
            delete[] unpacked_input[i];
        }
        delete[] unpacked_input;
        std::cout << "Freed unpacked input data memory." << std::endl;

        std::cout << "Running model..." << std::endl;
        auto run_model_start = std::chrono::high_resolution_clock::now();
        client.run_model("test_model", {"test_data_input"}, {"test_data_output"});
        auto run_model_end = std::chrono::high_resolution_clock::now();
        auto run_model_duration = std::chrono::duration_cast<std::chrono::milliseconds>(run_model_end - run_model_start);
        std::cout << "Model executed. [" << run_model_duration.count() << " ms]" << std::endl;

        std::cout << "Running model a second time..." << std::endl;
        run_model_start = std::chrono::high_resolution_clock::now();
        client.run_model("test_model", {"test_data_input"}, {"test_data_output"});
        run_model_end = std::chrono::high_resolution_clock::now();
        run_model_duration = std::chrono::duration_cast<std::chrono::milliseconds>(run_model_end - run_model_start);
        std::cout << "Model executed the second time. [" << run_model_duration.count() << " ms]" << std::endl;
        
        std::cout << "Getting output tensor from database..." << std::endl;

        void* output = nullptr; // get_tensor allocates memory for us
        std::vector<size_t> dims;
        SRTensorType type;
        SRMemoryLayout mem_layout = SRMemoryLayout::SRMemLayoutNested;
        auto get_output_start = std::chrono::high_resolution_clock::now();
        client.get_tensor("test_data_output", output, dims, type, mem_layout);
        auto get_output_end = std::chrono::high_resolution_clock::now();
        auto get_output_duration = std::chrono::duration_cast<std::chrono::milliseconds>(get_output_end - get_output_start);
        std::cout << "Output tensor retrieved. [" << get_output_duration.count() << " ms]" << std::endl;

        std::cout << "Output tensor dimensions: ";
        for (size_t dim : dims) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        std::cout << "Output tensor type: " << static_cast<int>(type) << std::endl;
        std::cout << "Output tensor memory layout: " << static_cast<int>(mem_layout) << std::endl;
        std::cout << "Output tensor values: " << std::endl;
        float** output_data = static_cast<float**>(output);
        for (size_t i = 0; i < dims[0]; ++i) {
            std::cout << std::fixed << std::setprecision(2) << output_data[i][0] << " ";
        }
        std::cout << std::endl;

        // Memory freed by Client on destruction
        
        


        // Let's generate a larger random input tensor and see how long it takes to put it in the database and run the model on it
        std::cout << "Generating larger random input tensor..." << std::endl;

        std::vector<size_t> rows = {1,10,100,1000,10000,100000,1000000,10000000};

        // write to csv file
        std::ofstream csv_file("performance_results.csv");
        csv_file << "rows,alloc_time_ns,put_time_ns,run_time_ns,run_time_2_ns,get_output_time_ns,total_time_ns\n";

        for (size_t r : rows) {
            std::cout << "Testing with " << r << " rows...1000 times...";
            long long total_time = 0;
            for (int i = 0; i < 1000; ++i) {
                std::vector<long long> times = measure_times(client, r);
                if (times.size() >= 6) {
                    csv_file << r << "," << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << times[4] << "," << times[5] << "\n";
                    total_time += times[5];
                    csv_file.flush();
                }
            }
            std::cout << "Total time: " << std::fixed << std::setprecision(2) << (total_time / 1000000.0) << " ms." << std::endl;
        }

        csv_file.close();


        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
