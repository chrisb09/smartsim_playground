#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <cmath>
#include <algorithm>
#include <mpi.h>

#include "ml_coupling.hpp"
#include "data/ml_coupling_data.hpp"

struct BenchmarkConfig {
    std::string config_file;
    std::string provider = "smartsim"; // smartsim, aix, phydll
    std::string model_path = "train_models/model_a/watercnn_cuda.pt";
    std::string aix_comm_mode = "collective"; // collective, pipelined
    int samples_per_rank = 100000;
    int steps = 12;
    int warmup_steps = 2;
    int batch_size = 100000;
    int min_batch_size = 0;
    int min_batch_timeout = 0;
    std::string output_csv = "";
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --config <path>              Path to TOML config file\n"
              << "  --provider <name>            smartsim | aix | phydll (default: smartsim)\n"
              << "  --model-path <path>          Path to PyTorch model artifact\n"
              << "  --samples-per-rank <int>     Number of [18] float samples per rank (default: 100000)\n"
              << "  --steps <int>                Total steps (default: 12)\n"
              << "  --warmup-steps <int>         Warmup steps excluded from steady metrics (default: 2)\n"
              << "  --batch-size <int>           Provider forward/store batch size (default: 100000)\n"
              << "  --min-batch-size <int>       SmartSim min_batch_size (default: 0)\n"
              << "  --min-batch-timeout <int>    SmartSim min_batch_timeout in ms (default: 0)\n"
              << "  --aix-comm-mode <mode>       AIx communication mode: collective | pipelined\n"
              << "  --output-csv <path>          CSV file path to append benchmark stats\n";
}

static BenchmarkConfig parse_args(int argc, char** argv) {
    BenchmarkConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--config" && i + 1 < argc) {
            cfg.config_file = argv[++i];
        } else if (arg == "--provider" && i + 1 < argc) {
            cfg.provider = argv[++i];
        } else if (arg == "--model-path" && i + 1 < argc) {
            cfg.model_path = argv[++i];
        } else if (arg == "--samples-per-rank" && i + 1 < argc) {
            cfg.samples_per_rank = std::stoi(argv[++i]);
        } else if (arg == "--steps" && i + 1 < argc) {
            cfg.steps = std::stoi(argv[++i]);
        } else if (arg == "--warmup-steps" && i + 1 < argc) {
            cfg.warmup_steps = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            cfg.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--min-batch-size" && i + 1 < argc) {
            cfg.min_batch_size = std::stoi(argv[++i]);
        } else if (arg == "--min-batch-timeout" && i + 1 < argc) {
            cfg.min_batch_timeout = std::stoi(argv[++i]);
        } else if (arg == "--aix-comm-mode" && i + 1 < argc) {
            cfg.aix_comm_mode = argv[++i];
        } else if (arg == "--output-csv" && i + 1 < argc) {
            cfg.output_csv = argv[++i];
        }
    }
    return cfg;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    BenchmarkConfig cfg = parse_args(argc, argv);

    // PhyDLL MPMD handling: if this process is a PhyDLL DL client, split communicator and exit
    const char* phydll_dl_env = std::getenv("PHYDLL_DL_CLIENT");
    const bool is_phydll_dl = (phydll_dl_env != nullptr && std::string(phydll_dl_env) == "1");

    MPI_Comm solver_comm = MPI_COMM_WORLD;
    if (is_phydll_dl) {
        MPI_Comm dummy_comm;
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, world_rank, &dummy_comm);
        MPI_Finalize();
        return 0;
    }

    if (cfg.provider == "phydll") {
        MPI_Comm app_comm;
        MPI_Comm_split(MPI_COMM_WORLD, 0, world_rank, &app_comm);
        solver_comm = app_comm;
    }

    int solver_rank = 0;
    int solver_size = 1;
    MPI_Comm_rank(solver_comm, &solver_rank);
    MPI_Comm_size(solver_comm, &solver_size);

    if (solver_rank == 0) {
        std::cout << "=================================================================\n"
                  << "=== CMI Batch & Memory Calibration Benchmark Harness          ===\n"
                  << "=================================================================\n"
                  << "Provider:           " << cfg.provider << "\n"
                  << "Config:             " << cfg.config_file << "\n"
                  << "Model Path:         " << cfg.model_path << "\n"
                  << "Solver Ranks:       " << solver_size << " (World size: " << world_size << ")\n"
                  << "Samples per Rank:   " << cfg.samples_per_rank << "\n"
                  << "Total Work/Step:    " << static_cast<long long>(solver_size) * cfg.samples_per_rank << " samples\n"
                  << "Target Batch Size:  " << cfg.batch_size << "\n"
                  << "SmartSim Min Batch: " << cfg.min_batch_size << " (timeout=" << cfg.min_batch_timeout << " ms)\n"
                  << "Total Steps:        " << cfg.steps << " (Warmup: " << cfg.warmup_steps << ")\n"
                  << "=================================================================\n" << std::endl;
    }

    // Allocate flat contiguous data buffers for [samples_per_rank, 18] -> [samples_per_rank]
    const std::size_t num_samples = static_cast<std::size_t>(cfg.samples_per_rank);
    std::vector<float> input_buffer(num_samples * 18);
    std::vector<float> output_buffer(num_samples, 0.0f);

    for (std::size_t i = 0; i < num_samples; ++i) {
        for (std::size_t f = 0; f < 18; ++f) {
            input_buffer[i * 18 + f] = static_cast<float>((i + solver_rank * 18 + f) % 100) * 0.01f;
        }
    }

    MLCouplingData<float> input_data;
    input_data.add_tensor(MLCouplingTensor<float>::wrap_flat(
        input_buffer.data(),
        std::vector<int>{static_cast<int>(num_samples), 18},
        MLCouplingMemLayoutContiguous,
        MLCouplingOwnershipExternal));

    MLCouplingData<float> output_data;
    output_data.add_tensor(MLCouplingTensor<float>::wrap_flat(
        output_buffer.data(),
        std::vector<int>{static_cast<int>(num_samples)},
        MLCouplingMemLayoutContiguous,
        MLCouplingOwnershipExternal));

    // Configure overrides for CMI
    ConfigDottedOverrides overrides;
    if (cfg.provider == "aix") {
        overrides.emplace("library.model_file", cfg.model_path);
        overrides.emplace("library.batchsize", static_cast<int64_t>(cfg.batch_size));
        overrides.emplace("library.app_comm", static_cast<void*>(MPI_COMM_WORLD));
        overrides.emplace("library.communication_mode", cfg.aix_comm_mode);
    } else if (cfg.provider == "phydll") {
        overrides.emplace("library.model_file", cfg.model_path);
        overrides.emplace("library.backend", std::string("TORCH"));
        overrides.emplace("library.device", std::string("GPU"));
        overrides.emplace("library.batch_size", static_cast<int64_t>(cfg.batch_size));
    } else { // smartsim
        overrides.emplace("library.device", std::string("GPU"));
        overrides.emplace("library.model_backend", std::string("TORCH"));
        overrides.emplace("library.model_path", cfg.model_path);
        overrides.emplace("library.model_name", std::string("watercnn_batch_bench"));
        overrides.emplace("library.num_gpus", static_cast<int64_t>(1));
        overrides.emplace("library.nodes", static_cast<int64_t>(1));
        overrides.emplace("library.batch_size", static_cast<int64_t>(cfg.batch_size));
        overrides.emplace("library.min_batch_size", static_cast<int64_t>(cfg.min_batch_size));
        overrides.emplace("library.min_batch_timeout", static_cast<int64_t>(cfg.min_batch_timeout));
    }

    std::unique_ptr<MLCoupling<float, float>> ml_coupling;
    try {
        ml_coupling.reset(MLCoupling<float, float>::create_from_config(
            cfg.config_file,
            std::move(input_data),
            std::move(output_data),
            ConfigOverrides(std::move(overrides))));
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << solver_rank << "] Error initializing CMI: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (solver_rank == 0) {
        std::cout << "CMI initialized successfully. Starting benchmark steps...\n" << std::endl;
    }

    std::vector<double> step_durations_ms;
    const long long total_samples_step = static_cast<long long>(solver_size) * cfg.samples_per_rank;

    for (int step = 1; step <= cfg.steps; ++step) {
        // Enforce strict lock-step synchronization before triggering inference
        MPI_Barrier(solver_comm);

        const auto t_start = std::chrono::high_resolution_clock::now();

        try {
            ml_coupling->step();
        } catch (const std::exception& e) {
            std::cerr << "[Rank " << solver_rank << "] Step " << step << " failed: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        const auto t_end = std::chrono::high_resolution_clock::now();
        const double local_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        double max_ms = 0.0;
        double min_ms = 0.0;
        double sum_ms = 0.0;

        MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, solver_comm);
        MPI_Reduce(&local_ms, &min_ms, 1, MPI_DOUBLE, MPI_MIN, 0, solver_comm);
        MPI_Reduce(&local_ms, &sum_ms, 1, MPI_DOUBLE, MPI_SUM, 0, solver_comm);

        if (solver_rank == 0) {
            const double avg_ms = sum_ms / solver_size;
            const double throughput = (max_ms > 0.0) ? (static_cast<double>(total_samples_step) / (max_ms / 1000.0)) : 0.0;
            std::cout << "STEP_TIMING step=" << step
                      << " latency_ms=" << std::fixed << std::setprecision(2) << max_ms
                      << " min_ms=" << std::setprecision(2) << min_ms
                      << " avg_ms=" << std::setprecision(2) << avg_ms
                      << " total_samples=" << total_samples_step
                      << " throughput_samples_sec=" << std::setprecision(0) << throughput
                      << std::endl;

            if (step > cfg.warmup_steps) {
                step_durations_ms.push_back(max_ms);
            }
        }
    }

    if (solver_rank == 0 && !step_durations_ms.empty()) {
        std::vector<double> sorted_ms = step_durations_ms;
        std::sort(sorted_ms.begin(), sorted_ms.end());
        const double median_ms = sorted_ms[sorted_ms.size() / 2];
        double sum = 0.0;
        for (double v : step_durations_ms) sum += v;
        const double mean_ms = sum / step_durations_ms.size();
        const double median_throughput = (median_ms > 0.0) ? (static_cast<double>(total_samples_step) / (median_ms / 1000.0)) : 0.0;

        std::cout << "\n=================================================================\n"
                  << "=== Benchmark Summary (Steady-State Steps " << (cfg.warmup_steps + 1) << " to " << cfg.steps << ") ===\n"
                  << "=================================================================\n"
                  << "Steady Step Count:   " << step_durations_ms.size() << "\n"
                  << "Median Latency:      " << std::fixed << std::setprecision(2) << median_ms << " ms\n"
                  << "Mean Latency:        " << std::setprecision(2) << mean_ms << " ms\n"
                  << "Min Latency:         " << std::setprecision(2) << sorted_ms.front() << " ms\n"
                  << "Max Latency:         " << std::setprecision(2) << sorted_ms.back() << " ms\n"
                  << "Median Throughput:   " << std::setprecision(0) << median_throughput << " samples/sec\n"
                  << "=================================================================\n";

        if (!cfg.output_csv.empty()) {
            std::ofstream csv(cfg.output_csv, std::ios::app);
            if (csv.is_open()) {
                csv << cfg.provider << ","
                    << cfg.batch_size << ","
                    << cfg.min_batch_size << ","
                    << cfg.samples_per_rank << ","
                    << solver_size << ","
                    << total_samples_step << ","
                    << median_ms << ","
                    << mean_ms << ","
                    << sorted_ms.front() << ","
                    << sorted_ms.back() << ","
                    << median_throughput << "\n";
            }
        }
    }

    ml_coupling.reset();

    const char* barrier_env = std::getenv("PHYDLL_MPMD_SHUTDOWN_BARRIER");
    if (barrier_env != nullptr && std::string(barrier_env) == "1") {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (solver_comm != MPI_COMM_WORLD && solver_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&solver_comm);
    }

    MPI_Finalize();
    return 0;
}
