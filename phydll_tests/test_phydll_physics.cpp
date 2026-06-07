#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>

extern "C" {
#include "phydll.h"
}

// We use the raw phydll C-API to precisely control the sizes sent over the handshake,
// which mimics the CPP-ML-Interface Phydll provider behavior without the overhead of
// configuring the entire interface. The bug is in the DL client side anyway.

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    std::string mode = "perfect";
    if (argc > 1) {
        mode = argv[1];
    }

    int hc_rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &hc_rank);

    phydll_init("physical");
    MPI_Comm comm = phydll_get_local_mpi_comm();
    int size, rank; 
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    phydll_opt_enable_cpl_loop();

    int field_size = 10; // default 10 elements per rank
    if (mode == "imperfect") {
        // e.g. rank 0 gets 11, rank 1 gets 10. Total 21. Not divisible by 2.
        if (rank == 0) {
            field_size = 11;
        } else {
            field_size = 10;
        }
    } else if (mode == "shape_mismatch") {
        field_size = 18; // this should trigger the mismatch since dl_client reshapes it based on batch size! Wait, if total is 36, batch size is 2, features is 18. This actually fits! To mismatch we need total to be divisible but not 18 features, e.g. 10 features!
        if (rank == 0) field_size = 10;
        else field_size = 10;
    }

    phydll_define_phy(1, field_size); // 1 field, field_size elements

    std::vector<double> phy_field(field_size, rank + 1.0);
    std::vector<double> dl_field(field_size, 0.0);

    // Sync handshake
    if (rank == 0) std::cout << "Starting loop for mode: " << mode << std::endl;

    for (int iter = 1; iter <= 2; iter++) {
        char label[] = "FIELD0";
        double* p_phy = phy_field.data();
        phydll_set_field(&p_phy, label);

        phydll_isend();
        phydll_wait_isend();
        phydll_recv();

        double* p_dl = dl_field.data();
        phydll_get_field(&p_dl, label);

        int rcv_size = 0;
        phydll_get_field_size(&rcv_size);
        
        if (rank == 0) {
            std::cout << "Iter " << iter << " Rank 0 Received " << rcv_size << " elements." << std::endl;
        }
    }

    phydll_finalize();
    MPI_Finalize();
    return 0;
}
