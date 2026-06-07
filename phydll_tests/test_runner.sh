#!/bin/bash
set -e

# Build the test
mkdir -p build
cd build
cmake ..
make
cd ..

DL_CLIENT="../CPP-ML-Interface/dl_clients/build/phydll_dl_client"

echo "=========================================="
echo "TEST 1: Perfect Split"
echo "=========================================="
mpiexec -n 2 ./build/test_phydll_physics perfect : -n 2 ${DL_CLIENT} --model_path minimal_model.pt

echo ""
echo "=========================================="
echo "TEST 2: Imperfect Split (Truncation Bug)"
echo "=========================================="
# We expect this to drop data or hang
mpiexec -n 2 ./build/test_phydll_physics imperfect : -n 2 ${DL_CLIENT} --model_path minimal_model.pt || true

echo ""
echo "=========================================="
echo "TEST 3: Shape Mismatch (Feature Size != 18 for Model 2)"
echo "=========================================="
# Model expects [N, 18], but we send total 20 elements (10 per rank).
# dl_client will reshape to [2, 10] -> This should crash the model forward pass!
mpiexec -n 2 ./build/test_phydll_physics shape_mismatch : -n 2 ${DL_CLIENT} --model_path shape_mismatch_model.pt || true

echo "All tests finished."
