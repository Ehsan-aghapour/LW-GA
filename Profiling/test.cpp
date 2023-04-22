#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <fstream>
#include <vector>

using namespace Eigen;

int main() {
    // set the frequencies to test for each processing element
    std::vector<VectorXd> frequencies = {{1.0, 1.2, 1.4}, {1.1, 1.3}, {1.0, 1.1, 1.2, 1.3}, {1.2, 1.3}};

    // set the number of processing elements
    int numPE = frequencies.size();

    // create a vector of matrices to store the performance data
    std::vector<MatrixXd> perfData(numPE);

    // measure performance for each frequency and processing element
    for (int i = 0; i < numPE; i++) {
        // create a matrix to store the performance data for the ith processing element
        int numFreqs = frequencies[i].size();
        MatrixXd peData(numFreqs, 3);

        for (int j = 0; j < numFreqs; j++) {
            // set the frequency for the ith processing element
            std::string command = "sudo dvfs " + std::to_string(i) + " " + std::to_string(frequencies[i](j)) + "GHz";

            auto start = std::chrono::high_resolution_clock::now();
            // execute the command
            int ret = system(command.c_str());
            auto end = std::chrono::high_resolution_clock::now();

            if (ret == -1) {
                std::cerr << "Error executing dvfs command" << std::endl;
                return 1;
            }

            // calculate the duration of the execution in microseconds
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            // store the duration in the peData matrix
            peData(j, 0) = i;
            peData(j, 1) = frequencies[i](j);
            peData(j, 2) = duration.count();

            // delay for some time to ensure that the system has stabilized
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        // add the peData matrix to the perfData vector
        perfData[i] = peData;
    }

    // write the performance data to a CSV file
    std::ofstream outputFile("perfData.csv");
    if (outputFile.is_open()) {
        outputFile << "PE,Frequency,Performance\n";
        for (int i = 0; i < numPE; i++) {
            for (int j = 0; j < perfData[i].rows(); j++) {
                outputFile << perfData[i](j, 0) << "," << perfData[i](j, 1) << "," << perfData[i](j, 2) << "\n";
            }
        }
        outputFile.close();
        std::cout << "Performance data written to perfData.csv" << std::endl;
    } else {
        std::cerr << "Error opening output file" << std::endl;
    }

    return 0;
}

