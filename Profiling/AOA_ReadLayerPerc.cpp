#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <algorithm> // Include the <algorithm> header for std::replace()

struct Key {
    std::string graph;
    int layer;
    std::string component;
};

bool operator==(const Key& lhs, const Key& rhs) {
    return lhs.graph == rhs.graph && lhs.layer == rhs.layer && lhs.component == rhs.component;
}

// Hash function for Key to be used in unordered_map
struct KeyHash {
    std::size_t operator()(const Key& key) const {
        std::size_t h1 = std::hash<std::string>{}(key.graph);
        std::size_t h2 = std::hash<int>{}(key.layer);
        std::size_t h3 = std::hash<std::string>{}(key.component);
        return h1 ^ h2 ^ h3;
    }
};

std::unordered_map<Key, double, KeyHash> data;

int Load_Layers_Percetage(){
    std::ifstream file("Layers_Percentage.csv"); // Replace "data.csv" with your actual CSV file name
    if (!file.is_open()) {
        std::cout << "Failed to open file" << std::endl;
        return 1;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        std::string graph;
        int layer;
        double timePercentageB;
        double timePercentageG;
        double timePercentageL;
        double timePercentageAverage;

        if (iss >> graph >> layer >> timePercentageB >> timePercentageG >> timePercentageL >> timePercentageAverage) {
            // Create a key based on graph, layer, and component
            Key key{graph, layer, "B"};
            data[key] = timePercentageB;
            key.component = "G";
            data[key] = timePercentageG;
            key.component = "L";
            data[key] = timePercentageL;
            key.component = "Average";
            data[key] = timePercentageAverage;
        } else {
            std::cout << "Failed to parse line: " << line << std::endl;
        }
    }
    return 0;
}

int main() {
 
    Load_Layers_Percetage();

    // Accessing data for desired key
    std::string graphToFind = "alex";
    int layerToFind = 5;
    std::string componentToFind = "B";

    Key keyToFind{graphToFind, layerToFind, componentToFind};
    auto it = data.find(keyToFind);
    if (it != data.end()) {
        double timePercentage = it->second;
        std::cout << "Time Percentage for " << graphToFind << " in component " << componentToFind
                  << " for layer " << layerToFind << ": " << timePercentage << std::endl;
    } else {
        std::cout << "Data not found for the given key." << std::endl;
    }

    std::cout<<"Another example (google,4,G): "<< data.find({"google",4,"G"})->second<<std::endl;

    return 0;
}
