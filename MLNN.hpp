#pragma once

#include "NNFunc.hpp"
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using std::copy;
using std::cout;
using std::endl;
using std::ifstream;
using std::ios;
using std::mt19937_64;
using std::ofstream;
using std::random_device;
using std::range_error;
using std::runtime_error;
using std::streamsize;
using std::string;
using std::to_string;
using std::vector;

class MLNN {
public:
    size_t input, layers;
    vector<size_t> nodes;

private:
    vector<vector<double>> activations;
    vector<vector<vector<double>>> weights, change;

public:
    /**
     * Empty initializer, but *please* for the love of god, don't use it
     * like that, nothing gets innitiated, and things will break
     */

    MLNN() {}

    /**
     * Creates the network with the given layer sizes
     *
     * @param inputLayer The amount of neurons for the input layer
     * @param hiddenLayers The size of each hidden layer
     * @param outputLayer The number of outputs
     */

    MLNN(const size_t inputLayer, const vector<size_t> hiddenLayers, const size_t outputLayer)
        : input(inputLayer), layers(hiddenLayers.size() + 2), activations(vector<vector<double>>(layers)),
          nodes(vector<size_t>(layers)),
          weights(vector<vector<vector<double>>>(layers - 1)), change(vector<vector<vector<double>>>(layers - 1)) {
        size_t i = 0;
        activations[i++] = vector<double>(nodes[i] = inputLayer + 1); // + 1 for bias layer

        for (const size_t node : hiddenLayers) {
            activations[i++] = vector<double>(nodes[i] = node);
        }
        activations[i] = vector<double>(nodes[i] = outputLayer);

        // Load weights and change
        for (size_t outputLayer = layers - 2; outputLayer > 0; outputLayer--) {
            const size_t inputLayer = outputLayer - 1;
            weights[inputLayer] = makeRandomMatrix(nodes[inputLayer], nodes[outputLayer], -0.2, 0.2);
            change[inputLayer] = makeMatrix(nodes[inputLayer], nodes[outputLayer]);
        }

        const size_t input = layers - 2, output = layers - 1;
        weights[input] = makeRandomMatrix(nodes[input], nodes[output], -2.0, 2.0);
        change[input] = makeMatrix(nodes[input], nodes[output]);
    }

    /**
     * Loads a network from a given file
     */

    MLNN(const char *path) {
        ifstream in;
        in.open(path, ios::in | ios::binary);
        if (!in.is_open()) {
            throw runtime_error("Cannot open the given file");
        }

        // Get the sizes of variables
        const streamsize
            st_size = sizeof(size_t),
            double_size = sizeof(double);

        // Load header stuff
        in.read((char *)&layers, st_size);
        activations = vector<vector<double>>(layers);
        nodes = vector<size_t>(layers);
        size_t i = 0;
        for (size_t &node : nodes) {
            in.read((char *)&node, st_size);
            activations[i++] = vector<double>(node);
        }
        input = nodes[0] - 1;

        const size_t layersToLoad = layers - 1;
        weights = change = vector<vector<vector<double>>>(layersToLoad);
        // Load weights
        for (size_t i = 0; i < layersToLoad; i++) {
            const size_t output = nodes[i + 1], input = nodes[i];
            weights[i] = change[i] = vector<vector<double>>(input);
            for (size_t j = 0; j < input; j++) {
                weights[i][j] = change[i][j] = vector<double>(output);
                for (size_t k = 0; k < output; k++) {
                    in.read((char *)&weights[i][j][k], double_size);
                    in.read((char *)&change[i][j][k], double_size);
                }
            }
        }

        in.close();
    }

    /**
     * Clones
     */

    MLNN(const MLNN &MLNN)
        : input(MLNN.input), layers(MLNN.layers), nodes(MLNN.nodes),
          activations(MLNN.activations),
          weights(MLNN.weights), change(MLNN.change) {}

    vector<double> predict(const vector<double> &inputs) {
        if (inputs.capacity() != input) {
            throw range_error("Incorrect number of inputs");
        }

        update(inputs);
        return activations[layers - 1];
    }

protected:
    void update(const vector<double> &inputs) {
        activations[0] = inputs;

        // Activations for input + hidden
        for (size_t inputLayer = 0; inputLayer < layers - 1; inputLayer++) {
            const size_t outputLayer = inputLayer + 1;
            for (size_t j = 0; j < nodes[outputLayer]; j++) {
                double sum = 0.0;
                for (size_t i = 0; i < nodes[inputLayer]; i++) {
                    sum += activations[inputLayer][i] * weights[inputLayer][i][j];
                }
                activations[outputLayer][j] = sigmoid(sum);
            }
        }
    }

    /**
     * Computes the gradiant, and adjusts the
     *
     * @param targets The targets of what the AI should be outputting
     * @return The deltas (gradiants)
     */

    vector<vector<double>> backPropagate(const vector<double> &targets) {
        const size_t last = layers - 1;
        size_t i = 0;
        vector<vector<double>> deltas(layers);

        for (const size_t &count : nodes)
            deltas[i++] = vector<double>(count);

        // Calculate error/loss

        // Work from back to front
        for (size_t k = 0; k < nodes[last]; k++) {
            deltas[last][k] = dsigmoid(activations[last][k]) * (targets[k] - activations[last][k]);
        }

        for (size_t outputLayer = last; outputLayer > 0; outputLayer--) {
            const size_t inputLayer = outputLayer - 1;
            for (size_t j = 0; j < nodes[inputLayer]; j++) {
                double error = 0.0;
                for (size_t k = 0; k < nodes[outputLayer]; k++)
                    error += deltas[outputLayer][k] * weights[inputLayer][j][k];
                deltas[inputLayer][j] = dsigmoid(activations[inputLayer][j]) * error;
            }
        }

        return deltas;
    }

    /**
     * Updates the weights with the given deltas
     *
     * @param deltas The deltas
     * @param learningRate The learning rate
     * @param momentumFactor The momentum factor
     */

    void updateWeights(const vector<vector<double>> &deltas, const double learningRate, const double momentumFactor) {
        // Update weights

        for (size_t outputLayer = layers - 2; outputLayer > 0; outputLayer--) {
            const size_t inputLayer = outputLayer - 1;
            // cout << outputLayer << ", " << inputLayer << "\n";
            // continue;
            for (size_t j = 0; j < nodes[inputLayer]; j++) {
                for (size_t k = 0; k < nodes[outputLayer]; k++) {
                    const double change = deltas[outputLayer][k] * activations[inputLayer][j];
                    weights[inputLayer][j][k] += learningRate * change + momentumFactor * this->change[inputLayer][j][k];
                    this->change[inputLayer][j][k] = change;
                }
            }
        }
    }

    double calculateError(const vector<double> &targets) {
        const size_t end = layers - 1;
        // Calculate error
        double error = 0.0;
        for (size_t k = 0; k < nodes[end]; k++)
            error += (targets[k] - activations[end][k]) * (targets[k] - activations[end][k]);
        return error / nodes[end];
    }

public:
    void test(const vector<vector<vector<double>>> &patterns) {
        for (const vector<vector<double>> &pattern : patterns) {
            cout << nicePrint(pattern[0]);
            cout << " -> " << nicePrint(predict(pattern[0])) << endl;
        }
    }

    void test(const vector<vector<double>> &patterns) {
        for (const vector<double> &pattern : patterns) {
            // cout << nicePrint(pattern);
            cout << " -> " << nicePrint(predict(pattern)) << endl;
        }
    }

    double train(const vector<vector<vector<double>>> &patterns, const size_t iterations = 10000, const double N = 0.5, const double M = 0.1) {
        const size_t last = layers - 1;
        for (const vector<vector<double>> &pattern : patterns) {
            if (pattern[0].size() != input) {
                throw range_error("Wrong number of input values");
            }
            if (pattern[1].size() != nodes[last]) {
                throw range_error("Wrong number of target values");
            }
        }
        double error;
        for (size_t i = 0; i < iterations; i++) {
            for (const vector<vector<double>> &pattern : patterns) {
                update(pattern[0]);
                updateWeights(backPropagate(pattern[1]), N, M);
            }
            if (i % 10000 == 0) // Should make adjustable
            {
                error = 0.0;
                for (const vector<vector<double>> &pattern : patterns) {
                    update(pattern[0]);
                    error += calculateError(pattern[1]);
                }
                cout << "MSE: " << error << endl;
            }
        }

        // Get error
        error = 0.0;
        for (const vector<vector<double>> &pattern : patterns) {
            update(pattern[0]);
            error += calculateError(pattern[1]);
        }
        return error;
    }

    double randTrain(const vector<vector<vector<double>>> &patterns, const size_t iterations = 10000, const double N = 0.5, const double M = 0.1) {
        const size_t last = layers - 1;
        for (const vector<vector<double>> &pattern : patterns) {
            if (pattern[0].size() != input) {
                throw range_error("Wrong number of input values");
            }
            if (pattern[1].size() != nodes[last]) {
                throw range_error("Wrong number of target values");
            }
        }
        mt19937_64 g((random_device())());

        double error;
        for (size_t i = 0; i < iterations; i++) {
            vector<vector<vector<double>>> ps = patterns;
            shuffle(ps.begin(), ps.end(), g);

            for (const vector<vector<double>> &pattern : ps) {
                update(pattern[0]);
                updateWeights(backPropagate(pattern[1]), N, M);
            }
            if (i % 10000 == 0) // Should make adjustable
            {
                error = 0.0;
                for (const vector<vector<double>> &pattern : patterns) {
                    update(pattern[0]);
                    error += calculateError(pattern[1]);
                }
                cout << "MSE: " << error << endl;
            }
        }

        // Get error
        error = 0.0;
        for (const vector<vector<double>> &pattern : patterns) {
            update(pattern[0]);
            error += calculateError(pattern[1]);
        }
        return error;
    }

    string toString() {
        string output;

        size_t i = 0;
        for (const vector<vector<double>> &weight : weights) {
            output += "\nLayer " + to_string(i++) + ":\n" + nicePrint(weight);
        }

        return output;
    }

    MLNN &operator=(const MLNN &nn) {
        input = nn.input;
        layers = nn.layers;
        nodes = nn.nodes;

        activations = nn.activations;
        weights = nn.weights;
        change = nn.change;

        return *this;
    }

    bool toFile(const char *path) {
        // Attempt to open the output file
        ofstream out;
        out.open(path, ios::out | ios::trunc | ios::binary);
        if (!out.is_open()) {
            return false; // Failed to open
        }

        // Get the sizes of variables
        const streamsize
            st_size = sizeof(size_t),
            double_size = sizeof(double);

        // Write header stuff
        out.write((char *)&layers, st_size);
        for (const size_t &node : nodes) {
            out.write((char *)&node, st_size);
        }

        const size_t layersToLoad = layers - 1;
        // Write data
        for (size_t i = 0; i < layersToLoad; i++) {
            const size_t output = nodes[i + 1], input = nodes[i];
            for (size_t j = 0; j < input; j++) {
                for (size_t k = 0; k < output; k++) {
                    out.write((char *)&weights[i][j][k], double_size);
                    out.write((char *)&change[i][j][k], double_size);
                }
            }
        }

        out.close();
        return true;
    }
};