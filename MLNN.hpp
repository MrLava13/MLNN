#pragma once

#include "NNFunc.hpp"
#include <algorithm>
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
using std::shuffle;
using std::streamsize;
using std::string;
using std::to_string;
using std::vector;

class MLNN {
public:
    size_t input, layers;
    vector<size_t> nodes;

private:
    twoDimensional activations;
    threeDimensional weights, change;

    // Random number generator
    mt19937_64 g = mt19937_64((random_device())());

    // Variable sizes, for import and export
    static const streamsize
        st_size = sizeof(size_t),
        double_size = sizeof(double);

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
        : input(inputLayer), layers(hiddenLayers.size() + 2), activations(twoDimensional(layers)),
          nodes(vector<size_t>(layers)),
          weights(threeDimensional(layers - 1)), change(threeDimensional(layers - 1)) {
        const size_t end = layers - 2;
        activations[0].reserve(nodes[0] = inputLayer + 1); // + 1 for bias layer
        for(size_t i = 0; i < end; i++) {
            const size_t j = i + 1;
            activations[j].reserve(nodes[j] = hiddenLayers[i]);
        }
        activations[layers - 1].reserve(nodes[layers - 1] = outputLayer);

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
     *
     * @param path The file path
     */

    MLNN(const char *path) {
        ifstream in;
        in.open(path, ios::in | ios::binary);
        if (!in.is_open()) {
            throw runtime_error("Cannot open the given file");
        }

        // Load header stuff
        in.read((char *)&layers, st_size);
        activations.reserve(layers);
        nodes = vector<size_t>(layers);
        for (size_t i = 0; i < layers; i++) {
            size_t node;
            in.read((char *)&node, st_size);
            activations[i++].reserve(nodes[i] = node);
        }
        input = nodes[0] - 1; // Set the input

        const size_t layersToLoad = layers - 1;
        weights.reserve(layersToLoad);
        change.reserve(layersToLoad);
        // Load weights and change
        for (size_t i = 0; i < layersToLoad; i++) {
            const size_t output = nodes[i + 1], input = nodes[i];
            weights[i].reserve(input);
            change[i].reserve(input);
            for (size_t j = 0; j < input; j++) {
                weights[i][j].reserve(output);
                change[i][j].reserve(output);
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

    oneDimensional predict(const oneDimensional &inputs) const {
        // Verify the input is the correct size
        if (inputs.capacity() != input) {
            throw range_error("Incorrect number of inputs");
        }

        // Create the input output vectors
        oneDimensional in(nodes[1]), out;

        // Run the activations for the input layer
        for (size_t j = 0; j < nodes[1]; j++) {
            double sum = 0.0;
            for (size_t i = 0; i < nodes[0]; i++) {
                sum += inputs[i] * weights[0][i][j];
            }
            in[j] = sigmoid(sum);
        }

        // Run the activations for hidden layers
        const size_t end = layers - 1;
        for (size_t inputLayer = 1; inputLayer < end; inputLayer++) {
            const size_t outputLayer = inputLayer + 1;
            out.resize(nodes[outputLayer]);
            for (size_t j = 0; j < nodes[outputLayer]; j++) {
                double sum = 0.0;
                for (size_t i = 0; i < nodes[inputLayer]; i++) {
                    sum += in[i] * weights[inputLayer][i][j];
                }
                out[j] = sigmoid(sum);
            }
            if (outputLayer != end) {
                in = out;
            }
        }

        return out;
    }

protected:
    void update(const oneDimensional &inputs) {
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

    twoDimensional backPropagate(const oneDimensional &targets) {
        const size_t last = layers - 1;
        size_t i = 0;
        twoDimensional deltas(layers);

        for (const size_t &count : nodes) {
            deltas[i++].reserve(count);
        }

        // Calculate error/loss, working from back to front
        for (size_t k = 0; k < nodes[last]; k++) {
            deltas[last][k] = dsigmoid(activations[last][k]) * (targets[k] - activations[last][k]);
        }

        for (size_t outputLayer = last; outputLayer > 0; outputLayer--) {
            const size_t inputLayer = outputLayer - 1;
            for (size_t j = 0; j < nodes[inputLayer]; j++) {
                double error = 0.0;
                for (size_t k = 0; k < nodes[outputLayer]; k++) {
                    error += deltas[outputLayer][k] * weights[inputLayer][j][k];
                }
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

    void updateWeights(const twoDimensional &deltas, const double learningRate, const double momentumFactor) {
        // Update weights with the given deltas
        for (size_t outputLayer = layers - 2; outputLayer > 0; outputLayer--) {
            const size_t inputLayer = outputLayer - 1;
            for (size_t j = 0; j < nodes[inputLayer]; j++) {
                for (size_t k = 0; k < nodes[outputLayer]; k++) {
                    const double change = deltas[outputLayer][k] * activations[inputLayer][j];
                    weights[inputLayer][j][k] += learningRate * change + momentumFactor * this->change[inputLayer][j][k];
                    this->change[inputLayer][j][k] = change;
                }
            }
        }
    }

    double calculateError(const oneDimensional &targets) const {
        const size_t end = layers - 1;
        // Calculate error
        double error = 0.0;
        for (size_t k = 0; k < nodes[end]; k++) {
            error += (targets[k] - activations[end][k]) * (targets[k] - activations[end][k]);
        }
        return error / nodes[end];
    }

public:
    void test(const threeDimensional &patterns) const {
        for (const twoDimensional &pattern : patterns) {
            cout << prettifyVector(pattern[0]);
            cout << " -> " << prettifyVector(predict(pattern[0])) << endl;
        }
    }

    void test(const twoDimensional &patterns) const {
        for (const oneDimensional &pattern : patterns) {
            // cout << prettifyVector(pattern);
            cout << " -> " << prettifyVector(predict(pattern)) << endl;
        }
    }

    double train(const threeDimensional &patterns, const size_t iterations = 10000, const double N = 0.5, const double M = 0.1) {
        const size_t last = layers - 1;
        for (const twoDimensional &pattern : patterns) {
            if (pattern[0].size() != input) {
                throw range_error("Wrong number of input values");
            }
            if (pattern[1].size() != nodes[last]) {
                throw range_error("Wrong number of target values");
            }
        }
        for (size_t i = 0; i < iterations; i++) {
            for (const twoDimensional &pattern : patterns) {
                update(pattern[0]);
                updateWeights(backPropagate(pattern[1]), N, M);
            }
            if (i % 10000 == 0) // Should make adjustable
            {
                double error = 0.0;
                for (const twoDimensional &pattern : patterns) {
                    update(pattern[0]);
                    error += calculateError(pattern[1]);
                }
                cout << "MSE: " << error << endl;
            }
        }

        // Get error
        double error = 0.0;
        for (const twoDimensional &pattern : patterns) {
            update(pattern[0]);
            error += calculateError(pattern[1]);
        }
        return error;
    }

    double randTrain(threeDimensional patterns, const size_t iterations = 10000, const double N = 0.5, const double M = 0.1) {
        const size_t last = layers - 1;

        // Verify the patterns are the right size to be used in the network
        for (const twoDimensional &pattern : patterns) {
            if (pattern[0].size() != input) {
                throw range_error("Wrong number of input values");
            }
            if (pattern[1].size() != nodes[last]) {
                throw range_error("Wrong number of target values");
            }
        }

        for (size_t i = 0; i < iterations; i++) {
            // Randomize the values
            shuffle(patterns.begin(), patterns.end(), g);

            // Train on the values
            for (const twoDimensional &pattern : patterns) {
                update(pattern[0]);
                updateWeights(backPropagate(pattern[1]), N, M);
            }

            // Verbose on training (Should make adjustable)
            if (i % 10000 == 0) {
                double error = 0.0;
                for (const twoDimensional &pattern : patterns) {
                    update(pattern[0]);
                    error += calculateError(pattern[1]);
                }
                cout << "MSE: " << error << endl;
            }
        }

        // Get the error of the network
        double error = 0.0;
        for (const twoDimensional &pattern : patterns) {
            update(pattern[0]);
            error += calculateError(pattern[1]);
        }
        return error;
    }

    string toString() const {
        string output;

        size_t i = 0;
        for (const twoDimensional &weight : weights) {
            output += "\nLayer " + to_string(i++) + ":\n" + prettifyVector(weight);
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

    bool toFile(const char *path) const {
        // Attempt to open the output file
        ofstream out;
        out.open(path, ios::out | ios::trunc | ios::binary);
        if (!out.is_open()) {
            return false; // Failed to open
        }

        /**
         * Write header stuff to file
         *
         * Number of layers then the number of nodes per layer
         */

        out.write((char *)&layers, st_size);
        for (const size_t &node : nodes) {
            out.write((char *)&node, st_size);
        }

        // Write the values to file
        const size_t layersToExport = layers - 1;
        for (size_t i = 0; i < layersToExport; i++) {
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