#pragma once

#include <cstdlib>
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <exception>
#include "json.hpp"
#include "NNFunc.hpp"

using nlohmann::json;
using std::copy;
using std::cout;
using std::endl;
using std::range_error;
using std::string;
using std::to_string;
using std::vector;

class MLNN
{
public:
    const size_t input, layers;
    const vector<size_t> nodes;

private:
    vector<vector<double>> activations;
    vector<vector<vector<double>>> weights, change;

public:
    MLNN(const size_t inputLayer, const vector<size_t> hiddenLayers, const size_t outputLayer)
        : input(inputLayer), layers(hiddenLayers.size() + 2), activations(vector<vector<double>>(layers)),
          nodes(createNodes(layers, inputLayer, hiddenLayers, outputLayer)),
          weights(vector<vector<vector<double>>>(layers - 1)), change(vector<vector<vector<double>>>(layers - 1))
    {
        // Load activations
        size_t i = 0;
        for (const size_t node : nodes)
        {
            activations[i++] = vector<double>(node);
        }

        // Load weights and change
        for (size_t outputLayer = layers - 2; outputLayer > 0; outputLayer--)
        {
            const size_t inputLayer = outputLayer - 1;
            weights[inputLayer] = makeRandomMatrix(nodes[inputLayer], nodes[outputLayer], -0.2, 0.2);
            change[inputLayer] = makeMatrix(nodes[inputLayer], nodes[outputLayer]);
        }

        const size_t input = layers - 2, output = layers - 1;
        weights[input] = makeRandomMatrix(nodes[input], nodes[output], -2.0, 2.0);
        change[input] = makeMatrix(nodes[input], nodes[output]);
    }

    // Will change. Doesn't compile for most compilers... just GCC
    MLNN(const json &json)
        : input(json[0]), layers(json[1]),
          nodes(json[2]), activations(json[3]),
          weights(json[4]), change(json[5]) {}

    MLNN(const MLNN &MLNN)
        : input(MLNN.input), layers(MLNN.layers), nodes(MLNN.nodes),
          activations(MLNN.activations),
          weights(MLNN.weights), change(MLNN.change) {}

    vector<double> predict(const vector<double> &inputs)
    {
        update(inputs);
        return activations[layers - 1];
    }

protected:
    static vector<size_t> createNodes(const size_t layers, const size_t inputLayer, const vector<size_t> hiddenLayers, const size_t outputLayer)
    {
        vector<size_t> nodes(layers);

        size_t i = 0;
        // No need to set activation here since it gets set in the updater
        nodes[i++] = inputLayer + 1; // + 1 for bias layer
        for (const size_t node : hiddenLayers)
        {
            nodes[i++] = node;
        }
        nodes[i] = outputLayer;

        return nodes;
    }

    void update(const vector<double> &inputs)
    {
        if (inputs.capacity() != input)
        {
            throw range_error("Incorrect number of inputs");
        }
        activations[0] = inputs;

        // Activations for input + hidden
        for (size_t inputLayer = 0; inputLayer < layers - 1; inputLayer++)
        {
            const size_t outputLayer = inputLayer + 1;
            for (size_t j = 0; j < nodes[outputLayer]; j++)
            {
                double sum = 0.0;
                for (size_t i = 0; i < nodes[inputLayer]; i++)
                {
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

    vector<vector<double>> backPropagate(const vector<double> &targets)
    {
        const size_t last = layers - 1;
        if (targets.capacity() != nodes[last])
        {
            throw range_error("Wrong number of target values");
        }

        size_t i = 0;
        vector<vector<double>> deltas(layers);

        for (const size_t count : nodes)
        {
            deltas[i++] = vector<double>(count);
        }

        // Calculate error/loss
        for (size_t k = 0; k < nodes[last]; k++)
        {
            deltas[last][k] = dsigmoid(activations[last][k]) * (targets[k] - activations[last][k]);
        }

        for (size_t outputLayer = last; outputLayer > 0; outputLayer--) // -2
        {
            const size_t inputLayer = outputLayer - 1;
            for (size_t j = 0; j < nodes[inputLayer]; j++)
            {
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

    void updateWeights(const vector<vector<double>> &deltas, const double learningRate, const double momentumFactor)
    {
        // Update weights

        for (size_t outputLayer = layers - 2; outputLayer > 0; outputLayer--)
        {
            const size_t inputLayer = outputLayer - 1;
            for (size_t j = 0; j < nodes[inputLayer]; j++)
            {
                for (size_t k = 0; k < nodes[outputLayer]; k++)
                {
                    const double change = deltas[outputLayer][k] * activations[inputLayer][j];
                    weights[inputLayer][j][k] += learningRate * change + momentumFactor * this->change[inputLayer][j][k];
                    this->change[inputLayer][j][k] = change;
                }
            }
        }
    }

    double calculateError(const vector<double> &targets)
    {
        const size_t end = layers - 1;
        // Calculate error (MSE)
        double error = 0.0;
        for (size_t k = 0; k < nodes[end]; k++)
        {
            error += (targets[k] - activations[end][k]) * (targets[k] - activations[end][k]);
        }
        return error / nodes[end];
    }

public:
    void test(const vector<vector<vector<double>>> &patterns)
    {
        for (const vector<vector<double>> pattern : patterns)
        {
            cout << nicePrint(pattern[0]);
            cout << " -> " << nicePrint(predict(pattern[0])) << endl;
        }
    }

    void test(const vector<vector<double>> &patterns)
    {
        for (const vector<double> pattern : patterns)
        {
            // cout << nicePrint(pattern);
            cout << " -> " << nicePrint(predict(pattern)) << endl;
        }
    }

    double train(const vector<vector<vector<double>>> &patterns, const size_t iterations = 10000, const double N = 0.5, const double M = 0.1)
    {
        double error;
        for (size_t i = 0; i < iterations; i++)
        {
            for (const vector<vector<double>> pattern : patterns)
            {
                update(pattern[0]);
                // backPropagate(pattern[1]);
                updateWeights(backPropagate(pattern[1]), N, M);
            }
            if (i % 10000 == 0)
            {
                error = 0.0;
                for (const vector<vector<double>> pattern : patterns)
                {
                    update(pattern[0]);
                    error += calculateError(pattern[1]);
                }
                cout << "MSE: " << error << endl;
            }
        }

        // Get error
        error = 0.0;
        for (const vector<vector<double>> pattern : patterns)
        {
            update(pattern[0]);
            error += calculateError(pattern[1]);
        }
        return error;
    }

    string toString()
    {
        string output;

        size_t i = 0;
        for (const vector<vector<double>> weight : weights)
        {
            output += "\nLayer " + to_string(i++) + ":\n" + nicePrint(weight);
        }

        return output;
    }

    json toJSON()
    {
        return {input,
                layers,
                nodes,
                activations,
                weights,
                change};
    }
};