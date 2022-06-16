#pragma once

#include <cmath>
#include <exception>
#include <string>
#include <vector>

using std::exception;
using std::pair;
using std::string;
using std::to_string;
using std::vector;

// Some type defines because it is somewhat tedious to type out
typedef vector<float> oneDimensional;
typedef vector<oneDimensional> twoDimensional;
typedef vector<twoDimensional> threeDimensional;

oneDimensional fill(const size_t size, const vector<size_t> &inputs) {
    oneDimensional output(size);
    for (const size_t input : inputs) {
        output[input] = 1.0;
    }
    return output;
}

#define RAND_DIV 3.0518509475997192270657620838392176665365695953369140625e-05
#define rnd(a, b) (b - a) * (static_cast<float>(std::rand()) * RAND_DIV) + a

twoDimensional makeMatrix(const size_t I, const size_t J, const float fill = 1.0) {
    twoDimensional m(I);
    for (size_t i = 0; i < I; i++) {
        m[i] = oneDimensional(J);
        for (size_t j = 0; j < J; j++) {
            m[i][j] = fill;
        }
    }
    return m;
}

oneDimensional makeRandom(const size_t I, const float a, const float b) {
    oneDimensional o(I);
    for (size_t i = 0; i < I; i++) {
        o[i] = rnd(a, b);
    }
    return o;
}

twoDimensional makeRandomMatrix(const size_t I, const size_t J, const float a, const float b) {
    twoDimensional m(I);
    for (size_t i = 0; i < I; i++) {
        m[i] = oneDimensional(J);
        for (size_t j = 0; j < J; j++) {
            m[i][j] = rnd(a, b);
        }
    }
    return m;
}

#define sigmoid(x) tanhf(x)

const inline constexpr float dsigmoid(const float y) {
    const float t = sigmoid(y);
    return 1.0 - (t * t);
}

/**
 * Takes a vector and creates a pretty string from it, displaying all of it's values
 *
 * @param pattern The vector data, full of numerical values
 */

template <class T>
const string prettifyVector(const vector<T> &pattern) {
    string output = "[";
    for (const T &p : pattern) {
        output += to_string(p) + ", ";
    }
    return output.substr(0, output.length() - 2) + ']';
}

/**
 * Takes a two dimensional vector and creates a pretty string from it, displaying all of it's values
 *
 * @param pattern The vector data, full of numerical values
 */

template <class T>
const string prettifyVector(const vector<vector<T>> &pattern) {
    string output = "[";
    for (const vector<T> &p : pattern) {
        output += "\n    " + prettifyVector(p) + ',';
    }
    return output.substr(0, output.length() - 1) + "\n]";
}
