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

typedef vector<double> oneDimensional;
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
#define rnd(a, b) (b - a) * (static_cast<double>(std::rand()) * RAND_DIV) + a

twoDimensional makeMatrix(const size_t I, const size_t J, const double fill = 1.0) {
    twoDimensional m(I);
    for (size_t i = 0; i < I; i++) {
        m[i] = oneDimensional(J);
        for (size_t j = 0; j < J; j++) {
            m[i][j] = fill;
        }
    }
    return m;
}

oneDimensional makeRandom(const size_t I, const double a, const double b) {
    oneDimensional o(I);
    for (size_t i = 0; i < I; i++) {
        o[i] = rnd(a, b);
    }
    return o;
}

twoDimensional makeRandomMatrix(const size_t I, const size_t J, const double a, const double b) {
    twoDimensional m(I);
    for (size_t i = 0; i < I; i++) {
        m[i] = oneDimensional(J);
        for (size_t j = 0; j < J; j++) {
            m[i][j] = rnd(a, b);
        }
    }
    return m;
}

#define sigmoid(x) tanh(x)

const inline constexpr double dsigmoid(const double y) {
    const double t = sigmoid(y);
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
