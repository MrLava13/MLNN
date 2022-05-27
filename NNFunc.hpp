#pragma once

#include <cmath>
#include <exception>
#include <string>
#include <sstream>
#include <vector>

using std::exception;
using std::pair;
using std::string;
using std::vector;
using std::stringstream;

vector<double> fill(const size_t size, const vector<size_t> inputs) {
    vector<double> output(size);
    for (const size_t input : inputs) {
        output[input] = 1.0;
    }
    return output;
}

#define RAND_DIV 3.0518509475997192270657620838392176665365695953369140625e-05
#define rnd(a, b) (b - a) * (static_cast<double>(std::rand()) * RAND_DIV) + a

vector<vector<double>> makeMatrix(const size_t I, const size_t J, const double fill = 1.0) {
    vector<vector<double>> m(I);
    for (size_t i = 0; i < I; i++) {
        m[i] = vector<double>(J);
        for (size_t j = 0; j < J; j++) {
            m[i][j] = fill;
        }
    }
    return m;
}

vector<double> makeRandom(const size_t I, const double a, const double b) {
    vector<double> o(I);
    for (size_t i = 0; i < I; i++) {
        o[i] = rnd(a, b);
    }
    return o;
}

vector<vector<double>> makeRandomMatrix(const size_t I, const size_t J, const double a, const double b) {
    vector<vector<double>> m(I);
    for (size_t i = 0; i < I; i++) {
        m[i] = vector<double>(J);
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

template <class T>
const string nicePrint(const vector<T> &pattern) {
    stringstream ss;
    ss << "[";

    for (const T p : pattern)
        ss << p << ", ";

    const string output = ss.str();
    return output.substr(0, output.length() - 2) + "]";
}

template <class T>
const string nicePrint(const vector<vector<T>> &pattern) {
    stringstream ss;

    ss << "[";

    for (const vector<T> p : pattern)
        ss << "\n    " << nicePrint(p) << ",";

    const string output = ss.str();
    return output.substr(0, output.length() - 1) + "\n]";
}
