#pragma once

#include "json.hpp"

#include <cstdlib>
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <exception>
#include <sstream>
#include <fstream>

using nlohmann::json;
using std::ifstream;
using std::ofstream;
using std::string;
using std::stringstream;
using std::to_string;
using std::vector;

json loadFromFile(const char *path)
{
    json j;

    ifstream i(path);

    i >> j;

    i.close();

    return j;
}
void saveToFile(json json, const char *path)
{
    ofstream myfile(path);
    myfile << json;
    myfile.close();
}

vector<double> fill(const size_t size, const vector<size_t> inputs)
{
    vector<double> output(size);
    for (const size_t input : inputs)
        output[input] = 1.0;
    return output;
}

template <class T>
T rand(const T a, const T b)
{
    return (b - a) * ((T)std::rand() / RAND_MAX) + a;
}

vector<vector<double>> makeMatrix(const size_t I, const size_t J, const double fill = 1.0)
{
    vector<vector<double>> m(I);
    for (size_t i = 0; i < I; i++)
    {
        m[i] = vector<double>(J);
        for (size_t j = 0; j < J; j++)
            m[i][j] = fill;
    }
    return m;
}

vector<double> makeRandom(const size_t I, const double a, const double b)
{
    vector<double> o(I);
    for (size_t i = 0; i < I; i++)
    {
        o[i] = rand(a, b);
    }
    return o;
}

vector<vector<double>> makeRandomMatrix(const size_t I, const size_t J, const double a, const double b)
{
    vector<vector<double>> m(I);
    for (size_t i = 0; i < I; i++)
    {
        m[i] = vector<double>(J);
        for (size_t j = 0; j < J; j++)
            m[i][j] = rand(a, b);
    }
    return m;
}
const inline float sigmoid(const float x)
{
    return tanhf(x);
}

const inline double sigmoid(const double x)
{
    return tanh(x);
}

const inline long double sigmoid(const long double x)
{
    return tanhl(x);
}

const inline long double dsigmoid(const long double y)
{
    return 1.0l - (y * y);
}

const inline double dsigmoid(const double y)
{
    return 1.0 - (y * y);
}

const inline float dsigmoid(const long y)
{
    return 1.0f - (y * y);
}

template <class T>
string nicePrint(const vector<T> pattern)
{
    stringstream ss;
    ss << "[";

    for (const T p : pattern)
        ss << p << ", ";

    const string output = ss.str();
    return output.substr(0, output.length() - 2) + "]";
}

template <class T>
string nicePrint(const vector<vector<T>> pattern)
{
    stringstream ss;

    ss << "[";

    for (const vector<T> p : pattern)
        ss << "\n    " << nicePrint(p) << ",";

    const string output = ss.str();
    return output.substr(0, output.length() - 1) + "\n]";
}
