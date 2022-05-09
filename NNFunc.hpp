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
#include <exception>

using nlohmann::json;
using std::exception;
using std::ifstream;
using std::istringstream;
using std::ofstream;
using std::pair;
using std::string;
using std::stringstream;
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

#define RAND_DIV 3.0518509475997192270657620838392176665365695953369140625e-05
#define rand(a, b) (b - a) * (static_cast<double>(std::rand()) * RAND_DIV) + a

vector<vector<double>> makeMatrix(const size_t I, const size_t J, const double fill = 1.0)
{
    vector<vector<double>> m(I);
    for (size_t i = 0; i < I; i++)
    {
        m[i] = vector<double>(J);
        for (size_t j = 0; j < J; j++)
        {
            m[i][j] = fill;
        }
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
        {
            m[i][j] = rand(a, b);
        }
    }
    return m;
}

#define sigmoid(x) tanh(x)

const inline constexpr double dsigmoid(const double y)
{
    const double t = sigmoid(y);
    return 1.0 - (t * t);
}

template <class T>
const string nicePrint(const vector<T> pattern)
{
    stringstream ss;
    ss << "[";

    for (const T p : pattern)
        ss << p << ", ";

    const string output = ss.str();
    return output.substr(0, output.length() - 2) + "]";
}

template <class T>
const string nicePrint(const vector<vector<T>> pattern)
{
    stringstream ss;

    ss << "[";

    for (const vector<T> p : pattern)
        ss << "\n    " << nicePrint(p) << ",";

    const string output = ss.str();
    return output.substr(0, output.length() - 1) + "\n]";
}

/**
 * For future stuff
 */
pair<size_t, vector<size_t>> pullLine(ifstream &infile)
{
    pair<size_t, vector<size_t>> output;
    string line;
    if (!getline(infile, line))
        throw new exception;

    istringstream iss(line);
    size_t a;

    if (!(iss >> output.first)) // error
        throw new exception;

    while (iss >> a)
    {
        output.second.push_back(a);
    }

    return output;
}