// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"

/**
 * 
 */
class MXNET_API ANN
{
public:
	ANN();
	~ANN();
	void Initialize(TArray<int> LayerDims, TArray<int> ActivationIDs, double LearningRate);
	void Forward(TArray<double>& Inputs);
	void Backward(TArray<double>& Inputs, TArray<double>& Outputs);
	void ApplyGradientDeltas();
	double ActivationFunction(double Input, int ActivationFunctionID);
	double ActivationFunctionDerivative(double Input, int ActivationFunctionID);
	int NeuronStart(int LayerID);
	int WeightStart(int LayerID);
	TArray<double> GetSlice(TArray<double>& TargetList, int StartIndex, int Length);
	void InitWeights();

	//////

	TArray<double> NetInputs;
	TArray<double> Activations;
	TArray<double> Derivatives;
	TArray<double> Errors;
	TArray<double> Bias;
	TArray<double> BiasDelta;
	TArray<double> BiasMean;
	TArray<double> BiasVar;
	TArray<double> Weights;
	TArray<double> WeightDeltas;
	TArray<double> DeltaMean;
	TArray<double> DeltaVar;
	TArray<int> ActivationFunctionIDs;
	TArray<int> LayerCounts;
	double LearningRate = 0.0001;
	double ReLUConst = 0.01;
	double ELUConst = 1.0;
	double AdamB1 = 0.9;
	double AdamB2 = 0.999;
	int Iteration = 1;
	int BatchIteration = 1;
	int BatchSize = 1;
};
