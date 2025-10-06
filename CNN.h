// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ANN.h"
#include "MNIST_Helper.h"
#include "Components/SceneCaptureComponent2D.h"
#include "CNN.generated.h"

UCLASS()
class MXNET_API ACNN : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ACNN();

	void Initialize(TArray<int> ActivationIDs, TArray<int> PoolingSched, TArray<int> FilterCounts, TArray<int> FilterDimSched, TArray<int> PaddingSched, double LearnRate, int ImageDimPx, TArray<int> ANNLayerCounts, TArray<int> ANNActivationIDs, double LearningRate);
	void Forward(TArray<TArray<double>>& Inputs);
	void Backward(TArray<TArray<double>>& Inputs, TArray<double> Outputs);
	void ApplyGradientDeltas();
	TArray<double> GradientsWRTInputs(int LayerID);
	TArray<double> GradientsWRTOutputs(TArray<double>& Inputs,int LayerID);
	void GradientsWRTFilters(int LayerID);
	void ActivationFunction(int LayerID);
	void ActivationFunctionDerivative(int LayerID);
	void Convolve(int LayerID);
	void LayerNorm(int LayerID);
	TArray<double> LayerNormBackward(TArray<double>& Gradients, int LayerID);
	TArray<double> Flatten2DArray(TArray<TArray<double>>& Inputs);
	int FeatureStart(int LayerID);
	int FilterStart(int LayerID);
	int BiasStart(int LayerID);
	TArray<TArray<double>> GetFeatureMap(int LayerID, int MapID);
	double GetMean(TArray<double>& Inputs);
	double GetStdDev(TArray<double>& Inputs);
	void SaveNetwork(FString FilePath);
	void LoadNetwork(FString FilePath);

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

///////////////////

public:
	ANN* NeuralNet; //Network head
	MNIST_Helper* Helper;
	int Iteration = 1;
	int BatchIteration = 1;
	int BatchSize = 1;
	double LearningRate = 0.0001;
	int ImageSizePx = 0;
	TArray<int> StrideSchedule;
	TArray<int> FilterDimSchedule;
	TArray<int> ActivationFunctionSchedule;
	TArray<int> PaddingSchedule;
	TArray<double> Filters;
	TArray<double> Bias;
	TArray<double> BiasDelta;
	TArray<double> BiasMean;
	TArray<double> BiasVar;
	TArray<double> Deltas;
	TArray<double> DeltaMean;
	TArray<double> DeltaVar;
	TArray<double> Features;
	TArray<double> NormFeatures;
	TArray<double> NetFeatures;
	TArray<double> Derivatives;
	TArray<double> Errors;
	TArray<double> TempGradientsWRTInputs;
	TArray<double> NormMean;
	TArray<double> NormVar;
	TArray<double> NormGamma;
	TArray<double> NormGammaDelta;
	TArray<double> NormGammaMean;
	TArray<double> NormGammaVar;
	TArray<double> NormBeta;
	TArray<double> NormBetaDelta;
	TArray<double> NormBetaMean;
	TArray<double> NormBetaVar;
	TArray<int> MapCounts;
	TArray<int> MapDims;
	TArray<int> FilterDims;
	TArray<double> TransitionActivations;
	TArray<double> TransitionWeights;
	TArray<double> TransitionDeltas;
	TArray<double> TransitionDeltasMean;
	TArray<double> TransitionDeltasVar;
	USceneCaptureComponent2D* CaptureComp; //For implementation in level
};
