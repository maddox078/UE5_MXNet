// Fill out your copyright notice in the Description page of Project Settings.


#include "CNN.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/SceneCapture2D.h"
#include "Kismet/KismetRenderingLibrary.h"
#include "ImageUtils.h"
#include "Kismet/GameplayStatics.h"
#include "Camera/CameraComponent.h"
#include "Components/SceneCaptureComponent2D.h"

// Sets default values
ACNN::ACNN()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	UTextureRenderTarget2D* RenderTarget = NewObject<UTextureRenderTarget2D>();

	NeuralNet = new ANN();
	Helper = new MNIST_Helper();
	CaptureComp = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("SceneCapture"));
	CaptureComp->SetupAttachment(RootComponent);
	CaptureComp->bCaptureEveryFrame = false;
	CaptureComp->bCaptureOnMovement = false;
	CaptureComp->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
	
	RenderTarget->InitAutoFormat(256, 256);
	RenderTarget->RenderTargetFormat = RTF_RGBA8;
	RenderTarget->ClearColor = FLinearColor::Black;
	RenderTarget->UpdateResourceImmediate(true);
	CaptureComp->TextureTarget = RenderTarget;
	CaptureComp->ShowFlags.SetPostProcessing(true);
	CaptureComp->ShowFlags.SetAtmosphere(true);
	CaptureComp->ShowFlags.SetSkyLighting(true);
}

// Called when the game starts or when spawned
void ACNN::BeginPlay()
{
	Super::BeginPlay();
	
	///TEST///
	TArray<int> Activations;
	TArray<int> Pooling;
	TArray<int> FilterCount;
	TArray<int> FilterDim;
	TArray<int> Padding;
	TArray<int> ANNCount;
	TArray<int> ANNActivations;
	TArray<TArray<double>> TestInput;
	TArray<double> SubTestInput;

	Activations.Add(2);
	Activations.Add(2);
	Activations.Add(2);
	Activations.Add(2);
	//Activations.Add(2);

	Pooling.Add(2);
	Pooling.Add(2);
	Pooling.Add(2);
	Pooling.Add(2);
	//Pooling.Add(2);

	Padding.Add(1);
	Padding.Add(1);
	Padding.Add(1);
	Padding.Add(1);
	//Padding.Add(1);

	FilterCount.Add(16);
	FilterCount.Add(32);
	FilterCount.Add(64);
	FilterCount.Add(128);
	//FilterCount.Add(256);

	FilterDim.Add(3);
	FilterDim.Add(3);
	FilterDim.Add(3);
	FilterDim.Add(3);
	//FilterDim.Add(3);

	ANNCount.Add(1260);
	ANNCount.Add(5 * (16 * 16));
	//ANNCount.Add(10);
	//ANNCount.Add(10);

	ANNActivations.Add(2);
	ANNActivations.Add(1);
	//ANNActivations.Add(9);

	this->Initialize(Activations, Pooling, FilterCount, FilterDim, Padding, 0.0003, 256, ANNCount, ANNActivations, 0.0002);

	this->Helper->LoadTrainingFolder(this->Helper->YOLOFolder);
	this->Helper->LoadTrainingFolderLabels(this->Helper->YOLOLabelsTxt);

}

// Called every frame
void ACNN::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

void ACNN::Initialize(TArray<int> ActivationIDs, TArray<int> PoolingSched, TArray<int> FilterCounts, TArray<int> FilterDimSched, TArray<int> PaddingSched, double CNNLearningRate, int ImageDimPx, TArray<int> ANNLayerCounts, TArray<int> ANNActivationIDs, double ANNLearningRate)
{
	int A = 0;
	int B = 0;
	int C = 0;
	int D = 0;
	int E = 0;
	double StdDev = 0;
	double TempImageDims = 0;
	int InputCount = 0;
	int FilterCount = 0;
	int FeatureCount = 0;
	int BiasCount = 0;
	int TransitionDeltaCount = 0;
	int TransitionActivationCount = 0;

	StrideSchedule = PoolingSched;
	FilterDimSchedule = FilterDimSched;
	ActivationFunctionSchedule = ActivationIDs;
	MapCounts = FilterCounts;
	FilterDims = FilterDimSched;
	ImageSizePx = ImageDimPx;
	PaddingSchedule = PaddingSched;
	LearningRate = CNNLearningRate;

	MapDims.SetNum(ActivationIDs.Num());

	NeuralNet->Initialize(ANNLayerCounts, ANNActivationIDs, ANNLearningRate);

	TempImageDims = FMath::Floor((ImageDimPx + 2.0 * PaddingSchedule[0] - FilterDimSchedule[0]) / StrideSchedule[0]) + 1.0;

	while (A < ActivationIDs.Num())
	{
		if (A == 0)
		{
			InputCount = 1;
		}
		else
		{
			InputCount = MapCounts[A - 1];
		}

		MapDims[A] = TempImageDims;

		FeatureCount += FilterCounts[A] * TempImageDims * TempImageDims;
		FilterCount += FilterCounts[A] * InputCount * FilterDimSched[A] * FilterDimSched[A];
		BiasCount += FilterCounts[A];

		if (A < ActivationIDs.Num() - 1)
		{
			TempImageDims = FMath::Floor((TempImageDims + 2.0 * PaddingSchedule[A + 1] - FilterDimSchedule[A + 1]) / StrideSchedule[A + 1]) + 1.0;
		}

		A++;
	}

	Features.SetNumZeroed(FeatureCount);
	NetFeatures.SetNumZeroed(FeatureCount);
	Errors.SetNumZeroed(FeatureCount);
	TempGradientsWRTInputs.SetNumZeroed(FeatureCount);
	Derivatives.SetNumZeroed(FeatureCount);
	NormFeatures.SetNumZeroed(FeatureCount);

	Bias.SetNumZeroed(BiasCount);
	BiasDelta.SetNumZeroed(BiasCount);
	BiasMean.SetNumZeroed(BiasCount);
	BiasVar.SetNumZeroed(BiasCount);
	NormGamma.SetNumZeroed(BiasCount);
	NormGammaDelta.SetNumZeroed(BiasCount);
	NormGammaMean.SetNumZeroed(BiasCount);
	NormGammaVar.SetNumZeroed(BiasCount);
	NormBeta.SetNumZeroed(BiasCount);
	NormBetaDelta.SetNumZeroed(BiasCount);
	NormBetaMean.SetNumZeroed(BiasCount);
	NormBetaVar.SetNumZeroed(BiasCount);
	NormMean.SetNumZeroed(BiasCount);
	NormVar.SetNumZeroed(BiasCount);

	Deltas.SetNumZeroed(FilterCount);
	DeltaMean.SetNumZeroed(FilterCount);
	DeltaVar.SetNumZeroed(FilterCount);
	Filters.SetNumZeroed(FilterCount);

	TransitionDeltaCount = MapDims.Last() * MapDims.Last() * FilterCounts.Last() * NeuralNet->LayerCounts[0];
	TransitionWeights.SetNumZeroed(TransitionDeltaCount);
	TransitionDeltas.SetNumZeroed(TransitionDeltaCount);
	TransitionDeltasMean.SetNumZeroed(TransitionDeltaCount);
	TransitionDeltasVar.SetNumZeroed(TransitionDeltaCount);
	TransitionActivations.SetNumZeroed(NeuralNet->LayerCounts[0]);

	StdDev = FMath::Sqrt(6.0 / (double)(MapDims.Last() * MapDims.Last() * FilterCounts.Last()));

	A = 0;
	while (A < BiasCount)
	{
		NormVar[A] = 1.0;
		NormGamma[A] = 1.0;

		A++;
	}

	A = 0;
	while (A < FilterCount)
	{
		Filters[A] = FMath::FRandRange(-0.1, 0.1);

		A++;
	}

	A = 0;
	while (A < TransitionDeltaCount)
	{
		TransitionWeights[A] = FMath::FRandRange(-StdDev, StdDev);

		A++;
	}
}

void ACNN::Forward(TArray<TArray<double>>& Inputs)
{
	int A = 0;
	int B = 0;
	double NetSum = 0;
	TArray<double> FlatInput = Flatten2DArray(Inputs);

	while (A < FlatInput.Num())
	{
		Features[A] = FlatInput[A];

		A++;
	}

	A = 0;
	while (A < MapCounts.Num())
	{
		Convolve(A);
		LayerNorm(A);
		ActivationFunction(A);
		ActivationFunctionDerivative(A);

		A++;
	}

	A = FeatureStart(MapCounts.Num() - 1);
	FlatInput = NeuralNet->GetSlice(Features,A,Features.Num() - A);
	
	A = 0;
	while (A < TransitionActivations.Num())
	{
		NetSum = 0;

		B = 0;
		while (B < FlatInput.Num())
		{
			NetSum += TransitionWeights[A * FlatInput.Num() + B] * FlatInput[B];

			B++;
		}

		TransitionActivations[A] = NetSum;

		A++;
	}

	NeuralNet->Forward(TransitionActivations);
}

void ACNN::Convolve(int LayerID)
{
	int A = 0;
	int B = 0;
	int C = 0;
	int X = 0;
	int Y = 0;
	int Z = 0;
	int FeatureOffset = 0;
	int InputOffset = 0;
	int BiasOffset = 0;
	int FilterOffset = 0;
	int TempMapDims = 0;
	int InputCount = 0;
	int InputDim = 0;
	int TgtIndex = 0;
	int NewX = 0;
	int NewY = 0;
	int FeatureIndex = 0;
	int FilterIndex = 0;
	double NetSum = 0;

	if (LayerID > 0)
	{
		InputOffset = FeatureStart(LayerID - 1);
		FeatureOffset = FeatureStart(LayerID);
		BiasOffset = BiasStart(LayerID);
		FilterOffset = FilterStart(LayerID);
		InputCount = MapCounts[LayerID - 1];
		InputDim = MapDims[LayerID - 1];
		TempMapDims = FMath::Floor((MapDims[LayerID - 1] + 2.0 * PaddingSchedule[LayerID] - FilterDims[LayerID]) / StrideSchedule[LayerID]) + 1.0;
	}
	else
	{
		InputCount = 1;
		InputDim = ImageSizePx;
		TempMapDims = FMath::Floor((InputDim + 2.0 * PaddingSchedule[0] - FilterDims[0]) / StrideSchedule[0]) + 1.0;
	}

	while (A < MapCounts[LayerID])
	{
		B = 0;
		while (B < TempMapDims)
		{
			C = 0;
			while (C < TempMapDims)
			{
				NetSum = 0;

				X = 0;
				while (X < InputCount)
				{
					TgtIndex = FilterOffset + A * InputCount * FilterDims[LayerID] * FilterDims[LayerID] + X * FilterDims[LayerID] * FilterDims[LayerID];

					Y = 0;
					while (Y < FilterDims[LayerID])
					{
						Z = 0;
						while (Z < FilterDims[LayerID])
						{
							NewX = B * StrideSchedule[LayerID] - PaddingSchedule[LayerID] + Y;
							NewY = C * StrideSchedule[LayerID] - PaddingSchedule[LayerID] + Z;

							if (NewX < InputDim && NewY < InputDim && NewX >= 0 && NewY >= 0)
							{
								FeatureIndex = InputOffset + X * InputDim * InputDim + NewX * InputDim + NewY;
								FilterIndex = TgtIndex + Y * FilterDims[LayerID] + Z;

								if (FilterIndex < Filters.Num() && FeatureIndex < Features.Num())
								{
									NetSum += Features[FeatureIndex] * Filters[FilterIndex];
								}

							}

							Z++;
						}

						Y++;
					}

					X++;
				}

				NetFeatures[FeatureOffset + A * TempMapDims * TempMapDims + B * TempMapDims + C] = NetSum + Bias[BiasOffset + A];

				C++;
			}

			B++;
		}

		A++;
	}
}

void ACNN::ActivationFunction(int LayerID)
{
	int A = 0;
	int B = 0;
	int C = 0;
	int NeuronIndex = 0;
	double TempVal = 0;

	while (A < MapCounts[LayerID])
	{
		B = 0;
		while (B < MapDims[LayerID])
		{
			C = 0;
			while (C < MapDims[LayerID])
			{
				NeuronIndex = FeatureStart(LayerID) + (A * (MapDims[LayerID] * MapDims[LayerID])) + (B * MapDims[LayerID]) + C;

				switch (ActivationFunctionSchedule[LayerID])
				{
				case 0:
					Features[NeuronIndex] = FMath::Tanh(NormFeatures[NeuronIndex]);  // Use Features
					break;
				case 1:
					Features[NeuronIndex] = 1.0 / (1.0 + FMath::Exp(-1.0 * NormFeatures[NeuronIndex]));  // Use Features
					break;
				case 2:
					if (NormFeatures[NeuronIndex] > 0)  // Use Features consistently
					{
						Features[NeuronIndex] = NormFeatures[NeuronIndex];
					}
					else
					{
						Features[NeuronIndex] = NeuralNet->ELUConst * (FMath::Exp(NormFeatures[NeuronIndex]) - 1.0);
					}
					break;
				case 3:
					if (NormFeatures[NeuronIndex] > 0)  // Use Features
					{
						Features[NeuronIndex] = NormFeatures[NeuronIndex];
					}
					else
					{
						Features[NeuronIndex] = NeuralNet->ReLUConst * NormFeatures[NeuronIndex];
					}
					break;
				default:
					Features[NeuronIndex] = NormFeatures[NeuronIndex];  // Use Features
					break;
				}

				C++;
			}

			B++;
		}

		A++;
	}
}

void ACNN::ActivationFunctionDerivative(int LayerID)
{
	int A = 0;
	int B = 0;
	int C = 0;
	int NeuronIndex = 0;
	double TempVal = 0;

	while (A < MapCounts[LayerID])
	{
		B = 0;
		while (B < MapDims[LayerID])
		{
			C = 0;
			while (C < MapDims[LayerID])
			{
				NeuronIndex = FeatureStart(LayerID) + (A * (MapDims[LayerID] * MapDims[LayerID])) + (B * MapDims[LayerID]) + C;

				switch (ActivationFunctionSchedule[LayerID])
				{
				case 0:
					TempVal = FMath::Tanh(NormFeatures[NeuronIndex]);  // Use Features
					Derivatives[NeuronIndex] = 1.0 - (TempVal * TempVal);
					break;
				case 1:
					TempVal = 1.0 / (1.0 + FMath::Exp(-1 * NormFeatures[NeuronIndex]));  // Use Features
					Derivatives[NeuronIndex] = TempVal * (1.0 - TempVal);
					break;
				case 2:
					if (NormFeatures[NeuronIndex] > 0)  // Use Features
					{
						Derivatives[NeuronIndex] = 0.5;
					}
					else
					{
						Derivatives[NeuronIndex] = NormFeatures[NeuronIndex] * 1.0;
					}
					break;
				case 3:
					if (NormFeatures[NeuronIndex] > 0)  // Use Features
					{
						Derivatives[NeuronIndex] = 1.0;
					}
					else
					{
						Derivatives[NeuronIndex] = NeuralNet->ReLUConst;
					}
					break;
				default:
					Derivatives[NeuronIndex] = 1.0;
					break;
				}

				C++;
			}

			B++;
		}

		A++;
	}

}

TArray<double> ACNN::Flatten2DArray(TArray<TArray<double>>& Input)
{
	TArray<double> RetVal;
	int A = 0;
	int B = 0;

	RetVal.SetNum(Input.Num() * Input[0].Num());

	while (A < Input.Num())
	{
		B = 0;
		while (B < Input[A].Num())
		{
			RetVal[A * Input[0].Num() + B] = Input[A][B];

			B++;
		}

		A++;
	}

	return RetVal;
}

int ACNN::FeatureStart(int LayerID)
{
	int RetVal = 0;
	int A = 0;

	while (A < LayerID)
	{
		RetVal += MapCounts[A] * MapDims[A] * MapDims[A];

		A++;
	}

	return RetVal;
}

int ACNN::BiasStart(int LayerID)
{
	int RetVal = 0;
	int A = 0;

	while (A < LayerID)
	{
		RetVal += MapCounts[A];

		A++;
	}

	return RetVal;
}

int ACNN::FilterStart(int LayerID)
{
	int RetVal = 0;
	int A = 0;

	while (A < LayerID)
	{
		if (A == 0)
		{
			RetVal += MapCounts[A] * FilterDims[A] * FilterDims[A];
		}
		else
		{
			RetVal += MapCounts[A] * MapCounts[A - 1] * FilterDims[A] * FilterDims[A];
		}


		A++;
	}

	return RetVal;
}

void ACNN::LayerNorm(int LayerID)
{
	int A = 0;
	int B = 0;
	int C = 0;
	double Mean = 0;
	double Var = 0;
	double InvStd = 0;
	int BiasOffset = BiasStart(LayerID);
	int FeatureOffset = FeatureStart(LayerID);
	int TgtIndex = 0;

	while (A < MapCounts[LayerID])
	{
		Mean = 0;
		B = 0;
		while (B < MapDims[LayerID])
		{
			C = 0;
			while (C < MapDims[LayerID])
			{
				Mean += NetFeatures[FeatureOffset + A * MapDims[LayerID] * MapDims[LayerID] + B * MapDims[LayerID] + C];

				C++;
			}

			B++;
		}

		NormMean[BiasOffset + A] = Mean / (MapDims[LayerID] * MapDims[LayerID]);

		Var = 0;
		B = 0;
		while (B < MapDims[LayerID])
		{
			C = 0;
			while (C < MapDims[LayerID])
			{
				Var += FMath::Pow(NetFeatures[FeatureOffset + A * MapDims[LayerID] * MapDims[LayerID] + B * MapDims[LayerID] + C] - NormMean[BiasOffset+A], 2.0);

				C++;
			}

			B++;
		}

		NormVar[BiasOffset + A] = Var / (MapDims[LayerID] * MapDims[LayerID]);

		InvStd = 1.0 / FMath::Sqrt(NormVar[BiasOffset + A] + 0.00000001);
		B = 0;
		while (B < MapDims[LayerID])
		{
			C = 0;
			while (C < MapDims[LayerID])
			{
				TgtIndex = FeatureOffset + A * MapDims[LayerID] * MapDims[LayerID] + B * MapDims[LayerID] + C;
				NormFeatures[TgtIndex] = (NetFeatures[TgtIndex] - NormMean[BiasOffset + A]) * InvStd;
				NormFeatures[TgtIndex] = NormFeatures[TgtIndex] * NormGamma[BiasOffset + A] + NormBeta[BiasOffset + A];

				C++;
			}

			B++;
		}

		A++;
	}
}

void ACNN::Backward(TArray<TArray<double>>& Inputs, TArray<double> Outputs)
{
	int A = 0;
	int B = 0;
	int C = 0;
	TArray<double> FlatNeurons;
	TArray<double> FlatFeatures;
	TArray<double> TransitionErrors;
	TArray<double> TempGradients;
	int LastFeatureStart = FeatureStart(MapCounts.Num() - 1);
	double NetSum = 0;

	Forward(Inputs);
	NeuralNet->Backward(TransitionActivations, Outputs);


	//Propagate loss gradient through transition layer
	FlatFeatures = NeuralNet->GetSlice(Features, LastFeatureStart, Features.Num() - LastFeatureStart);
	TransitionErrors.SetNumZeroed(FlatFeatures.Num());


	while (A < FlatFeatures.Num())
	{
		NetSum = 0;
		
		B = 0;
		while (B < TransitionActivations.Num())
		{
			NetSum += NeuralNet->Errors[B] * TransitionWeights[B * (Features.Num() - LastFeatureStart) + A];

			B++;
		}

		TransitionErrors[A] = NetSum * Derivatives[LastFeatureStart + A];

		A++;
	}

	A = 0;
	while (A < TransitionActivations.Num())
	{
		B = 0;
		while (B < FlatFeatures.Num())
		{
			TransitionDeltas[A * FlatFeatures.Num() + B] += FlatNeurons[A] * FlatFeatures[B];

			B++;
		}

		A++;
	}

	//Now through Conv layers
	A = MapCounts.Num() - 1;
	while (A >= 0)
	{
		if (A == MapCounts.Num() - 1)
		{
			TempGradients = LayerNormBackward(TransitionErrors, A);
		}
		else
		{
			TempGradients = LayerNormBackward(TempGradientsWRTInputs, A);
		}

		GradientsWRTOutputs(TempGradients, A);
		GradientsWRTFilters(A);
		TempGradientsWRTInputs = GradientsWRTInputs(A);

		A--;
	}

	if (Iteration % BatchSize == 0)
	{
		ApplyGradientDeltas();
		BatchIteration++;
	}

	Iteration++;
}

TArray<double> ACNN::LayerNormBackward(TArray<double>& Gradients, int LayerID)
{
	TArray<double> Outputs;
	TArray<double> XHats;
	int A = 0;
	int B = 0;
	double Mean = 0;
	double Var = 0;
	double StdDev = 0;
	int FeatureOffset = FeatureStart(LayerID);
	int BiasOffset = BiasStart(LayerID);
	int TempOffset = 0;
	double SumG = 0;
	double SumGXHat = 0;
	double GammaTemp = 0;
	double BetaTemp = 0;
	int MapSquared = MapDims[LayerID] * MapDims[LayerID];

	Outputs.SetNumZeroed(MapSquared * MapCounts[LayerID]);

	while (A < MapCounts[LayerID])
	{
		SumG = 0;
		SumGXHat = 0;
		GammaTemp = 0;
		BetaTemp = 0;
		TempOffset = FeatureOffset + A * MapSquared;
		Mean = NormMean[BiasOffset + A];
		Var = NormVar[BiasOffset + A];
		StdDev = FMath::Sqrt(Var + 0.00000001);
		XHats.SetNumZeroed(MapSquared);

		B = 0;
		while (B < MapSquared)
		{
			XHats[B] = (NetFeatures[TempOffset + B] - Mean) / StdDev;
			SumG += Gradients[TempOffset + B - FeatureOffset];
			SumGXHat += Gradients[TempOffset + B - FeatureOffset] * XHats[B];
			GammaTemp += Gradients[TempOffset + B - FeatureOffset] * XHats[B];
			BetaTemp += Gradients[TempOffset + B - FeatureOffset];

			B++;
		}

		B = 0;
		while (B < MapDims[LayerID] * MapDims[LayerID])
		{
			Outputs[TempOffset + B - FeatureOffset] = (NormGamma[BiasOffset + A] / (StdDev * MapSquared)) * (MapSquared * Gradients[TempOffset + B - FeatureOffset] - SumG - XHats[B] * SumGXHat);

			B++;
		}

		NormGammaDelta[BiasOffset + A] += GammaTemp / MapSquared;
		NormBetaDelta[BiasOffset + A] += BetaTemp / MapSquared;

		A++;
	}

	return Outputs;
}


TArray<double> ACNN::GradientsWRTOutputs(TArray<double>& Input1, int LayerID)
{
	int Start = FeatureStart(LayerID);
	int LayerSize = MapCounts[LayerID] * MapDims[LayerID] * MapDims[LayerID];
	TArray<double> OutputGrad;
	int ElementsToProcess = FMath::Min(Input1.Num(), LayerSize);
	
	OutputGrad.SetNumZeroed(LayerSize);

	// Critical bounds check - ensure we don't read beyond Input1
	

	for (int A = 0; A < ElementsToProcess; A++)
	{
		Errors[Start + A] = Input1[A] * Derivatives[Start + A];
	}

	// Remaining elements are already zeroed by SetNumZeroed
	return OutputGrad;
}

TArray<double> ACNN::GradientsWRTInputs(int LayerID)
{
	int A = 0;
	int B = 0;
	int C = 0;
	int X = 0;
	int Y = 0;
	int Z = 0;
	int InputOffset = 0;
	int OutputOffset = 0;
	int FilterOffset = 0;
	int FilterBase = 0;
	int BiasOffset = 0;
	double TempVal = 0;
	int InputCount = 0;
	int InputDims = 0;
	int NewX = 0;
	int NewY = 0;
	int FlippedB = 0;
	int FlippedC = 0;
	int GradientIndex = 0;
	int InputIndex = 0;
	int FilterIndex = 0;
	int PaddedY = 0;
	int PaddedZ = 0;
	TArray<double> RetVal;

	if (LayerID > 0)
	{
		InputOffset = FeatureStart(LayerID - 1);
		OutputOffset = FeatureStart(LayerID);
		FilterOffset = FilterStart(LayerID);
		BiasOffset = BiasStart(LayerID);
		InputCount = MapCounts[LayerID - 1];
		InputDims = MapDims[LayerID - 1];
	}
	else
	{
		InputCount = 1;
		InputDims = MapDims[0] * StrideSchedule[0];
	}

	RetVal.SetNumZeroed(InputCount * InputDims * InputDims);

	while (A < MapCounts[LayerID])
	{
		X = 0;
		while (X < InputCount)
		{
			B = 0;
			while (B < FilterDims[LayerID])
			{
				C = 0;
				while (C < FilterDims[LayerID])
				{
					FlippedB = FilterDims[LayerID] - 1 - B;
					FlippedC = FilterDims[LayerID] - 1 - C;
					TempVal = 0;

					Y = 0;
					while (Y < MapDims[LayerID])
					{
						Z = 0;
						while (Z < MapDims[LayerID])
						{
							PaddedY = Y * StrideSchedule[LayerID] + B - PaddingSchedule[LayerID];
							PaddedZ = Z * StrideSchedule[LayerID] + C - PaddingSchedule[LayerID];

							if (PaddedY >= 0 && PaddedY < InputDims && PaddedZ >= 0 && PaddedZ < InputDims)
							{
								GradientIndex = OutputOffset + A * (MapDims[LayerID] * MapDims[LayerID]) + Y * MapDims[LayerID] + Z;
								InputIndex = X * (InputDims * InputDims) + PaddedY * InputDims + PaddedZ;
								FilterIndex = FilterOffset + A * (InputCount * FilterDims[LayerID] * FilterDims[LayerID]) + X * (FilterDims[LayerID] * FilterDims[LayerID]) + FlippedB * FilterDims[LayerID] + FlippedC;

								if (InputIndex < RetVal.Num() && FilterIndex < Filters.Num())
								{
									RetVal[InputIndex] += Errors[GradientIndex] * Filters[FilterIndex];
								}

							}

							Z++;
						}

						Y++;
					}

					C++;
				}

				B++;
			}

			X++;
		}

		A++;
	}

	return RetVal;
}

void ACNN::GradientsWRTFilters(int LayerID)
{
	int A = 0;
	int B = 0;
	int C = 0;
	int X = 0;
	int Y = 0;
	int Z = 0;
	int InputOffset = 0;
	int OutputOffset = 0;
	int FilterOffset = 0;
	int FilterBase = 0;
	int BiasOffset = 0;
	double TempVal = 0;
	int InputCount = 0;
	int InputDims = 0;
	int NewX = 0;
	int NewY = 0;
	int GradientIndex = 0;
	int InputIndex = 0;
	int FilterIndex = 0;
	TArray<double> RetVal;

	if (LayerID > 0)
	{
		InputOffset = FeatureStart(LayerID - 1);
		OutputOffset = FeatureStart(LayerID);
		FilterOffset = FilterStart(LayerID);
		BiasOffset = BiasStart(LayerID);
		InputCount = MapCounts[LayerID - 1];
		InputDims = MapDims[LayerID - 1];
		RetVal.SetNum(FilterStart(LayerID) - FilterStart(LayerID - 1));
	}
	else
	{
		InputCount = 1;
		InputDims = ImageSizePx;
		RetVal.SetNum(FilterStart(LayerID + 1) - 1);
	}

	while (A < MapCounts[LayerID])
	{
		X = 0;
		while (X < InputCount)
		{
			B = 0;
			while (B < FilterDims[LayerID])
			{
				C = 0;
				while (C < FilterDims[LayerID])
				{
					TempVal = 0;

					Y = 0;
					while (Y < MapDims[LayerID])
					{
						Z = 0;
						while (Z < MapDims[LayerID])
						{
							GradientIndex = OutputOffset + A * (MapDims[LayerID] * MapDims[LayerID]) + Y * MapDims[LayerID] + Z;
							InputIndex = InputOffset + X * (InputDims * InputDims) + ((Y * StrideSchedule[LayerID] + B) - PaddingSchedule[LayerID]) * InputDims + ((Z * StrideSchedule[LayerID] + C) - PaddingSchedule[LayerID]);

							if (InputIndex >= 0)
							{
								TempVal += Errors[GradientIndex] * Features[InputIndex];
							}

							Z++;
						}

						Y++;
					}

					FilterIndex = FilterOffset + A * (InputCount * FilterDims[LayerID] * FilterDims[LayerID]) + X * (FilterDims[LayerID] * FilterDims[LayerID]) + B * FilterDims[LayerID] + C;

					if (FilterIndex < Deltas.Num())
					{
						Deltas[FilterIndex] += TempVal;
					}


					C++;
				}

				B++;
			}

			X++;
		}

		TempVal = 0;
		Y = 0;

		while (Y < MapDims[LayerID])
		{
			Z = 0;
			while (Z < MapDims[LayerID])
			{
				TempVal += Errors[OutputOffset + A * (MapDims[LayerID] * MapDims[LayerID]) + Y * MapDims[LayerID] + Z];

				Z++;
			}

			Y++;
		}

		BiasDelta[BiasOffset + A] += TempVal;

		A++;
	}
}

void ACNN::ApplyGradientDeltas()
{
	int A = 0;
	int B = 0;
	int C = 0;
	int D = 0;
	int E = 0;
	double B1Corrected = 0;
	double B2Corrected = 0;
	double TempVal = 0;

	while (A < Deltas.Num())
	{
		DeltaMean[A] = ((NeuralNet->AdamB1 * DeltaMean[A]) + ((1.0 - NeuralNet->AdamB1) * Deltas[A]));
		B1Corrected = DeltaMean[A] / (1.0 - FMath::Pow(NeuralNet->AdamB1, BatchIteration));

		DeltaVar[A] = ((NeuralNet->AdamB2 * DeltaVar[A]) + ((1.0 - NeuralNet->AdamB2) * FMath::Pow(Deltas[A], 2.0)));
		B2Corrected = DeltaVar[A] / (1.0 - FMath::Pow(NeuralNet->AdamB2, BatchIteration));

		TempVal = LearningRate * (B1Corrected / (FMath::Sqrt(B2Corrected) + 0.00000001));

		Filters[A] -= TempVal;
		Deltas[A] = 0;

		A++;
	}

	A = 0;
	while (A < BiasDelta.Num())
	{
		BiasMean[A] = ((NeuralNet->AdamB1 * BiasMean[A]) + ((1.0 - NeuralNet->AdamB1) * BiasDelta[A]));
		B1Corrected = BiasMean[A] / (1.0 - FMath::Pow(NeuralNet->AdamB1, BatchIteration));

		BiasVar[A] = ((NeuralNet->AdamB2 * BiasVar[A]) + ((1.0 - NeuralNet->AdamB2) * FMath::Pow(BiasDelta[A], 2.0)));
		B2Corrected = BiasVar[A] / (1.0 - FMath::Pow(NeuralNet->AdamB2, BatchIteration));

		TempVal = LearningRate * (B1Corrected / (FMath::Sqrt(B2Corrected) + 0.00000001));

		Bias[A] -= TempVal;
		BiasDelta[A] = 0;

		A++;
	}

	A = 0;
	while (A < NormGammaDelta.Num())
	{
		NormGammaMean[A] = ((NeuralNet->AdamB1 * NormGammaMean[A]) + ((1.0 - NeuralNet->AdamB1) * NormGammaDelta[A]));
		B1Corrected = NormGammaMean[A] / (1.0 - FMath::Pow(NeuralNet->AdamB1, BatchIteration));

		NormGammaVar[A] = ((NeuralNet->AdamB2 * NormGammaVar[A]) + ((1.0 - NeuralNet->AdamB2) * FMath::Pow(NormGammaDelta[A], 2.0)));
		B2Corrected = NormGammaVar[A] / (1.0 - FMath::Pow(NeuralNet->AdamB2, BatchIteration));

		TempVal = LearningRate * (B1Corrected / (FMath::Sqrt(B2Corrected) + 0.00000001));

		NormGamma[A] -= TempVal;
		NormGammaDelta[A] = 0;

		NormBetaMean[A] = ((NeuralNet->AdamB1 * NormBetaMean[A]) + ((1.0 - NeuralNet->AdamB1) * NormBetaDelta[A]));
		B1Corrected = NormBetaMean[A] / (1.0 - FMath::Pow(NeuralNet->AdamB1, BatchIteration));

		NormBetaVar[A] = ((NeuralNet->AdamB2 * NormBetaVar[A]) + ((1.0 - NeuralNet->AdamB2) * FMath::Pow(NormBetaDelta[A], 2.0)));
		B2Corrected = NormBetaVar[A] / (1.0 - FMath::Pow(NeuralNet->AdamB2, BatchIteration));

		TempVal = LearningRate * (B1Corrected / (FMath::Sqrt(B2Corrected) + 0.00000001));

		NormBeta[A] -= TempVal;
		NormBetaDelta[A] = 0;

		A++;
	}

	B = 0;
	while (B < TransitionDeltas.Num())
	{
		TransitionDeltasMean[B] = ((NeuralNet->AdamB1 * TransitionDeltasMean[B]) + ((1.0 - NeuralNet->AdamB1) * TransitionDeltas[B]));
		B1Corrected = TransitionDeltasMean[B] / (1.0 - FMath::Pow(NeuralNet->AdamB1, BatchIteration));

		TransitionDeltasVar[B] = ((NeuralNet->AdamB2 * TransitionDeltasVar[B]) + ((1.0 - NeuralNet->AdamB2) * FMath::Pow(TransitionDeltas[B], 2.0)));
		B2Corrected = TransitionDeltasVar[B] / (1.0 - FMath::Pow(NeuralNet->AdamB2, BatchIteration));

		TempVal = NeuralNet->LearningRate * (B1Corrected / (FMath::Sqrt(B2Corrected) + 0.00000001));

		TransitionWeights[B] -= TempVal;
		TransitionDeltas[B] = 0;

		B++;
	}
}
