// Fill out your copyright notice in the Description page of Project Settings.


#include "ANN.h"

ANN::ANN()
{
}

ANN::~ANN()
{
}

void ANN::Initialize(TArray<int> LayerDims, TArray<int> ActivationIDs, double LearnRate)
{
	int A = 0;
	double TempVal = 0;
	int NeuronTotal = 0;
	int WeightTotal = 0;

	this->LearningRate = LearnRate;
	this->ActivationFunctionIDs = ActivationIDs;
	this->LayerCounts = LayerDims;

	WeightTotal = 0;
	
	//1 since input layer has no weights
	A = 1;
	while (A < LayerDims.Num())
	{
		WeightTotal += LayerDims[A] * LayerDims[A - 1];

		A++;
	}

	//Use SetNum() instead of direct .Add since that redims array each time, would scale poorly
	this->Weights.SetNum(WeightTotal);
	this->WeightDeltas.SetNum(WeightTotal);
	this->DeltaMean.SetNum(WeightTotal);
	this->DeltaVar.SetNum(WeightTotal);

	NeuronTotal = 0;
	A = 0;
	while (A < LayerDims.Num())
	{
		NeuronTotal += LayerDims[A];

		A++;
	}

	this->NetInputs.SetNum(NeuronTotal);
	this->Activations.SetNum(NeuronTotal);
	this->Derivatives.SetNum(NeuronTotal);
	this->Errors.SetNum(NeuronTotal);
	this->Bias.SetNum(NeuronTotal);
	this->BiasDelta.SetNum(NeuronTotal);
	this->BiasMean.SetNum(NeuronTotal);
	this->BiasVar.SetNum(NeuronTotal);

	A = 0;
	while (A < WeightTotal)
	{
		//Weights initialize later
		this->Weights[A] = 0;
		this->WeightDeltas[A] = 0;
		this->DeltaMean[A] = 0;
		this->DeltaVar[A] = 0;

		A++;
	}

	A = 0;
	while (A < NeuronTotal)
	{
		this->NetInputs[A] = 0;
		this->Activations[A] = 0;
		this->Derivatives[A] = 0;
		this->Errors[A] = 0;
		this->Bias[A] = 0;
		this->BiasDelta[A] = 0;
		this->BiasMean[A] = 0;
		this->BiasVar[A] = 0;

		A++;
	}

	this->InitWeights();
}

void ANN::InitWeights()
{
	int A = 1;
	int B = 0;
	int C = 0;
	double StdDev = 0;
	int Offset = 0;

	while (A < LayerCounts.Num())
	{
		StdDev = FMath::Sqrt(2.0 / LayerCounts[A - 1]);
		Offset = WeightStart(A);

		B = 0;
		while (B < LayerCounts[A])
		{
			C = 0;
			while (C < LayerCounts[A - 1])
			{
				Weights[Offset + B * LayerCounts[A - 1] + C] = FMath::FRandRange(-StdDev, StdDev);

				C++;
			}

			B++;
		}

		A++;
	}

}

int ANN::WeightStart(int LayerID)
{
	int Offset = 0;
	int A = 1;

	while (A < LayerID)
	{
		Offset += LayerCounts[A] * LayerCounts[A - 1];

		A++;
	}

	return Offset;
}

int ANN::NeuronStart(int LayerID)
{
	int Offset = 0;
	int A = 0;

	while(A < LayerID)
	{
		Offset += LayerCounts[A];  // sum all neurons in previous layers
	}

	return Offset;
}

double ANN::ActivationFunction(double Input, int ActivationFunctionID)
{
	double Output = 0;
	
	switch (ActivationFunctionID)
	{
		case 0:
			Output = FMath::Tanh(Input);
			break;
		case 1:
			Output = 1.0 / (1.0 + FMath::Exp(-Input));
			break;
		case 2:
			if (Input > 0)
			{
				Output = Input;
			}
			else
			{
				Output = ELUConst * (FMath::Exp(Input) - 1.0);
			}
			break;
		case 3:
			if (Input > 0)
			{
				Output = Input;
			}
			else
			{
				Output = ReLUConst * Input;
			}
			break;
		default:
			Output = Input;
			break;
	}

	return Output;
}

double ANN::ActivationFunctionDerivative(double Input, int ActivationFunctionID)
{
	double Output = 0;
	
	switch (ActivationFunctionID)
	{
		case 0:
			Output = FMath::Tanh(Input);
			Output = 1.0 - (Input * Input);
			break;
		case 1:
			Output = 1.0 / (1.0 + FMath::Exp(-Input));
			Output = Output * (1.0 - Output);
			break;
		case 2:
			if (Input > 0)
			{
				Output = 1.0;
			}
			else
			{
				Output = Input * ELUConst;
			}
			break;
		case 3:
			if (Input > 0)
			{
				Output = 1.0;
			}
			else
			{
				Output = ReLUConst;
			}
			break;
		default:
			Output = 1.0;
			break;
	}

	return Output;
}

TArray<double> ANN::GetSlice(TArray<double>& TargetList, int StartIndex, int Length)
{
	TArray<double> RetVal;
	int A = StartIndex;

	RetVal.SetNum(Length - StartIndex);

	while (A < StartIndex + Length)
	{
		RetVal[A] = TargetList[A];

		A++;
	}

	return RetVal;
}

void ANN::Forward(TArray<double>& Inputs)
{
	int A = 0;
	int B = 0;
	int C = 0;
	int NeuronOffset = 0;
	int LayerOffset = 0;
	int PrevLayerOffset = 0;
	int TgtIndex = 0;
	double NetSum = 0;

	//Temporarily store inputs in the beginning of the feature buffers
	while (A < LayerCounts[0])
	{
		NetInputs[A] = Inputs[A];
		Activations[A] = ActivationFunction(NetInputs[A], ActivationFunctionIDs[0]);
		Derivatives[A] = ActivationFunctionDerivative(Activations[A], ActivationFunctionIDs[0]);

		A++;
	}

	A = 1;
	while(A < LayerCounts.Num())
	{
		LayerOffset = NeuronStart(A);
		PrevLayerOffset = NeuronStart(A - 1);

		B = 0;
		while (B < LayerCounts[A])
		{
			NetSum = 0;
			
			C = 0;
			while (C < LayerCounts[A - 1])
			{
				NetSum += Weights[WeightStart(A) + B * LayerCounts[A - 1] + C] * Activations[PrevLayerOffset + C];

				C++;
			}

			TgtIndex = LayerOffset + B;

			NetInputs[TgtIndex] = NetSum + Bias[TgtIndex];
			Activations[TgtIndex] = ActivationFunction(NetInputs[TgtIndex], ActivationFunctionIDs[A]);
			Derivatives[TgtIndex] = ActivationFunctionDerivative(Activations[A], ActivationFunctionIDs[A]);

			B++;
		}

		A++;
	}
}

void ANN::Backward(TArray<double>& Inputs, TArray<double>& Outputs)
{
	int A = 0;
	int B = 0;
	int C = 0;
	int D = 0;
	double NetSum = 0;
	int Offset = 0;
	int WeightOffset = 0;
	int PrevWeightOffset = 0;
	int NextWeightOffset = 0;
	int LayerOffset = 0;
	int NextLayerOffset = 0;
	int PrevLayerOffset = 0;
	int TgtIndex = 0;
	int TgtIndex2 = 0;

	Forward(Inputs);

	Offset = NeuronStart(LayerCounts.Num()-1);

	//YOLO Output Head
	while (A < LayerCounts.Last())
	{
		TgtIndex = Offset + A;

		if (TgtIndex + 4 >= Activations.Num() || A + 4 >= Outputs.Num())
		{
			continue;
		}

		if (Outputs[A] == 1.0)
		{
			Errors[TgtIndex] = 1.0 * (Activations[TgtIndex] - Outputs[A]) * Derivatives[TgtIndex];
			Errors[TgtIndex + 1] = 1.0 * (Activations[TgtIndex + 1] - Outputs[A+1]) * Derivatives[TgtIndex + 1];
			Errors[TgtIndex + 2] = 1.0 * (Activations[TgtIndex + 2] - Outputs[A+2]) * Derivatives[TgtIndex + 2];
			Errors[TgtIndex + 3] = 1.0 * (Activations[TgtIndex + 3] - Outputs[A + 3]) * Derivatives[TgtIndex + 3];
			Errors[TgtIndex + 4] = 1.0 * (Activations[TgtIndex + 4] - Outputs[A + 4]) * Derivatives[TgtIndex + 4];

			Bias[TgtIndex] += Errors[TgtIndex];
			Bias[TgtIndex + 1] += Errors[TgtIndex + 1];
			Bias[TgtIndex + 2] += Errors[TgtIndex + 2];
			Bias[TgtIndex + 3] += Errors[TgtIndex + 3];
			Bias[TgtIndex + 4] += Errors[TgtIndex + 4];
		}
		else
		{
			Errors[TgtIndex] = 0.5 * Activations[TgtIndex] * Derivatives[TgtIndex];
			Errors[TgtIndex + 1] = 0.01 * (Activations[TgtIndex + 1] - 0.5) * Derivatives[TgtIndex + 1];
			Errors[TgtIndex + 2] = 0.01 * (Activations[TgtIndex + 2] - 0.5) * Derivatives[TgtIndex + 2];
			Errors[TgtIndex + 3] = 0.01 * (Activations[TgtIndex + 3] - 0.1) * Derivatives[TgtIndex + 3];
			Errors[TgtIndex + 4] = 0.01 * (Activations[TgtIndex + 4] - 0.1) * Derivatives[TgtIndex + 4];

			Bias[TgtIndex] += Errors[TgtIndex];
			Bias[TgtIndex + 1] += Errors[TgtIndex + 1];
			Bias[TgtIndex + 2] += Errors[TgtIndex + 2];
			Bias[TgtIndex + 3] += Errors[TgtIndex + 3];
			Bias[TgtIndex + 4] += Errors[TgtIndex + 4];
		}

		//Gather weight deltas if more than one layer, one layer may typically be used as a shallow head for sister CNN class
		if (LayerCounts.Num() > 1)
		{
			WeightOffset = WeightStart(LayerCounts.Num() - 1);

			B = 0;
			while (B < LayerCounts[LayerCounts.Num() - 2])
			{
				C = 0;
				while (C < 5) //5 for YOLO block size
				{
					TgtIndex2 = WeightOffset + (A + C) * LayerCounts[LayerCounts.Num() - 2] + B;

					if (TgtIndex2 < WeightDeltas.Num() && Offset + B < Activations.Num())
					{
						WeightDeltas[TgtIndex2] += Errors[A + C] * Activations[Offset + B];
					}

					C++;
				}

				B++;
			}
		}

		A += 5;
	}

	//Propagate through hidden layers
	A = LayerCounts.Num() - 2;

	while (A >= 0)
	{
		LayerOffset = NeuronStart(A);
		PrevLayerOffset = NeuronStart(A - 1);
		NextLayerOffset = NeuronStart(A + 1);
		NextWeightOffset = WeightStart(A + 1);
		WeightOffset = WeightStart(A);

		B = 0;
		while (B < LayerCounts[A])
		{
			NetSum = 0.0;

			C = 0;
			while (C < LayerCounts[A + 1])
			{
				TgtIndex = NextWeightOffset + C * LayerCounts[A] + B;
				NetSum += Errors[NextLayerOffset + C] * Weights[TgtIndex];

				C++;
			}

			Errors[LayerOffset + B] = NetSum * Derivatives[LayerOffset + B];
			BiasDelta[LayerOffset + B] = Errors[LayerOffset + B];

			if (A > 0)
			{			
				C = 0;
				while (C < LayerCounts[A])
				{
					D = 0;
					while (D < LayerCounts[A - 1])
					{
						TgtIndex = WeightOffset + C * LayerCounts[A - 1] + D;

						WeightDeltas[TgtIndex] += Errors[LayerOffset + C] * Activations[PrevLayerOffset + D];

						D++;
					}

					C++;
				}
			}

			B++;
		}

		A--;
	}


	if (Iteration % BatchSize == 0)
	{
		ApplyGradientDeltas();

		BatchIteration++;
	}

	Iteration++;
}

void ANN::ApplyGradientDeltas()
{
	double B1Corrected = 0;
	double B2Corrected = 0;
	double TempVal = 0;
	int LayerOffset = 0;
	int NeuronOffset = 0;
	int TgtIndex = 0;
	int A = 1;
	int B = 0;
	int C = 0;

	// Loop over layers starting from first hidden layer
	while(A < LayerCounts.Num())
	{
		LayerOffset = WeightStart(A);
		NeuronOffset = NeuronStart(A);

		B = 0;
		while(B < LayerCounts[A])
		{
			C = 0;
			while(C < LayerCounts[A - 1])
			{
				TgtIndex = LayerOffset + B * LayerCounts[A - 1] + C;

				// Update moving averages for Adam-like optimizer
				DeltaMean[TgtIndex] = (AdamB1 * DeltaMean[TgtIndex]) + ((1.0 - AdamB1) * WeightDeltas[TgtIndex]);
				B1Corrected = DeltaMean[TgtIndex] / (1.0 - FMath::Pow(AdamB1, BatchIteration));

				DeltaVar[TgtIndex] = (AdamB2 * DeltaVar[TgtIndex]) + ((1.0 - AdamB2) * FMath::Pow(WeightDeltas[TgtIndex], 2));
				B2Corrected = DeltaVar[TgtIndex] / (1.0 - FMath::Pow(AdamB2, BatchIteration));

				TempVal = LearningRate * (B1Corrected / (FMath::Sqrt(B2Corrected) + 1e-8));

				// Apply gradient
				Weights[TgtIndex] -= TempVal;

				// Clear delta
				WeightDeltas[TgtIndex] = 0;

				C++;
			}

			// Update bias for this neuron
			TgtIndex = NeuronOffset + B;  // flat index for bias (can also have a separate BiasStart helper)
			BiasMean[TgtIndex] = (AdamB1 * BiasMean[TgtIndex]) + ((1.0 - AdamB1) * BiasDelta[TgtIndex]);
			B1Corrected = BiasMean[TgtIndex] / (1.0 - FMath::Pow(AdamB1, BatchIteration));

			BiasVar[TgtIndex] = (AdamB2 * BiasVar[TgtIndex]) + ((1.0 - AdamB2) * FMath::Pow(BiasDelta[TgtIndex], 2));
			B2Corrected = BiasVar[TgtIndex] / (1.0 - FMath::Pow(AdamB2, BatchIteration));

			TempVal = LearningRate * (B1Corrected / (FMath::Sqrt(B2Corrected) + 1e-8));

			Bias[TgtIndex] -= TempVal;

			BiasDelta[TgtIndex] = 0;

			B++;
		}

		A++;
	}
}
