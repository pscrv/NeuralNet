using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;

namespace NeuralNet2
{
    public class WeightedCombiner : TrainableComponent
    {
        #region protected attributes
        #endregion


        #region Constructors
        public WeightedCombiner(WeightsMatrix weights, BiasesVector biases)
            : base (weights, biases)
        { }
        
        public WeightedCombiner(WeightsMatrix weights)
            : this (weights, new BiasesVector(weights.NumberOfOutputs)) { }
        #endregion


        #region TrainableComponent overrides
        protected override VectorBatch _trainingRun(VectorBatch input)
        {
            _input = input;

            VectorBatch combination = _weights.ApplyForwards(input);
            VectorBatch output = combination.AddToEachVector(_biases);
            return output;
        }

        protected override VectorBatch _getInputGradient(VectorBatch outputGradient)
        {
            return _weights.ApplyBackwards(outputGradient);
        }

        protected override void _updateBiases(VectorBatch outputGradient)
        {
            BiasesVector biasesGradient = new BiasesVector(outputGradient.SumColumnsAsMatrix());
            _biases = _biases.Add(Strategy.BiasesUpdate(biasesGradient));
        }

        protected override void _updateWeights(VectorBatch outputGradient)
        {
            WeightsMatrix weightsGradient = WeightsMatrix.FromVectorBatchPair(_input, outputGradient);
            _weights = _weights.Add(Strategy.WeightsUpdate(weightsGradient));
        }
               #endregion
    }
}



