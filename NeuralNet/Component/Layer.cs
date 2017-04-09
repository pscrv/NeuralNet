using System;
using System.Collections.Generic;

namespace NeuralNet2
{
    public class Layer : WeightedCombiner
    {
        #region static
        public static Layer CreateLinearLayer(WeightsMatrix weights, BiasesVector biases)
        {
            return new Layer(weights, biases, null, null);
        }

        public static Layer CreateLinearLayer(WeightsMatrix weights)
        {
            return new Layer(weights, new BiasesVector(weights.NumberOfOutputs), null, null);
        }


        public static Layer CreateLogisticLayer(WeightsMatrix weights, BiasesVector biases)
        {
            return new Layer(weights, biases, NeuralFunction.__Logistic, NeuralFunction.__LogisticDerivative);
        }

        public static Layer CreateLogisticLayer(WeightsMatrix weights)
        {
            return new Layer(weights, new BiasesVector(weights.NumberOfOutputs), NeuralFunction.__Logistic, NeuralFunction.__LogisticDerivative);
        }
        #endregion


        #region protectred attributes
        protected NeuralFunction _neuralFunction;
        protected VectorBatch _activationGradient;
        #endregion


        #region constructors
        public Layer (
            WeightsMatrix weights,
            BiasesVector biases,
            ActivationFunction activationfunction, 
            DerivativeFunction derivativefunction
            )
            : base (weights, biases)
        {
            if (weights == null || biases == null)
                throw new ArgumentException("Attempt to make a layer with null weights or biases.");
            

            if (activationfunction == null)
            {
                _neuralFunction = null;
            }
            else
            {
                _neuralFunction = new NeuralFunction(NumberOfOutputs, activationfunction, derivativefunction);
            }
        }

        public Layer(WeightsMatrix weights, BiasesVector biases)
            : this (weights, biases, null, null)
        { }
        
        public Layer(WeightsMatrix weights)
            : this(weights, new BiasesVector(weights.NumberOfOutputs), null, null)
        { }
        #endregion


        #region WeightedCombiner overrides
        protected override VectorBatch _backPropagate(VectorBatch outputGradient)
        {           
            _activationGradient = outputGradient;
            if (_neuralFunction != null)
                _activationGradient = _neuralFunction.BackPropagate(outputGradient);
            return base._backPropagate(outputGradient);
        }

        protected override VectorBatch _trainingRun(VectorBatch input)
        {
            VectorBatch output = base._trainingRun(input);
            if (_neuralFunction != null)
            {
                output = _neuralFunction.Run(output);
            }
            return output;
        }

        protected override VectorBatch _getInputGradient(VectorBatch outputGradient)
        {
            return base._getInputGradient(_activationGradient);
        }

        protected override void _updateBiases(VectorBatch outputGradient)
        {
            BiasesVector biasesGradient = new BiasesVector(_activationGradient.SumColumnsAsMatrix());
            _biases = _biases.Add(Strategy.BiasesUpdate(biasesGradient));
        }

        protected override void _updateWeights(VectorBatch outputGradient)
        {
            WeightsMatrix weightsGradient = WeightsMatrix.FromVectorBatchPair(_input, _activationGradient);
            _weights = _weights.Add(Strategy.WeightsUpdate(weightsGradient));
        }
        #endregion




    }
}
