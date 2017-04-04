using System;
using System.Collections.Generic;

namespace NeuralNet
{
    public class Layer : TrainableComponent
    {
        #region static
        public static Layer CreateLinearLayer(WeightsMatrix weights, NetworkVector biases)
        {
            return new Layer(weights, biases, null, null);
        }

        public static Layer CreateLinearLayer(WeightsMatrix weights)
        {
            return new Layer(weights, new NetworkVector(weights.NumberOfOutputs), null, null);
        }


        public static Layer CreateLogisticLayer(WeightsMatrix weights, NetworkVector biases)
        {
            return new Layer(weights, biases, NeuralFunction.__Logistic, NeuralFunction.__LogisticDerivative);
        }

        public static Layer CreateLogisticLayer(WeightsMatrix weights)
        {
            return new Layer(weights, new NetworkVector(weights.NumberOfOutputs), NeuralFunction.__Logistic, NeuralFunction.__LogisticDerivative);
        }
        #endregion


        #region protectred attributes
        protected WeightedCombiner _combiner;
        protected NeuralFunction _neuralFunction;
        #endregion

        
        #region NetComponent overrides
        public override int NumberOfInputs { get { return _combiner.NumberOfInputs; } }
        public override int NumberOfOutputs { get { return _combiner.NumberOfOutputs; } }
        
        public override NetworkVector InputGradient(NetworkVector outputgradient)
        {
            return _combiner.InputGradient(ActivationGradient(outputgradient));
        }

        public override VectorBatch InputGradient(VectorBatch outputgradients)
        {
            return _combiner.InputGradient(ActivationGradient(outputgradients));
        }
        #endregion

        #region TrainableComponent overrides
        public override WeightsMatrix Weights { get { return _combiner.Weights; } }
        public override NetworkVector Biases { get { return _combiner.Biases; } }

        public override NetworkVector BiasesGradient(NetworkVector outputgradient)
        {
            return _combiner.BiasesGradient(ActivationGradient(outputgradient));
        }

        public NetworkVector BiasesGradient(VectorBatch outputgradient)
        {
            return _combiner.BiasesGradient(ActivationGradient(outputgradient));
        }

        public override WeightsMatrix WeightsGradient(NetworkVector outputgradient)
        {
            return _combiner.WeightsGradient(ActivationGradient(outputgradient));
        }

        public WeightsMatrix WeightsGradient(NetworkVector outputgradient, NetworkVector input)
        {
            return _combiner.WeightsGradient(ActivationGradient(outputgradient), input);
        }

        public WeightsMatrix WeightsGradient(VectorBatch outputgradient, VectorBatch input)
        {
            return _combiner.WeightsGradient(ActivationGradient(outputgradient), input);
        }

        public override void Update(AdaptationStrategy strategy)
        {
            Biases.Add(strategy.BiasesUpdate( _biasesGradientAccumulator));
            Weights.Add(strategy.WeightsUpdate( _weightsGradientAccumulator));
            _biasesGradientAccumulator.Zero();
            _weightsGradientAccumulator.Zero();
        }

        public override void BackPropagate(NetworkVector outputgradient)
        {
            BackPropagate(outputgradient, VectorInput);
        }

        public override void BackPropagate(NetworkVector outputgradient, NetworkVector input)
        {
            _biasesGradientAccumulator.Add(BiasesGradient(outputgradient));
            _weightsGradientAccumulator.Add(WeightsGradient(outputgradient, input));
        }

        public override void BackPropagate(VectorBatch outputgradient)
        {
            BackPropagate(outputgradient, BatchInput);
        }

        public override void BackPropagate(VectorBatch outputgradient, VectorBatch input)
        {
            _biasesGradientAccumulator.Add(BiasesGradient(outputgradient));
            _weightsGradientAccumulator.Add(WeightsGradient(outputgradient, input));
        }


        protected override NetworkVector _run(NetworkVector inputvalues)
        {
            if (inputvalues.Dimension != NumberOfInputs)
                throw new ArgumentException("Input dimension does not match this Layer.");

            NetworkVector result = _combiner.Run(inputvalues);
            if (_neuralFunction != null)
                result = _neuralFunction.Run(result);

            return result;
        }

        protected override VectorBatch _run(VectorBatch inputbatch)
        {
            if (inputbatch.Dimension != NumberOfInputs)
                throw new ArgumentException("Input dimension does not match this Layer.");

            VectorBatch result = _combiner.Run(inputbatch);
            if (_neuralFunction != null)
                result = _neuralFunction.Run(result);

            return result;
        }
        #endregion


        #region constructors
        public Layer (
            WeightsMatrix weights,
            NetworkVector biases,
            ActivationFunction activationfunction, 
            DerivativeFunction derivativefunction
            )
            : base (weights.NumberOfOutputs, weights.NumberOfInputs)
        {
            if (activationfunction != null && derivativefunction == null)
                throw new ArgumentException("derivativefunction cannot be null, if activatioin is not null");

            if (weights == null || biases == null)
                throw new ArgumentException("Attempt to make a layer with null weights or biases.");

            _combiner = new WeightedCombiner(weights, biases);

            if (activationfunction == null)
            {
                _neuralFunction = null;
            }
            else
            {
                _neuralFunction = new NeuralFunction(_combiner.NumberOfOutputs, activationfunction, derivativefunction);
            }
        }

        public Layer(WeightsMatrix weights, NetworkVector biases)
            : this (weights, biases, null, null)
        { }
        
        public Layer(WeightsMatrix weights)
            : this(weights, new NetworkVector(weights.NumberOfOutputs), null, null)
        { }
        #endregion


        #region public methods
        public NetworkVector ActivationGradient(NetworkVector outputgradient)
        {
            if (_neuralFunction == null)
                return outputgradient;
            return _neuralFunction.InputGradient(outputgradient);
        }

        public VectorBatch ActivationGradient(VectorBatch outputgradient)
        {
            if (_neuralFunction == null)
                return outputgradient;
            return _neuralFunction.InputGradient(outputgradient);
        }
        #endregion

    }
}
