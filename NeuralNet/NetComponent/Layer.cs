using System;
using System.Collections.Generic;

namespace NeuralNet.NetComponent
{
    public class Layer2 : TrainableComponent
    {
        #region static
        public static Layer2 CreateLinearLayer(NetworkMatrix weights, NetworkVector biases)
        {
            return new Layer2(weights, biases, null, null);
        }

        public static Layer2 CreateLinearLayer(NetworkMatrix weights)
        {
            return new Layer2(weights, new NetworkVector(weights.NumberOfOutputs), null, null);
        }


        public static Layer2 CreateLogisticLayer(NetworkMatrix weights, NetworkVector biases)
        {
            return new Layer2(weights, biases, NeuralFunction.__Logistic, NeuralFunction.__LogisticDerivative);
        }

        public static Layer2 CreateLogisticLayer(NetworkMatrix weights)
        {
            return new Layer2(weights, new NetworkVector(weights.NumberOfOutputs), NeuralFunction.__Logistic, NeuralFunction.__LogisticDerivative);
        }
        #endregion


        #region protectred attributes
        protected WeightedCombiner _combiner;
        protected NeuralFunction _neuralFunction;
        #endregion

        
        #region NetComponent overrides
        public override int NumberOfInputs { get { return _combiner.NumberOfInputs; } }
        public override int NumberOfOutputs { get { return _combiner.NumberOfOutputs; } }
        public override NetworkVector Output
        {
            get
            {
                if (_neuralFunction == null)
                    return _combiner.Output;

                return _neuralFunction.Output;
            }

            protected set { Output = value; }
        }

        public override NetworkVector InputGradient(NetworkVector outputgradient)
        {
            return _combiner.InputGradient(_getActivationGradient(outputgradient));
        }
        
        public override void Run(NetworkVector inputvalues)
        {
            if (inputvalues.Dimension != NumberOfInputs)
                throw new ArgumentException("Input dimension does not match this Layer.");
            _combiner.Run(inputvalues);
            if (_neuralFunction != null)
                _neuralFunction.Run(_combiner.Output);
        }
        #endregion

        #region TrainableComponent overrides
        public override NetworkMatrix Weights { get { return _combiner.Weights; } }
        public override NetworkVector Biases { get { return _combiner.Biases; } }

        public override NetworkVector BiasesGradient(NetworkVector outputgradient)
        {
            return _combiner.BiasesGradient(_getActivationGradient(outputgradient));
        }

        public override NetworkMatrix WeightsGradient(NetworkVector outputgradient)
        {
            return _combiner.WeightsGradient(_getActivationGradient(outputgradient));
        }

        public override void Update(NetworkVector biasesdelta, NetworkMatrix weightsdelta)
        {
            _combiner.Update(biasesdelta, weightsdelta);
        }
        #endregion
                

        #region constructors
        public Layer2 (
            NetworkMatrix weights,
            NetworkVector biases,
            ActivationFunction activationfunction, 
            DerivativeFunction derivativefunction
            )
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
                _neuralFunction = new NeuralFunction(NumberOfOutputs, activationfunction, derivativefunction);
            }
        }

        public Layer2(NetworkMatrix weights, NetworkVector biases)
            : this (weights, biases, null, null)
        { }
        
        public Layer2(NetworkMatrix weights)
            : this(weights, new NetworkVector(weights.NumberOfOutputs), null, null)
        { }
        #endregion


        #region private methods
        private NetworkVector _getActivationGradient(NetworkVector outputgradient)
        {
            if (_neuralFunction == null)
                return outputgradient;
            return _neuralFunction.InputGradient(outputgradient);
        }
        #endregion

    }
}
