using System;
using System.Collections.Generic;

namespace NeuralNet
{
    public class Layer : NetworkComponent
    {
        #region protectred attributes
        protected WeightedCombiner _combiner;
        protected NeuralFunction _neuralFunction;
        #endregion

        #region public properties
        public override int NumberOfInputs { get { return _combiner.NumberOfInputs; } }
        public override int NumberOfOutputs { get { return _combiner.NumberOfOutputs; } }
        public override NetworkVector InputGradient { get { return _combiner.InputGradient; } }
        public override NetworkVector Output
        {
            get
            {
                if (_neuralFunction == null)
                    return _combiner.Output;

                return _neuralFunction.Output;
            }
        }
        public LayerState State { get { return  _combiner.State; } }
        #endregion


        #region constructors
        public Layer (double[,] weights, double[] biases = null)
        {
            int numberOfOutputs = weights.GetLength(0);
            int numberOfInputs = weights.GetLength(1);

            _combiner = new WeightedCombiner(weights, biases);
            _neuralFunction = null;
        }

        public Layer (double[,] weights, ActivationFunction activationfunction, DerivativeFunction derivativefunction, double[] biases = null)
            : this (weights, biases)
        {
            if (activationfunction != null && derivativefunction == null)
                throw new ArgumentException("derivativefunction cannot be null, if activatioin is not null");

            _neuralFunction = new NeuralFunction(NumberOfOutputs, activationfunction, derivativefunction);
        }

        protected Layer(Layer layer)
        {
            if (layer == null)
                throw new ArgumentException("Attempt to create a LayerBank with a null layer.");

            _combiner = layer._combiner;
            _neuralFunction = layer._neuralFunction;
        }
        #endregion


        #region public methods
        public override void Run(NetworkVector inputvalues)
        {
            if (inputvalues.Dimension != NumberOfInputs)
                throw new ArgumentException("Input dimension does not match this Layer.");
            _combiner.Run(inputvalues);
            if (_neuralFunction != null)
                _neuralFunction.Run(_combiner.Output);
        }

        public override void BackPropagate(NetworkVector outputgradient)
        {
            if (_neuralFunction == null)
            {
                _combiner.BackPropagate(outputgradient);
            }
            else
            {
                _neuralFunction.BackPropagate(outputgradient);
                _combiner.BackPropagate(_neuralFunction.InputGradient);
            }
        }
        #endregion
    }
}
