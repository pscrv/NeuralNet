using System;
using System.Collections.Generic;

namespace NeuralNet
{



    public class Layer : LayerComponent
    {
        #region delegates
        public delegate double ActivationFunction(double summedInput);
        public delegate double DerivativeFunction(double input, double output);
        #endregion

        #region private attributes
        NetworkVector _inputs;
        NetworkMatrix _weights;
        NetworkVector _biases;
        NetworkVector _activations;
        NetworkVector _outputs;
        NetworkVector _inputGradient;
        protected ActivationFunction _neuralFunction;
        protected DerivativeFunction _neuralFunctionDerivative;
        #endregion

        #region public properties
        public override int NumberOfInputs { get { return _weights.NumberOfInputs; } }
        public override int NumberOfOutputs { get { return _weights.NumberOfNeurons; } }
        public override NetworkVector InputGradient { get { return _inputGradient; } }
        public override NetworkVector Output { get { return _outputs; } }
        public LayerState State { get { return  new LayerState(_weights, _biases); } }
        #endregion

        #region Constructors
        public Layer(double[,] weights, double[] biases = null)
        {
            _neuralFunction = null;
            _weights = new NetworkMatrix(weights);

            _inputs = new NetworkVector(NumberOfInputs);
            _activations = new NetworkVector(NumberOfOutputs);
            _outputs = new NetworkVector(NumberOfOutputs);
            _inputGradient = new NetworkVector(NumberOfInputs);

            if (biases != null)
            {
                if (biases.Length != NumberOfOutputs)
                    throw new ArgumentException("Length of biases must the the same as the number of neurons.");
                _biases = new NetworkVector(biases);
            }
            else
            {
                _biases = new NetworkVector(NumberOfOutputs);
            }
        }

        public Layer(double[,] weights, ActivationFunction activationfunction, DerivativeFunction derivativefunction, double[] biases = null)
            : this(weights, biases)
        {
            if (activationfunction != null && derivativefunction == null)
                throw new ArgumentException("derivativefunction cannot be null, if activatioin is not null");

            _neuralFunction = activationfunction;
            _neuralFunctionDerivative = derivativefunction;
        }
        #endregion

        #region public methods
        public override void Run(NetworkVector inputvalues)
        {
            _inputs = inputvalues;
            _activations = _biases.SumWith(_weights.LeftMultiply(_inputs));
            if (_neuralFunction != null)
            {
                _outputs = NetworkVector.ApplyFunctionComponentWise(_activations, x => _neuralFunction(x));
            }
            else
            {
                _outputs = _activations.Copy();
            }
        }

        public override void BackPropagate(NetworkVector outputgradient)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");
            
            NetworkVector activationGradient = _getActivationGradient(outputgradient);
            NetworkMatrix weightsGradient = _getWeightsGradient(activationGradient);
            NetworkVector biasesGradient = _getBiasessGradient(activationGradient);

            _setInputGradient(activationGradient);
            _updateWeights(weightsGradient);
            _updateBiases(biasesGradient);
        }

        #endregion
        


        #region private methods for back propagations
        private NetworkVector _getActivationGradient(NetworkVector outputgradient)
        {
            NetworkVector activationGradient = outputgradient.Copy();
            if (_neuralFunctionDerivative != null)
                activationGradient = NetworkVector.ApplyFunctionComponentWise(_activations, _outputs, (x, y) => _neuralFunctionDerivative(x, y));
            return activationGradient;
        }

        private NetworkMatrix _getWeightsGradient(NetworkVector activationGradient)
        {
            return activationGradient.LeftMultiply(_inputs);
        }

        private NetworkVector _getBiasessGradient(NetworkVector activationGradient)
        {
            return activationGradient.Copy();
        }

        private void _setInputGradient(NetworkVector activationGradient)
        {
            _inputGradient = _weights.DotWithWeightsPerInput(activationGradient);
        }

        private void _updateWeights(NetworkMatrix matrixToSubtract)
        {
            _weights.Subtract(matrixToSubtract);
        }

        private void _updateBiases(NetworkVector vectorToSubtract)
        {
            _biases.Subtract(vectorToSubtract);
        }
        #endregion

    }
}
