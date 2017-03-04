using System;

namespace NeuralNet
{

    public class Layer
    {
        public delegate double ActivationFunction(double summedInput);
        public delegate double DerivativeFunction(double input, double output);

        protected double[] _inputs;
        protected double[,] _weights;
        protected double[] _biases;
        protected double[] _activations;
        protected double[] _outputs;
        protected ActivationFunction _neuralFunction;
        protected DerivativeFunction _neuralFunctionDerivative;
        protected double[] _inputGradient;

        public int NumberOfNeurons { get { return _weights.GetLength(0); } }
        public int NumberOfInputs { get { return _weights.GetLength(1); } }
        public double[,] Weights { get { return _weights; } }
        public double[] InputGradient { get { return _inputGradient; } }
        public double[] Output { get { return _outputs; } }


        #region Constructors
        public Layer(double[,] weights, double[] biases = null)
        {
            _neuralFunction = null;
            _weights = weights;
            _inputs = new double[NumberOfInputs];
            _activations = new double[NumberOfNeurons];
            _outputs = new double[NumberOfNeurons];
            _inputGradient = new double[NumberOfInputs];

            if (biases != null)
            {
                if (biases.Length != NumberOfNeurons)
                    throw new ArgumentException("Length of biases must the the same as the number of neurons.");
                _biases = biases;
            }
            else
            {
                _biases = new double[NumberOfNeurons];
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
        public double[] Run(double[] inputvalues)
        {
            _inputs = inputvalues;
            double[] sum = _multiplyByWeights(inputvalues);
            _activations = _addBiasesTo(sum);
            if (_neuralFunction != null)
            {
                for (int i = 0; i < _activations.Length; i++)
                {
                    _outputs[i] = _neuralFunction(_activations[i]);
                }
            }
            else
            {
                _outputs = _activations;
            }
            return _outputs;
        }

        public void BackPropagate(double[] outputgradient)
        {
            if (outputgradient == null || outputgradient.Length != NumberOfNeurons)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");

            double[] activationGradient;
            double[,] weightsGradient;

            activationGradient = _getActivationGradient(outputgradient);
            _setInputGradient(activationGradient);
            weightsGradient = _getWeightsGradient(activationGradient);
            _addToWeights(weightsGradient);
        }

        #endregion

        #region private methods for back propagations
        private double[] _getActivationGradient(double[] outputgradient)
        {
            double[] activationGradient = new double[NumberOfNeurons];
            for (int i = 0; i < NumberOfNeurons; i++)
            {
                activationGradient[i] = outputgradient[i];
                if (_neuralFunctionDerivative != null)
                    activationGradient[i] *= _neuralFunctionDerivative(_activations[i], _outputs[i]);
            }
            return activationGradient;
        }

        private double[,] _getWeightsGradient(double[] activationGradient)
        {
            double[,] weightsGradient = new double[NumberOfNeurons, NumberOfInputs];
            for (int i = 0; i < NumberOfNeurons; i++)
            {
                for (int j = 0; j < NumberOfInputs; j++)
                {
                    weightsGradient[i, j] = _inputs[j] * activationGradient[i];
                }
            }
            return weightsGradient;
        }

        private void _setInputGradient(double[] activationGradient)
        {
            for (int i = 0; i < NumberOfInputs; i++)
            {
                _inputGradient[i] = 0.0;
                for (int j = 0; j < NumberOfNeurons; j++)
                {
                    _inputGradient[i] += _weights[j, i] * activationGradient[j];
                }
            }
        }
        #endregion


        #region private methods for forward run
        private double[] _multiplyByWeights(double[] vector)
        {
            int numberOfNeurons = NumberOfNeurons;
            double sum;
            double[] result = new double[numberOfNeurons];
            for (int i = 0; i < numberOfNeurons; i++)
            {
                sum = 0;
                for (int j= 0; j < vector.Length; j++)
                {
                    sum += _weights[i, j] * vector[j];
                }
                result[i] = sum;
            }

            return result;
        }

        private double[] _addBiasesTo(double[] vectorToAdd)
        {
            double[] result = new double[NumberOfNeurons];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = vectorToAdd[i] + _biases[i];
            }

            return result;
        }

        private void _addToWeights(double[,] matrixToAdd)
        {
            for (int i = 0; i < NumberOfNeurons; i++)
            {
                for (int j = 0; j < NumberOfInputs; j++)
                {
                    _weights[i, j] += matrixToAdd[i, j];
                }
            }
        }
        #endregion
    }
}
