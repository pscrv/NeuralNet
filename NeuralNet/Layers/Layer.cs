using System;

namespace NeuralNet
{
    public class LayerState
    {
        public double[,] Weights { get; private set; }
        public double[] Biases { get; private set; }

        public LayerState(NetworkMatrix weights, NetworkVector biases)
        {
            int numberOfNeurons = weights.NumberOfNeurons;
            int numberOfInputs = weights.NumberOfInputs;

            if (biases.Dimension != numberOfNeurons)
                throw new ArgumentException("Invalid layer state: weights and biases do not agree on the number of neurons.");

            Weights = new double[numberOfNeurons, numberOfInputs];
            Biases = new double[numberOfNeurons];

            for (int i = 0; i < numberOfNeurons; i++)
            {
                Biases[i] = biases.Values[i];

                for (int j = 0; j < numberOfInputs; j++)
                {
                    Weights[i, j] = weights.Values[i, j];
                }
            }
        }
    }


    public class Layer
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
        public int NumberOfNeurons { get { return _weights.NumberOfNeurons; } }
        public int NumberOfInputs { get { return _weights.NumberOfInputs; } }
        public NetworkVector InputGradient { get { return _inputGradient; } }
        public NetworkVector Output { get { return _outputs; } }
        public LayerState State { get { return new LayerState(_weights, _biases); } }
        #endregion

        #region Constructors
        public Layer(double[,] weights, double[] biases = null)
        {
            _neuralFunction = null;
            _weights = new NetworkMatrix(weights);

            _inputs = new NetworkVector(NumberOfInputs);
            _activations = new NetworkVector(NumberOfNeurons);
            _outputs = new NetworkVector(NumberOfNeurons);
            _inputGradient = new NetworkVector(NumberOfInputs);

            if (biases != null)
            {
                if (biases.Length != NumberOfNeurons)
                    throw new ArgumentException("Length of biases must the the same as the number of neurons.");
                _biases = new NetworkVector(biases);
            }
            else
            {
                _biases = new NetworkVector(NumberOfNeurons);
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
        public NetworkVector Run(double[] inputvalues)
        {
            _inputs.SetValues(inputvalues);
            NetworkVector sum = _weights.LeftMultiply(_inputs);
            _activations = _biases.SumWith(sum);
            if (_neuralFunction != null)
            {
                _outputs = NetworkVector.ApplyFunctionComponentWise(_activations, x => _neuralFunction(x));
            }
            else
            {
                _outputs = _activations.Copy();
            }
            return _outputs;
        }

        public void BackPropagate(NetworkVector outputgradient)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfNeurons)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");

            NetworkVector activationGradient;
            NetworkMatrix weightsGradient;
            NetworkVector biasesGradient;

            activationGradient = _getActivationGradient(outputgradient);
            weightsGradient = _getWeightsGradient(activationGradient);
            biasesGradient = _getBiasessGradient(activationGradient);

            _setInputGradient(activationGradient);
            _subtractFromWeights(weightsGradient);
            _subtractFromBiases(biasesGradient);
        }

        #endregion




        #region private methods for forward run
        #endregion




        #region private methods for back propagations
        private NetworkVector _getActivationGradient(NetworkVector outputgradient)
        {
            NetworkVector activationGradient = new NetworkVector(NumberOfNeurons);
            activationGradient = outputgradient.Copy();
            if (_neuralFunctionDerivative != null)
                activationGradient = NetworkVector.ApplyFunctionComponentWise(_activations, _outputs, (x, y) => _neuralFunctionDerivative(x, y));
            return activationGradient;
        }

        private NetworkMatrix _getWeightsGradient(NetworkVector activationGradient)
        {
            return _inputs.LeftMultiply(activationGradient);
        }

        private NetworkVector _getBiasessGradient(NetworkVector activationGradient)
        {
            return activationGradient.Copy();
        }

        private void _setInputGradient(NetworkVector activationGradient)
        {
            _inputGradient = _weights.NeuronWiseWeightAndSum(activationGradient);
        }

        private void _subtractFromWeights(NetworkMatrix matrixToSubtract)
        {
            _weights.Subtract(matrixToSubtract);
        }

        private void _subtractFromBiases(NetworkVector vectorToSubtract)
        {
            _biases.Subtract(vectorToSubtract);
        }
        #endregion

    }
}
