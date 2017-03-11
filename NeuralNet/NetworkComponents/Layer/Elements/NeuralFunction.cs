using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{

    public class NeuralFunction : NetworkComponent
    {
        #region private attributes
        int _numberOfUnits;
        NetworkVector _inputs;
        NetworkVector _activations;
        NetworkVector _outputs;
        NetworkVector _inputGradient;
        protected ActivationFunction _neuralFunction;
        protected DerivativeFunction _neuralFunctionDerivative;
        #endregion

        #region public properties
        public override int NumberOfInputs { get { return _numberOfUnits; } }
        public override int NumberOfOutputs { get { return _numberOfUnits; } }
        public override NetworkVector InputGradient { get { return _inputGradient; } }
        public override NetworkVector Output { get { return _outputs; } }
        #endregion

        #region Constructors
        public NeuralFunction(int numberOfUnits)
        {
            _numberOfUnits = numberOfUnits;
            _neuralFunction = null;

            _inputs = new NetworkVector(numberOfUnits);
            _activations = new NetworkVector(numberOfUnits);
            _outputs = new NetworkVector(numberOfUnits);
            _inputGradient = new NetworkVector(numberOfUnits);
        }

        public NeuralFunction(int numberOfUnits, ActivationFunction activationfunction, DerivativeFunction derivativefunction, double[] biases = null)
            : this(numberOfUnits)
        {
            if (activationfunction != null && derivativefunction == null)
                throw new ArgumentException("derivativefunction cannot be null, if activation is not null");

            _neuralFunction = activationfunction;
            _neuralFunctionDerivative = derivativefunction;
        }
        #endregion

        #region public methods
        public override void Run(NetworkVector inputvalues)
        {
            _inputs = inputvalues;
            if (_neuralFunction != null)
            {
                _outputs = NetworkVector.ApplyFunctionComponentWise(_inputs, x => _neuralFunction(x));
            }
            else
            {
                _outputs = _inputs.Copy();
            }
        }

        public override void BackPropagate(NetworkVector outputgradient)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");

            _inputGradient = NetworkVector.ApplyFunctionComponentWise(_inputs, _outputs, (x, y) => _neuralFunctionDerivative(x, y));

        }
        #endregion        
    }
}
