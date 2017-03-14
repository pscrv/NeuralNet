using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.NetComponent
{
    public class NeuralFunction : NetComponent
    {
        #region private attributes
        int _numberOfUnits;
        protected ActivationFunction _neuralFunction;
        protected DerivativeFunction _neuralFunctionDerivative;
        #endregion

        #region NetworkComponent overrides
        public override int NumberOfInputs { get { return _numberOfUnits; } }
        public override int NumberOfOutputs { get { return _numberOfUnits; } }
        public override NetworkVector Output { get; protected set; }


        public override void Run(NetworkVector inputvalues)
        {
            Input = inputvalues;
            if (_neuralFunction != null)
            {
                Output = NetworkVector.ApplyFunctionComponentWise(Input, x => _neuralFunction(x));
            }
            else
            {
                Output = Input.Copy();
            }
        }
        
        public override NetworkVector InputGradient(NetworkVector outputgradient)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");

            if (_neuralFunctionDerivative == null)
                return outputgradient.Copy();
            return NetworkVector.ApplyFunctionComponentWise(Input, Output, (x, y) => _neuralFunctionDerivative(x, y));
            
        }       
        #endregion

        #region pupblic properties
        public NetworkVector Input { get; protected set; }
        public NetworkVector Activations { get; protected set; }
        #endregion

        #region Constructors
        public NeuralFunction(int numberOfUnits)
        {
            _numberOfUnits = numberOfUnits;
            _neuralFunction = null;

            Input = new NetworkVector(numberOfUnits);
            Activations = new NetworkVector(numberOfUnits);
            Output = new NetworkVector(numberOfUnits);
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
    }
}
