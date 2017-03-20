using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class NeuralFunction : NetComponent
    {
        public static double __Logistic(double input)
        {
            return 1.0 / (1 + Math.Exp(-input));
        }

        public static double __LogisticDerivative(double input, double output)
        {
            return output * (1 - output);
        }

        #region protected attributes
        protected int _numberOfUnits;
        protected NetworkVector _input;
        protected NetworkVector _output;
        protected ActivationFunction _neuralFunction;
        protected DerivativeFunction _neuralFunctionDerivative;
        #endregion

        #region NetworkComponent overrides
        public override int NumberOfInputs { get { return _numberOfUnits; } }
        public override int NumberOfOutputs { get { return _numberOfUnits; } }
        public override NetworkVector Input { get { return _input; } protected set { _input = value; } }
        public override NetworkVector Output { get { return _output; } protected set { _output = value; } }


        public override void Run(NetworkVector inputvalues)
        {
            Input = inputvalues;
            if (_neuralFunction != null)
            {
                Output =  NetworkVector.ApplyFunctionComponentWise(Input, x => _neuralFunction(x));
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

        #region public properties
        public NetworkVector Activations { get; protected set; }
        #endregion

        #region Constructors
        public NeuralFunction(int numberofunits)
        {
            _numberOfUnits = numberofunits;
            _neuralFunction = null;
            _input = new NetworkVector(numberofunits);
            _output = new NetworkVector(numberofunits);
            Activations = new NetworkVector(numberofunits);
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
