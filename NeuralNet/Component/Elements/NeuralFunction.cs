using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet2
{
    public class NeuralFunction : Component
    {
        #region static
        public static double __Logistic(double input)
        {
            return 1.0 / (1.0 + Math.Exp(-input));
        }

        public static double __LogisticDerivative(double input, double output)
        {
            return output * (1 - output);
        }
        #endregion


        #region attributes
        protected int _numberOfUnits;
        protected VectorBatch _input;
        protected VectorBatch _output;
        protected ActivationFunction _neuralFunction;
        protected DerivativeFunction _neuralFunctionDerivative;
        #endregion


        #region Constructors
        public NeuralFunction(int numberOfUnits)
            : this (numberOfUnits, null, null, null)
        { }

        public NeuralFunction(int numberOfUnits, ActivationFunction activationfunction, DerivativeFunction derivativefunction, double[] biases = null)
            : base (numberOfUnits, numberOfUnits)
        {
            if (activationfunction != null && derivativefunction == null)
                throw new ArgumentException("derivativefunction cannot be null, if activation is not null");

            _neuralFunction = activationfunction;
            _neuralFunctionDerivative = derivativefunction;
        }
        #endregion


        #region NetworkComponent overrides
        protected override VectorBatch _run(VectorBatch inputbatch)
        {
            _input = inputbatch;

            if (_neuralFunction != null)
            {
                _output = VectorBatch.ApplyFunction(x => _neuralFunction(x), _input);
            }
            else
            {
                _output = inputbatch;
            }

            return _output;
        }



        protected override VectorBatch _backPropagate(VectorBatch outputGradient)
        {
            if (outputGradient == null || outputGradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");

            if (_neuralFunctionDerivative == null)
                return outputGradient;

            VectorBatch derivative = VectorBatch.ApplyFunction( (x, y) => _neuralFunctionDerivative(x, y), _input, _output);
            VectorBatch result = VectorBatch.ApplyFunction((x, y) => x * y, derivative, outputGradient);
            return result;
        }
        #endregion        
    }
}
